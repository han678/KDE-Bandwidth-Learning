import torch
import torch.nn.functional as F
from torch import Tensor
from kde.utils import safe_x_log_y, EPS_PROB


def canonical_calib_error(R: Tensor, f: Tensor, ce_type: str) -> Tensor:
    if ce_type == "l2":
        return torch.square(R - f).sum(dim=1).mean()
    kl_div = safe_x_log_y(R, R) - safe_x_log_y(R, f)
    return kl_div.sum(dim=1).mean()

def calib_error_binary(R: Tensor, f: Tensor, ce_type: str) -> Tensor:
    R, f = R.flatten(), f.flatten()
    if ce_type == "l2":
        return (2.0 * torch.square(R - f)).mean()
    kl_pos = safe_x_log_y(R, R) - safe_x_log_y(R, f)
    kl_neg = safe_x_log_y(1.0 - R, 1.0 - R) - safe_x_log_y(1.0 - R, 1.0 - f)
    return (kl_pos + kl_neg).mean()

def refinement_canonical(R: Tensor, ce_type: str) -> Tensor:
    if ce_type == "l2":
        return (1.0 - torch.square(R).sum(dim=1)).mean()
    return -safe_x_log_y(R, R).sum(dim=1).mean()

def refinement_binary(R: Tensor, ce_type: str) -> Tensor:
    R = R.flatten()
    if ce_type == "l2":
        return (2.0 * R * (1.0 - R)).mean()
    ent_pos = safe_x_log_y(R, R)
    ent_neg = safe_x_log_y(1.0 - R, 1.0 - R)
    return -(ent_pos + ent_neg).mean()

def get_canonical_bin_params(f: Tensor, y: Tensor, n_bins: int):
    N, K = f.shape
    device = f.device
    y_ohe = F.one_hot(y.long(), num_classes=K).to(f.dtype)

    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=device)
    bin_indices = (torch.bucketize(f.contiguous(), edges, right=True) - 1).clamp(0, n_bins - 1)
    _, bin_ids, counts = torch.unique(bin_indices, dim=0, return_inverse=True, return_counts=True)
    num_active_bins = counts.shape[0]

    bin_sum_y = torch.zeros((num_active_bins, K), device=device, dtype=f.dtype)
    bin_sum_y.index_add_(0, bin_ids, y_ohe)

    sum_y_all = bin_sum_y[bin_ids]
    m_all = counts[bin_ids].view(N, 1)

    return sum_y_all, m_all

def get_bin_ratio(f: Tensor, y: Tensor, n_bins: int, adaptive: bool) -> Tensor:
    N, K = f.shape

    def _edges(x: Tensor) -> Tensor:
        if adaptive:
            xs, _ = torch.sort(x)
            idx = torch.linspace(0, N - 1, n_bins + 1, device=x.device).long()
            return xs[idx]
        return torch.linspace(0.0, 1.0, n_bins + 1, device=x.device)

    def _bin_mean(x: Tensor, w: Tensor) -> Tensor:
        x = x.clamp(0.0, 1.0)
        edges = _edges(x)
        b = (torch.bucketize(x, edges, right=True) - 1).clamp(0, n_bins - 1)
        cnt = torch.bincount(b, minlength=n_bins).to(f.dtype).clamp_min(1.0)
        s = torch.bincount(b, weights=w.to(f.dtype), minlength=n_bins)
        return (s / cnt)[b]

    if K == 1:
        return _bin_mean(f[:, 0], y.long().view(-1)).view(N, 1)

    y_ohe = F.one_hot(y.long(), num_classes=K).to(f.dtype)
    R = torch.stack([_bin_mean(f[:, k], y_ohe[:, k]) for k in range(K)], dim=1)
    return R

def get_ece_bin(f: Tensor, y: Tensor, n_bins: int, mode: str, ce_type: str, adaptive: bool = False, return_square: bool = True):
    N, K = f.shape
    
    if mode == "canonical" and K > 1:
        sum_y_all, m_all = get_canonical_bin_params(f, y, n_bins)
        valid_mask = (m_all > 1).squeeze()
        if not valid_mask.any(): return torch.tensor(0.0, device=f.device)
        
        y_ohe = F.one_hot(y.long(), num_classes=K).to(f.dtype)
        R = (sum_y_all[valid_mask] - y_ohe[valid_mask]) / (m_all[valid_mask] - 1)
        f_eval = f[valid_mask]
        return canonical_calib_error(R.clamp(EPS_PROB, 1.0 - EPS_PROB), f_eval, ce_type)
    
    R = get_bin_ratio(f, y, n_bins, adaptive).clamp(EPS_PROB, 1.0 - EPS_PROB)
    f_eval = f[:, -1:] if (mode == "binary" or K == 1) else f
    R_eval = R[:, -1:] if (mode == "binary" or K == 1) else R
    res = sum(calib_error_binary(R_eval[:, k], f_eval[:, k], ce_type) for k in range(R_eval.shape[1]))
    
    if ce_type == "l2" and not return_square:
        return torch.sqrt(res.clamp(min=1e-12))
    return res

def get_refinement_bin(f: Tensor, y: Tensor, n_bins: int, mode: str, ce_type: str, adaptive: bool = False):
    N, K = f.shape
    
    if mode == "canonical" and K > 1:
        sum_y_all, m_all = get_canonical_bin_params(f, y, n_bins)
        valid_mask = (m_all > 1).squeeze()
        if not valid_mask.any(): return torch.tensor(0.0, device=f.device)
        
        y_ohe = F.one_hot(y.long(), num_classes=K).to(f.dtype)
        R = (sum_y_all[valid_mask] - y_ohe[valid_mask]) / (m_all[valid_mask] - 1)
        return refinement_canonical(R.clamp(EPS_PROB, 1.0 - EPS_PROB), ce_type)

    R = get_bin_ratio(f, y, n_bins, adaptive).clamp(EPS_PROB, 1.0 - EPS_PROB)
    R_eval = R[:, -1:] if (mode == "binary" or K == 1) else R
    return sum(refinement_binary(R_eval[:, k], ce_type) for k in range(R_eval.shape[1]))

def get_risk_bin(f: Tensor, y: Tensor, n_bins: int, mode: str, ce_type: str = "kl", 
                 adaptive: bool = False, return_components: bool = False):
    N, K = f.shape

    if mode == "canonical" and K > 1:
        sum_y_all, m_all = get_canonical_bin_params(f, y, n_bins)
        valid_mask = (m_all > 1).squeeze()
        
        if not valid_mask.any():
            res = (torch.tensor(0.0), torch.tensor(0.0)) if return_components else torch.tensor(0.0)
            return res
        
        y_ohe = F.one_hot(y.long(), num_classes=K).to(f.dtype)
        R = (sum_y_all[valid_mask] - y_ohe[valid_mask]) / (m_all[valid_mask] - 1)
        R = R.clamp(EPS_PROB, 1.0 - EPS_PROB)
        f_eval = f[valid_mask]
        
        cal = canonical_calib_error(R, f_eval, ce_type)
        ref = refinement_canonical(R, ce_type)
        if return_components: return cal, ref
        return cal + ref

    R = get_bin_ratio(f, y, n_bins, adaptive).clamp(EPS_PROB, 1.0 - EPS_PROB)
    f_eval = f[:, -1:] if (mode == "binary" or K == 1) else f
    R_eval = R[:, -1:] if (mode == "binary" or K == 1) else R

    cal = sum(calib_error_binary(R_eval[:, k], f_eval[:, k], ce_type) for k in range(R_eval.shape[1]))
    ref = sum(refinement_binary(R_eval[:, k], ce_type) for k in range(R_eval.shape[1]))

    if return_components: return cal, ref

    if ce_type == "l2":
        v = (f_eval.square() - 2.0 * R_eval * f_eval + R_eval).sum(dim=1).mean()
        return v if mode == "canonical" else 2.0 * v

    return cal + ref
