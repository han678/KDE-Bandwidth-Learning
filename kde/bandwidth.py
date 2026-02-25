
from kde.kde_estimators import get_risk_kde
from kde.utils import kde_log_kernel, safe_x_log_y, EPS_PROB
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Union

def empirical_risk(probs, targets, ce_type, mode="canonical", return_mean=True, return_matrix=False):
    N, K = probs.shape
    K_effective = 2 if K == 1 else K
    y_ohe = F.one_hot(targets.long(), K_effective).to(probs.dtype)
    probs_safe = probs.clamp(EPS_PROB, 1.0 - EPS_PROB)
    if mode == "binary" or K == 1:
        p, y = probs_safe[:, -1], y_ohe[:, -1]
        if ce_type == "l2":
            v = 2.0 * torch.square(p - y)
        else:
            v = -(safe_x_log_y(y, p) + safe_x_log_y(1.0 - y, 1.0 - p))
        if return_matrix: return v.view(N, 1)
    if ce_type == "l2":
        err_sq_matrix = torch.square(probs_safe - y_ohe)
        if mode != "canonical":
            err_sq_matrix = 2.0 * err_sq_matrix
            
        if return_matrix: return err_sq_matrix
        v = err_sq_matrix.sum(dim=1)
        return v.mean() if return_mean else v
    if mode == "canonical":
        kl_matrix = -safe_x_log_y(y_ohe, probs_safe)
    else:
        kl_matrix = -(safe_x_log_y(y_ohe, probs_safe) + safe_x_log_y(1.0 - y_ohe, 1.0 - probs_safe))
    if return_matrix: return kl_matrix
    v = kl_matrix.sum(dim=1)
    return v.mean() if return_mean else v

def _get_bw_candidates(mode: str, K: int, dev: torch.device, dtype: torch.dtype) -> Tensor:
    if mode == "classwise" or mode == "binary" or K <= 2:
        bws = torch.cat((
            torch.logspace(-4, -1, 60, device=dev, dtype=dtype),
            torch.linspace(0.1, 0.2, 10, device=dev, dtype=dtype)
        ))
    else:
        effective_low = -3
        bws = torch.cat((
            torch.logspace(effective_low, -2, 20, device=dev, dtype=dtype),
            torch.logspace(-2, -1, 40, device=dev, dtype=dtype),
            torch.logspace(-1, -0.7, 30, device=dev, dtype=dtype),
            torch.linspace(0.2, 1.0, 40, device=dev, dtype=dtype)
        ))
    return bws.unique().sort()[0]

def select_bandwidth(
    f: Tensor, y: Tensor, mode: str, ce_type: str, method: str = "risk-loo"
) -> Union[float, Tensor]:
    dev, dtype = f.device, f.dtype
    N, K = f.shape
    candidates = _get_bw_candidates(mode, K, dev, dtype)

    def _run_search(f_in: Tensor, y_in: Tensor, m_in: str) -> float:
        if method == "MLE-loo":
            return _search_mle_loo(f_in, candidates)
        elif method == "risk-loo":
            return _search_risk_loo(f_in, y_in, m_in, ce_type, candidates)
        else:
            raise ValueError(f"Unknown bandwidth selection method: {method}")

    if mode == "classwise":
        best_bws = torch.zeros(K, device=dev, dtype=dtype)
        for k in range(K):
            best_bws[k] = _run_search(f[:, k:k+1], (y == k).long(), "binary")
        return best_bws
    return _run_search(f, y, mode)

@torch.no_grad()
def _search_risk_loo(f: Tensor, y: Tensor, mode: str, ce_type: str, candidates: Tensor) -> float:
    N, K = f.shape
    f_safe = f.clamp(EPS_PROB, 1.0 - EPS_PROB)
    target_matrix = empirical_risk(f_safe, y, ce_type, mode=mode, return_mean=False, return_matrix=True)
    best_loss, best_bw = float('inf'), candidates[0].item()
    for bw in candidates:
        v_matrix = get_risk_kde(f, y, bw, mode, ce_type, return_mean=False, return_matrix=True)
        if mode == "canonical":
            v_sum = v_matrix.sum(dim=1)
            t_sum = target_matrix.sum(dim=1)
            loss = torch.sum((v_sum - t_sum)**2).item()
        else:
            loss = torch.sum((v_matrix - target_matrix)**2).item()
            
        if loss < best_loss:
            best_loss, best_bw = loss, bw.item()
    return best_bw

def _search_mle_loo(f: Tensor, candidates: Tensor) -> float:
    """Maximize the Leave-One-Out Log-Likelihood using the underlying kernel."""
    N = f.size(0)
    log_n_minus_1 = torch.log(torch.tensor(N - 1, device=f.device, dtype=f.dtype))
    best_ll, best_bw = -float('inf'), candidates[0]

    for bw in candidates:
        # both 1D (Beta) and MD (Dirichlet)
        log_k = kde_log_kernel(f, bw, leave_one_out=True)
        total_ll = (torch.logsumexp(log_k, dim=1) - log_n_minus_1).sum().item()
            
        if total_ll > best_ll:
            best_ll, best_bw = total_ll, bw
    return best_bw.item()




