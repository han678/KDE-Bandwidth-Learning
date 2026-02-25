import torch
from kde.utils import ratio_binary, ratio_canonical, ratio_classwise, safe_x_log_y, EPS_PROB
from torch import Tensor
from typing import Union

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

def get_ece_kde(f: Tensor, y: Tensor, bw: Union[float, Tensor], mode: str, ce_type: str, return_square: bool = True):
    if f.dim() == 1: f = f.unsqueeze(1)
    N, K = f.shape
    
    if mode == "binary" or K == 1:
        R = ratio_binary(f, y, bw)
        if R.dim() == 1: R = R.unsqueeze(1) 
    elif mode == "canonical":
        R = ratio_canonical(f, y, bw)
    else:
        R = ratio_classwise(f, y, bw)
    
    R = R.clamp(EPS_PROB, 1.0 - EPS_PROB)
    
    if mode == "canonical" and K > 1:
        return canonical_calib_error(R, f, ce_type, return_square=return_square)
    
    f_eval = f[:, -1:] if (mode == "binary" or K == 1) else f
    R_eval = R[:, -1:] if (mode == "binary" or K == 1) else R
    res = sum(calib_error_binary(R_eval[:, k], f_eval[:, k], ce_type) for k in range(R_eval.shape[1]))
    
    if ce_type == "l2" and not return_square:
        return torch.sqrt(res.clamp(min=1e-12))
    return res

def get_refinement_kde(f: Tensor, y: Tensor, bw: Union[float, Tensor], mode: str, ce_type: str):
    N, K = f.shape
    if mode == "binary" or K == 1:
        R = ratio_binary(f, y, bw)
    elif mode == "canonical":
        R = ratio_canonical(f, y, bw)
    else:
        R = ratio_classwise(f, y, bw)
        
    R = R.clamp(EPS_PROB, 1.0 - EPS_PROB)
    
    if mode == "canonical" and K > 1:
        return refinement_canonical(R, ce_type)
    
    R_eval = R[:, -1:] if (mode == "binary" or K == 1) else R
    return sum(refinement_binary(R_eval[:, k], ce_type) for k in range(R_eval.shape[1]))

def get_risk_kde(f: Tensor, y: Tensor, bw: Union[float, Tensor], mode: str, ce_type: str = "kl", 
                 return_components: bool = False, return_mean: bool = True, return_matrix: bool = False):
    N, K = f.shape
    if mode == "binary" or K == 1:
        R = ratio_binary(f, y, bw).view(N, 1)
        f_eval, R_eval = f[:, -1:], R
    else:
        R = ratio_canonical(f, y, bw) if mode == "canonical" else ratio_classwise(f, y, bw)
        f_eval, R_eval = f, R

    R_eval = R_eval.clamp(EPS_PROB, 1.0 - EPS_PROB)
    
    if return_components:
        if mode == "canonical":
            cal_vec = canonical_calib_error(R_eval, f_eval, ce_type)
            ref_vec = refinement_canonical(R_eval, ce_type)
        else:
            cal_vec = sum(calib_error_binary(R_eval[:, k], f_eval[:, k], ce_type) for k in range(R_eval.shape[1]))
            ref_vec = sum(refinement_binary(R_eval[:, k], ce_type) for k in range(R_eval.shape[1]))
        return (cal_vec.mean(), ref_vec.mean()) if return_mean else (cal_vec, ref_vec)
    if ce_type == "l2":
        v_matrix = f_eval.square() - 2.0 * R_eval * f_eval + R_eval
        if mode != "canonical": 
            v_matrix = 2.0 * v_matrix
    else: # kl
        if mode == "canonical":
            v_matrix = -safe_x_log_y(R_eval, f_eval)
        else:
            v_matrix = -(safe_x_log_y(R_eval, f_eval) + safe_x_log_y(1.0 - R_eval, 1.0 - f_eval))

    if return_matrix: 
        return v_matrix
    
    v_sample = v_matrix.sum(dim=1)
    return v_sample.mean() if return_mean else v_sample