import torch
from torch import Tensor
from typing import Union

GLOBAL_NEG_INF = -1e10
GLOBAL_LOG_MIN = -100.0 
GLOBAL_ISLAND_THRESH = -20.0  
EPS_PROB = 1e-9

def _neg_inf() -> float:
    return GLOBAL_NEG_INF

def safe_x_log_y(x: Tensor, y: Tensor) -> Tensor:
    x_safe = x.clamp(min=0.0) 
    y_safe = y.clamp(EPS_PROB, 1.0 - EPS_PROB)
    return torch.xlogy(x_safe, y_safe)

def beta_kernel_logpdf(z: Tensor, bandwidth: Union[float, Tensor]) -> Tensor:
    # z: [N, 1]
    bw = torch.as_tensor(bandwidth, device=z.device, dtype=z.dtype)
    z1 = z[:, 0]
    
    nu = 1.0 / bw
    p = z1 * nu 
    q = (1.0 - z1) * nu 

    z_log = torch.where(z1 > 1e-12, z1.log(), torch.as_tensor(-100.0, device=z.device))
    z_inv_log = torch.where((1.0 - z1) > 1e-12, torch.log1p(-z1), torch.as_tensor(-100.0, device=z.device))

    return (p[None] - 1.0) * z_log[:, None] + (q[None] - 1.0) * z_inv_log[:, None] - (
        torch.lgamma(p)[None] + torch.lgamma(q)[None] - torch.lgamma(nu)[None]
    )


def beta_kernel_logpdf_cross(z: Tensor, zi: Tensor, bandwidth: Union[float, Tensor]) -> Tensor:
    bw = torch.as_tensor(bandwidth, device=zi.device, dtype=zi.dtype)
    p_off = zi / bw  # p-1
    q_off = (1.0 - zi) / bw  # q-1
    z = z.unsqueeze(1)

    term1 = torch.nan_to_num(p_off[None] * z.log(), nan=0.0)
    term2 = torch.nan_to_num(q_off[None] * torch.log1p(-z), nan=0.0)

    res = term1 + term2 - (
        torch.lgamma(p_off[None] + 1.0) + 
        torch.lgamma(q_off[None] + 1.0) - 
        torch.lgamma(p_off[None] + q_off[None] + 2.0)
    )
    return res

def dirichlet_kernel_logpdf(z: Tensor, bandwidth: Union[float, Tensor]) -> Tensor:
    bw = torch.as_tensor(bandwidth, device=z.device, dtype=z.dtype)
    N, K = z.shape
    
    alpha_minus_1 = z / bw 
    z_log = torch.where(z > 1e-15, z.log(), torch.as_tensor(-100.0, device=z.device, dtype=z.dtype))
    
    log_num = z_log @ alpha_minus_1.T 
    sum_alphas = 1.0 / bw + K
    log_beta = torch.lgamma(alpha_minus_1 + 1.0).sum(dim=1) - torch.lgamma(sum_alphas)
    return log_num - log_beta.unsqueeze(0)

def kde_log_kernel(
    f: Tensor,
    bandwidth: Union[float, Tensor],
    leave_one_out: bool = True,
) -> Tensor:
    """
    Build log-kernel matrix [N,N].
      - If K==1: beta kernel 
      - Else:   dirichlet kernel 
    If leave_one_out=True, set diagonal to -inf.
    """
    bw = torch.as_tensor(bandwidth, device=f.device, dtype=f.dtype)

    log_k = beta_kernel_logpdf(f, bw) if f.size(1) == 1 else dirichlet_kernel_logpdf(f, bw)
    if leave_one_out:
        log_k = log_k.clone()
        log_k.fill_diagonal_(_neg_inf())

    return log_k


def ratio_binary(
    probs_1d: Tensor,
    y01: Tensor,
    bandwidth: Union[float, Tensor],
) -> Tensor:
    y01 = y01.to(torch.long).view(-1)
    f = probs_1d.view(-1) 

    log_k = kde_log_kernel(f[:, None], bandwidth, leave_one_out=True)  
    log_den = torch.logsumexp(log_k, dim=1)                        

    idx_pos = (y01 == 1).nonzero(as_tuple=True)[0]
    if idx_pos.numel() == 0:
        return torch.zeros_like(f)
        
    log_num = torch.logsumexp(log_k[:, idx_pos], dim=1)             
    
    valid_mask = log_den > GLOBAL_NEG_INF
    rk_log = torch.full_like(log_den, GLOBAL_NEG_INF)
    rk_log[valid_mask] = (log_num[valid_mask] - log_den[valid_mask]).clamp(max=100.0)
    
    return torch.exp(rk_log)

def ratio_classwise(
    f: Tensor,                   
    y: Tensor,                    
    bandwidth: Union[float, Tensor],
    chunk_size: int = 1024,
) -> Tensor:
    N, K = f.shape
    bw = torch.as_tensor(bandwidth, device=f.device, dtype=f.dtype)
    bw = bw.expand(K) if bw.numel() == 1 else bw
    R = torch.empty_like(f)

    for k in range(K):
        fk = f[:, k:k+1]                        
        yk = (y == k)                                 
        mask_k = torch.where(yk, 0.0, GLOBAL_NEG_INF).to(f.dtype)  

        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            lk = beta_kernel_logpdf_cross(fk[i:end], fk, bw[k])[..., 0]    

            r = torch.arange(end - i, device=f.device)
            lk[r, torch.arange(i, end, device=f.device)] = GLOBAL_NEG_INF  

            log_den = torch.logsumexp(lk, dim=1)
            log_num = torch.logsumexp(lk + mask_k[None, :], dim=1)
            
            valid_mask = log_den > GLOBAL_NEG_INF
            rk_log = torch.full_like(log_den, GLOBAL_NEG_INF)
            rk_log[valid_mask] = (log_num[valid_mask] - log_den[valid_mask]).clamp(max=100.0)
            
            R[i:end, k] = torch.exp(rk_log)
    return R


def ratio_canonical(
    f: Tensor, y: Tensor, bandwidth: Union[float, Tensor], chunk_size: int = 2048
) -> Tensor:
    N, K = f.shape
    y = y.long()
    if y.min() < 0 or y.max() >= K:
        raise ValueError(f"Label out of range: y in [{y.min()}, {y.max()}] but K={K}")

    bw = torch.as_tensor(bandwidth, device=f.device, dtype=f.dtype)
    log_k = kde_log_kernel(f, bw, leave_one_out=True)                    
    log_den = torch.logsumexp(log_k, 1, keepdim=True)                    

    neg_inf = torch.as_tensor(GLOBAL_NEG_INF, device=f.device, dtype=f.dtype)
    R = torch.empty((N, K), device=f.device, dtype=f.dtype)

    for i in range(0, N, chunk_size):
        lk = log_k[i : i + chunk_size]                                  
        ld = log_den[i : i + chunk_size]                               
        
        valid_mask = (ld > GLOBAL_NEG_INF).squeeze(1)                         

        for k in range(K):
            mk = torch.where(y == k, 0.0, neg_inf)                             
            ln = torch.logsumexp(lk + mk, 1, keepdim=True).squeeze(1)            
            
            rk_log = torch.full_like(ln, GLOBAL_NEG_INF)
            curr_ld = ld.squeeze(1)
            rk_log[valid_mask] = (ln[valid_mask] - curr_ld[valid_mask]).clamp(max=100.0)
            
            R[i : i + chunk_size, k] = torch.exp(rk_log)
    R = R / R.sum(dim=1, keepdim=True).clamp(min=1e-9)
    return R


def dirichlet_kernel_logpdf_cross(z_query, z_centers, bandwidth):
    bw = torch.as_tensor(bandwidth, device=z_query.device, dtype=z_query.dtype)
    K = z_query.size(1)
    alpha_offsets = z_centers / bw 
    sum_alphas = (1.0 / bw).sum(-1) + K if bw.ndim > 1 else 1.0 / bw + K
    
    log_beta = torch.lgamma(alpha_offsets + 1.0).sum(1) - torch.lgamma(sum_alphas)
    z_log = torch.where(z_query > 0, z_query.log(), torch.tensor(GLOBAL_NEG_INF, device=z_query.device))
    return z_log @ alpha_offsets.T - log_beta[None]

def kde_log_kernel_cross(
    f_query: Tensor,                  
    f_centers: Tensor,           
    bandwidth: Union[float, Tensor],  
) -> Tensor:
    bw = torch.as_tensor(bandwidth, device=f_query.device, dtype=f_query.dtype)
    return (
        beta_kernel_logpdf_cross(f_query, f_centers, bw)[..., 0]
        if f_query.size(1) == 1
        else dirichlet_kernel_logpdf_cross(f_query, f_centers, bw)
    )

@torch.no_grad()
def ratio_binary_cross(
    f_query: Tensor,              
    y_centers01: Tensor,            
    f_centers: Tensor,             
    bandwidth: Union[float, Tensor],
) -> Tensor:
    fq = f_query.view(-1, 1)
    fc = f_centers.view(-1, 1)
    y01 = y_centers01.long().view(-1)

    lk = kde_log_kernel_cross(fq, fc, bandwidth)          
    log_den = torch.logsumexp(lk, dim=1)                

    idx_pos = (y01 == 1).nonzero(as_tuple=True)[0]
    if idx_pos.numel() == 0:
        return torch.zeros((fq.size(0),), device=fq.device, dtype=fq.dtype)

    log_num = torch.logsumexp(lk[:, idx_pos], dim=1)     

    valid = log_den > GLOBAL_NEG_INF
    out_log = torch.full_like(log_den, GLOBAL_NEG_INF)
    out_log[valid] = (log_num[valid] - log_den[valid]).clamp(min=GLOBAL_LOG_MIN, max=100.0)
    return torch.exp(out_log)


@torch.no_grad()
def ratio_canonical_cross(
    f_query: Tensor,                
    y_centers: Tensor,             
    f_centers: Tensor,              
    bandwidth: Union[float, Tensor],
    chunk_size: int = 2048,
) -> Tensor:
    Nq, K = f_query.shape
    y_centers = y_centers.long().view(-1)
    neg_inf = torch.as_tensor(GLOBAL_NEG_INF, device=f_query.device, dtype=f_query.dtype)

    R = torch.empty((Nq, K), device=f_query.device, dtype=f_query.dtype)

    for i in range(0, Nq, chunk_size):
        fq = f_query[i:i + chunk_size]                         
        lk = kde_log_kernel_cross(fq, f_centers, bandwidth)      
        log_den = torch.logsumexp(lk, dim=1, keepdim=True)     
        valid = (log_den.squeeze(1) > GLOBAL_NEG_INF)

        for k in range(K):
            mk = torch.where(y_centers == k, 0.0, neg_inf)       
            log_num = torch.logsumexp(lk + mk.unsqueeze(0), dim=1)  
            out_log = torch.full_like(log_num, GLOBAL_NEG_INF)
            ld = log_den.squeeze(1)
            out_log[valid] = (log_num[valid] - ld[valid]).clamp(min=GLOBAL_LOG_MIN, max=100.0)
            R[i:i + chunk_size, k] = torch.exp(out_log)

    return R

@torch.no_grad()
def ratio_classwise_cross(
    f_query: torch.Tensor,      
    f_centers: torch.Tensor,   
    y_centers: torch.Tensor,    
    bw,                         
):
    Nq, K = f_query.shape
    bw_vec = bw if torch.is_tensor(bw) else torch.full((K,), float(bw), device=f_query.device, dtype=f_query.dtype)
    if torch.is_tensor(bw) and bw.ndim == 0:
        bw_vec = bw.repeat(K)

    R = []
    for k in range(K):
        yk = (y_centers.long() == k).long()       
        fk_q = f_query[:, k:k+1].contiguous()       
        fk_c = f_centers[:, k:k+1].contiguous()     
        Rk = ratio_binary_cross(fk_q, yk, fk_c, bw_vec[k]).view(Nq, 1)  
        R.append(Rk)
    return torch.cat(R, dim=1) 