import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import argparse
import random
from kde.utils import kde_log_kernel_cross, safe_x_log_y
from kde.kde_estimators import get_ece_kde
from src.utils import get_calibration_estimators
from src.synthetic import compute_true_calibration_error
from bin_method import get_ece_bin

def set_seed(seed=42):
    """Sets random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def sample_from_simplex(n_classes, size=1):
    """Samples from a n-dimensional simplex"""
    if n_classes == 2:
        u = np.random.rand(size)
        u = np.vstack([1 - u, u]).T
    else:
        u = np.random.rand(size, n_classes - 1)
        u.sort(axis=-1)
        _0s, _1s = np.zeros((size, 1)), np.ones((size, 1))
        u = np.hstack([u, _1s]) - np.hstack([_0s, u])
    return u

def inv_softmax(x, c=None):
    """Inverse softmax with optional temperature scaling"""
    if c is None: c = torch.log(torch.tensor(10.0))
    return torch.log(x) + c

@torch.no_grad()
def sample_points(num_samples, num_classes, temp1, temp2):
    """Samples data points and their corresponding targets"""
    s = sample_from_simplex(num_classes, num_samples)
    base_logits = inv_softmax(torch.tensor(s).float())
    p1 = torch.softmax(base_logits / temp1, dim=1)
    targets = torch.multinomial(p1, 1).squeeze()
    p2 = torch.softmax(base_logits / temp2, dim=1)
    return p1, p2, targets

def style_axes(ax):
    """Customizes axis style for plots"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, ls=':', alpha=0.5, color='#999999')
    ax.tick_params(labelsize=11)

def plot_bw_convergence(results, args, save_path):
    """Plots bandwidth convergence over different methods"""
    bg_color = "#f4f4f8"
    color_map = {
        'MLE-loo': '#4e79a7', 
        'risk-loo': '#f28e2b', 
        'Ground Truth': "#555555", 
        'Perfect': "#adb5bd", 
        'TrueR': "#000000"
    }
    
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=bg_color)
    for name, data in results.items():
        if name == "Equal Bin (20)": continue
        n_axis = np.array(data["N"])
        bw_raw = np.array(data["bw"])  # shape: (runs, ns) or (runs, ns, K)
        
        if bw_raw.ndim == 3:
            bw_raw = np.mean(bw_raw, axis=2)
            
        bw_mean = np.mean(bw_raw, axis=0)
        bw_std = np.std(bw_raw, axis=0)
        
        ax.fill_between(n_axis, bw_mean - bw_std, bw_mean + bw_std, 
                        color=color_map.get(name), alpha=0.15)
        ax.plot(n_axis, bw_mean, marker='s', color=color_map.get(name), 
                lw=2, markersize=7, label=map_method_label(name), markerfacecolor=bg_color)

    ax.set_yscale('log')
    ax.set_xlabel("Sample Size (N)", fontsize=14)
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5, integer=True))
    ax.set_ylabel("Bandwidth $h$", fontsize=14)
    ax.set_title(f"Bandwidth Selection ({args.num_runs} Runs)", fontsize=14, fontweight='bold')
    style_axes(ax)
    ax.legend(fontsize=15)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def compute_ground_truth_curve(f_pool, r_pool, k, num_bins=200):
    """Computes the ground truth curve for class-wise reliability"""
    f_k = f_pool[:, k]
    r_k = r_pool[:, k]
    
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(f_k, bins)
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    true_r_curve = []
    
    for i in range(1, len(bins)):
        mask = (bin_indices == i)
        if mask.any():
            true_r_curve.append(r_k[mask].mean())
        else:
            true_r_curve.append(np.nan)
            
    true_r_curve = np.array(true_r_curve)
    valid_mask = ~np.isnan(true_r_curve)
    
    return bin_centers[valid_mask], true_r_curve[valid_mask]

def map_method_label(name):
    """Academic mapping for method names"""
    mapping = {
        'MLE-loo': 'KDE(MLE)',
        'risk-loo': 'KDE(RA)'
    }
    return mapping.get(name, name)

def plot_rc_curves(f_sub, y_sub, results, true_ce, args, save_path, p1_pool, p2_pool):
    """Plots calibration curves and true calibration error"""
    plt.rcParams['font.family'] = 'serif'
    bg_color = "#f5f5f7"
    color_map = {
        'MLE-loo': "#EAB238", 
        'risk-loo': "#638EE5", 
        'Ground Truth': "#555555", 
        'Perfect': "#adb5bd", 
        'TrueR': "#000000",
        'Equal Bin (20)': "#4FBC47"
    }

    fig1, ax1 = plt.subplots(figsize=(7, 6), facecolor=bg_color)
    ax1.axhline(y=true_ce, color=color_map["Ground Truth"], ls="--", label=r"True $CWCE_2^2$", lw=2)
    
    for name, data in results.items():
        n_axis = np.array(data["N"])
        ce_matrix = np.array(data["ce"])
        ce_mean = np.mean(ce_matrix, axis=0)
        ce_std = np.std(ce_matrix, axis=0)
        ax1.fill_between(n_axis, ce_mean - ce_std, ce_mean + ce_std, color=color_map.get(name), alpha=0.15)
        ax1.plot(n_axis, ce_mean, marker='o', color=color_map.get(name), lw=2, label=map_method_label(name))
    
    style_axes(ax1)
    ax1.set_xlabel(r"Sample size", fontsize=14)
    ax1.set_ylabel(r"$CWCE_2^2$", fontsize=14)
    ax1.legend(fontsize=16)
    plt.tight_layout()
    fig1.savefig(save_path.replace(".png", "_ce_bin_conv.png"), dpi=300)

    f_cpu, y_cpu = f_sub.cpu(), y_sub.cpu()
    num_classes = f_cpu.shape[1]
    num_to_plot = min(num_classes, 4)
    
    fig2, axes = plt.subplots(1, num_to_plot, figsize=(7*num_to_plot, 6), facecolor=bg_color)
    if num_to_plot == 1: axes = [axes]
    
    f_query = torch.linspace(0.001, 0.999, 1000)
    p1_pool_torch = torch.as_tensor(p1_pool)
    p2_pool_torch = torch.as_tensor(p2_pool)

    ce_label = r"per-class $CE_2$"
    for k in range(num_to_plot):
        ax = axes[k]
        ax.set_facecolor(bg_color)
        ax.plot([0, 1], [0, 1], "--", color=color_map["Perfect"], alpha=0.8, zorder=2)
        
        true_r_k = p1_pool_torch[:, k]
        true_f_k = p2_pool_torch[:, k]
        if args.ce_type == "l2":
            true_ce_val = torch.mean((true_r_k - true_f_k).abs()).item()
        else:
            true_ce_val = (safe_x_log_y(true_r_k, true_r_k) - safe_x_log_y(true_r_k, true_f_k) + 
                            safe_x_log_y(1-true_r_k, 1-true_r_k) - safe_x_log_y(1-true_r_k, 1-true_f_k)).mean().item()

        bin_centers_gt, true_r_curve = compute_ground_truth_curve(p2_pool.numpy(), p1_pool.numpy(), k, num_bins=200)
        ax.plot(bin_centers_gt, true_r_curve, color=color_map["TrueR"], lw=3, 
                label=f"True $P(Y|f)$ ($CE_2$={true_ce_val:.4f})", zorder=5)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5, integer=True))
        curr_f, curr_y = f_cpu[:, k:k+1], (y_cpu == k).float()
        bin_ece_val = get_ece_bin(curr_f, (y_cpu == k).long(), n_bins=20, mode="binary", ce_type=args.ce_type, return_square=False).item()
        
        n_bins = 20
        edges = torch.linspace(0.0, 1.0, n_bins + 1)
        bin_width = 1.0 / n_bins
        b = (torch.bucketize(curr_f.squeeze(), edges, right=True) - 1).clamp(0, n_bins - 1)
        bin_counts = torch.bincount(b, minlength=n_bins).float()
        bin_true = torch.bincount(b, weights=curr_y.squeeze(), minlength=n_bins)
        acc_per_bin = (bin_true / bin_counts.clamp(min=1.0)).numpy()
        bin_centers = (edges[:-1] + edges[1:]).numpy() / 2
        valid = (bin_counts > 0).numpy()
        ax.bar(bin_centers[valid], acc_per_bin[valid], width=bin_width, 
               color=color_map.get('Equal Bin (20)'), alpha=0.3, 
               edgecolor=color_map.get('Equal Bin (20)'), label=f"20-bin ({ce_label}={bin_ece_val:.4f})", zorder=1)

        for name, res in results.items():
            if name == "Equal Bin (20)": continue
            bw_all_runs = np.array(res['bw'])
            last_bw_config = bw_all_runs[-1, -1]
            hk = float(last_bw_config[k]) if isinstance(last_bw_config, (list, np.ndarray, torch.Tensor)) else float(last_bw_config)
            if hk <= 0: continue
            
            kde_ce_val = get_ece_kde(curr_f, (y_cpu == k).long(), hk, mode="binary", ce_type=args.ce_type, return_square=False).item()
            log_k = kde_log_kernel_cross(f_query.unsqueeze(1), curr_f, hk)
            r_est_curve = torch.exp(torch.logsumexp(log_k[:, torch.where(curr_y==1)[0]], 1) - 
                                    torch.logsumexp(log_k, 1)).numpy()
            
            ax.plot(f_query.numpy(), r_est_curve, color=color_map.get(name), 
                    lw=3, alpha=0.8, label=f"{map_method_label(name)} ({ce_label}={kde_ce_val:.4f})")

        ax.set_title(f"Class {k} Reliability (N={f_cpu.shape[0]})", fontsize=16, fontweight='bold')
        ax.set_xlabel(r"Predicted Probability $f^{(k)}(x)$", fontsize=16)
        ax.set_ylabel(r"$P(Y=" + str(k) + "|f)$", fontsize=16)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5, integer=True))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        style_axes(ax)
        ax.legend(fontsize=14, loc='upper left', framealpha=0.5)

    plt.tight_layout()
    fig2.savefig(save_path.replace(".png", "_rc.png"), dpi=300)
    plt.close('all')

def main():
    set_seed(42) 
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="classwise")
    parser.add_argument("--ce_type", type=str, default="l2")
    parser.add_argument("--temp2", type=float, default=0.8)
    parser.add_argument("--num_runs", type=int, default=3)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    subset_sizes = [500, 1000, 5000, 10000]
    p1_pool, p2_pool, y_pool = sample_points(200000, 4, 1.0, args.temp2)
    true_ce = compute_true_calibration_error(p1_pool, p2_pool, args.ce_type, args.mode)
    methods = ["MLE-loo", "risk-loo", "Equal Bin (20)"] 
    estimators = get_calibration_estimators([(m, m, None) for m in ["MLE-loo", "risk-loo"]], args.ce_type, args.mode)
    results = {k: {"N": subset_sizes, "ce": [], "bw": []} for k in methods}

    for r in range(args.num_runs):
        print(f"Run {r+1}/{args.num_runs}...")
        for name in results:
            results[name]["ce"].append([]); results[name]["bw"].append([])

        for N in subset_sizes:
            idx = torch.randperm(len(p2_pool))[:N]
            fs, ys = p2_pool[idx].to(device), y_pool[idx].to(device)
            for name, run_fn in estimators.items():
                out = run_fn(fs, ys)
                results[name]["ce"][-1].append(out["ce"])
                results[name]["bw"][-1].append(out["bw"])
            bin_ce = get_ece_bin(fs, ys, n_bins=20, mode=args.mode, ce_type=args.ce_type, adaptive=False)
            results["Equal Bin (20)"]["ce"][-1].append(bin_ce.item())
            results["Equal Bin (20)"]["bw"][-1].append(0) 
    
    os.makedirs("./figs/bandwidth", exist_ok=True)
    base_path = f"./figs/bandwidth/bw_{args.mode}_{args.ce_type}"
    plot_rc_curves(fs, ys, results, true_ce, args, f"{base_path}.png", p1_pool, p2_pool)
    plot_bw_convergence(results, args, f"{base_path}_bw.png")
    print(f"All plots saved to ./figs/bandwidth/")

if __name__ == "__main__":
    main()