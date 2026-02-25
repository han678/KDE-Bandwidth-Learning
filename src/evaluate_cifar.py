import sys
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_path)
from kde.utils import EPS_PROB
from src.utils import get_calibration_estimators
from src.synthetic import map_method_label, label_cfg

def get_logits_and_labels(preds_dict, logits_key='preds_test', labels_key='gt_test'):
    logits = preds_dict[logits_key]
    labels = preds_dict[labels_key]
    if not isinstance(logits, torch.Tensor): logits = torch.tensor(logits)
    if not isinstance(labels, torch.Tensor): labels = torch.tensor(labels)
    return logits, labels

def load_data_compatible(path_to_root, dataset, model_name, seed, return_train=False):
    folder_pattern = f'{dataset}_{model_name}_*_{seed}_output_{seed}'
    search_path = os.path.join(path_to_root, folder_pattern)
    matched_folders = glob.glob(search_path)
    
    if len(matched_folders) == 0:
        folder_pattern_alt = f'{dataset}_{model_name}_*_{seed}'
        search_path_alt = os.path.join(path_to_root, folder_pattern_alt)
        matched_folders = glob.glob(search_path_alt)
        if len(matched_folders) == 0:
            print(f"[Skip] Folder not found for {dataset} {model_name} seed {seed}")
            return None, None

    root_folder = matched_folders[0]
    cache_matches = glob.glob(os.path.join(root_folder, '**', '*cache*'), recursive=True)
    if not cache_matches: return None, None
    cache_folder = cache_matches[0]
    
    pred_files = glob.glob(os.path.join(cache_folder, 'preds_*.pth'))
    if not pred_files: return None, None
    pred_path = sorted(pred_files)[-1] 
    
    print(f"Loading: {pred_path}")

    preds_dict = torch.load(pred_path, map_location='cpu')
    if return_train:
        return get_logits_and_labels(preds_dict, logits_key='preds_train', labels_key='gt_train')
    else:
        return get_logits_and_labels(preds_dict, logits_key='preds_test', labels_key='gt_test')

def map_model_name(model_name):
    mapping = {
        "PreResNet20": "PreResNet-20",
        "PreResNet56": "PreResNet-56",
        "VGG16BN": "VGG-16 (BN)",
        "WideResNet28x10": "WRN-28-10",
        "vit_small_patch16_224": "ViT-Small",
        "vit_base_patch16_224": "ViT-Base",
        "swin_small_patch4_window7_224": "Swin-Small",
        "deit_base_patch16_224": "DeiT-Base"
    }
    return mapping.get(model_name, model_name.replace("_", " ").title())

@torch.no_grad()
def compute_true_metrics(logits, targets, ce_type="l2", mode="canonical", device="cuda"):
    logits, targets = logits.to(device), targets.to(device)
    probs = F.softmax(logits, dim=1).clamp(EPS_PROB, 1-EPS_PROB)
    K = logits.size(1)

    if ce_type == "kl":
        if mode == "classwise":
            y_ohe = F.one_hot(targets, K).to(probs.dtype)
            true_risk = (-y_ohe * probs.log() - (1-y_ohe) * (1-probs).log()).sum(1).mean().item()
        else:
            true_risk = F.cross_entropy(logits, targets).item()
    else:
        y_ohe = F.one_hot(targets, K).to(probs.dtype)
        true_risk = (probs - y_ohe).pow(2).sum(1).mean().item()
        if mode == "classwise": true_risk *= 2.0
    # if mode == "classwise":
    #     n_bins = 20
    #     true_ce = get_ece_bin(f=probs, y=targets, n_bins=n_bins, mode=mode, ce_type=ce_type, adaptive=False).item()
    # else:
    #     method = "risk-loo"
    method = "krr" if ce_type == "l2" and mode == "canonical" else "risk-loo"
    specs = [(method, method, None)]
    estimators = get_calibration_estimators(specs, ce_type=ce_type, mode=mode)
    for key, run_est in estimators.items():
        out = run_est(probs, targets)
        true_ce = out["ce"]
        true_risk = out["risk"]      
    return true_ce, true_risk
    
def run_estimation_experiment(full_logits, full_targets, subset_sizes, repeats, methods, ce_type="l2", mode="canonical", device="cuda"):
    total_samples = full_logits.shape[0]
    true_ce_bench, true_risk_bench = compute_true_metrics(full_logits, full_targets, ce_type, mode, device)

    method_specs = []
    for m in methods:
        if m == "equal_bin": method_specs.extend([(f"equal_bin_n20", m, 20), (f"equal_bin_n15", m, 15)])
        elif m == "adapt_bin": method_specs.extend([(f"adapt_bin_n15", m, 15), (f"adapt_bin_n10", m, 10)])
        else: method_specs.append((m, m, None))

    estimators = get_calibration_estimators(method_specs, ce_type, mode)
    
    results = {k: {"N": [], "ce": [], "ce_std": [], "ce_mae": [], "ce_mae_std": [], "risk": [], "risk_std": [], "risk_mae": [], "bw": []} for k in estimators}
    results.update({"true_ce": true_ce_bench, "true_risk": true_risk_bench, "repeats": repeats, "_method_order": list(estimators.keys())})

    for N in subset_sizes:
        if N > total_samples: continue
        batch = {k: {"ce": [], "risk": [], "bw": []} for k in estimators}
        for _ in range(repeats):
            idx = torch.randperm(total_samples)[:N]
            sub_f = F.softmax(full_logits[idx], dim=1).clamp(EPS_PROB, 1-EPS_PROB).to(device)
            sub_y = full_targets[idx].to(device)
            for key, run_fn in estimators.items():
                out = run_fn(sub_f, sub_y)
                batch[key]["ce"].append(out["ce"])
                batch[key]["risk"].append(out["risk"])
                batch[key]["bw"].append(np.mean(out["bw"]) if isinstance(out["bw"], list) else out["bw"])
        
        for k in estimators:
            ce_vals, risk_vals = np.array(batch[k]["ce"]), np.array(batch[k]["risk"])
            abs_ce_errors = np.abs(ce_vals - true_ce_bench)
            res = results[k]
            res["N"].append(N)
            res["ce"].append(ce_vals.mean()); res["ce_std"].append(ce_vals.std())
            res["ce_mae"].append(abs_ce_errors.mean())
            res["ce_mae_std"].append(abs_ce_errors.std())
            res["risk"].append(risk_vals.mean()); res["risk_std"].append(risk_vals.std())
            res["risk_mae"].append(np.abs(risk_vals - true_risk_bench).mean())
            res["bw"].append(np.mean(batch[k]["bw"]))
    return results


def plot_comparison(all_results, metric, ce_type="l2", mode="canonical", save_path=None, interval="ci95"):
    if not all_results:
        print("[Warning] No results found to plot. Skipping...")
        return
        
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.unicode_minus'] = False
    bg_color = "#f5f5f7"
    colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#8c564b', '#b07aa1', '#ff9da7']
    
    print(label_cfg.get(ce_type, label_cfg[ce_type]))
    cfg = label_cfg.get(ce_type, label_cfg[ce_type])[mode]
    risk_name = {"kl": "Log Loss (KL)", "l2": "Brier Score ($L_2$)"}.get(ce_type, "Risk")
    risk_name = f"Classwise {risk_name}" if mode == "classwise" else f"Canonical {risk_name}"

    models = all_results.keys()
    num_models = len(models)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 6), facecolor=bg_color)
    if num_models == 1: axes = [axes]
    
    legend_data = [] 

    for idx, name in enumerate(models):
        ax = axes[idx]
        res = all_results[name]
        n_reps = res.get("repeats", 1)
        ax.set_facecolor(bg_color)
        ax.grid(True, color='white', lw=1.2, zorder=0)
        
        if metric == "ce":
            gt_val = res.get("true_ce", 0.0)
            l_gt = ax.axhline(y=gt_val, ls='--', color="#555555", lw=2, zorder=4, label="Full-Data")
            if idx == 0: legend_data.append((l_gt, "Full-Data"))
        elif "mae" in metric:
            l_ideal = ax.axhline(y=0.0, ls=':', color="black", lw=2, zorder=4)
            if idx == 0: legend_data.append((l_ideal, "Ideal (0)"))
        elif "risk" in metric:
            gt_risk = res.get("true_risk", 0.0)
            l_risk = ax.axhline(y=gt_risk, ls='--', color="black", lw=1.5, zorder=4)
            if idx == 0: legend_data.append((l_risk, "Full-Data Risk"))

        method_order = res.get("_method_order", [])
        for i, m in enumerate(method_order):
            r = res[m]
            if metric not in r or len(r["N"]) == 0: continue
            
            x = np.array(r["N"])
            y = np.array(r[metric])
            sort_idx = np.argsort(x)
            x_s, y_s = x[sort_idx], y[sort_idx]
            
            color = colors[i % len(colors)]
            is_key_method = "risk" in m.lower() or "krr" in m.lower()
            lw = 3.5 if is_key_method else 2
            zo = 14 if "mle" in m.lower() else 3
            line, = ax.plot(x_s, y_s, "o-", lw=lw, color=color, markersize=4, zorder=zo)
            if idx == 0:
                legend_data.append((line, map_method_label(m)))
            std_key = f"{metric}_std"
            if interval == "ci95" and std_key in r:
                s = np.array(r[std_key])[sort_idx]
                factor = 1.96 / np.sqrt(n_reps)
                ax.fill_between(x_s, y_s - s * factor, y_s + s * factor, 
                                 color=color, alpha=0.15, lw=0, zorder=2)

        ax.set_title(f"{map_model_name(name)}", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Number of samples (N)", fontsize=15)
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5, integer=True))
        
        if idx == 0:
            if "risk" in metric:
                base_y = cfg['risk']
            else:
                base_y = cfg['tex']
            
            ylabel = f"MAE of {base_y}" if "mae" in metric else base_y
            ax.set_ylabel(ylabel, fontsize=15)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=12)

    unique_handles, unique_labels = [], []
    seen = set()
    for h, l in legend_data:
        if l not in seen:
            seen.add(l); unique_handles.append(h); unique_labels.append(l)

    fig.legend(unique_handles, unique_labels, 
               loc='lower center', ncol=len(unique_labels), 
               bbox_to_anchor=(0.5, -0.034), frameon=False, fontsize=15)
    
    plt.tight_layout(rect=[0, 0.035, 1, 1])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=250, bbox_inches="tight", facecolor=bg_color)
        print(f"Comparison plot saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_trained_models', type=str, default='train_models/cifar_outputs')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--mode', type=str, default='classwise')
    parser.add_argument('--ce_type', type=str, default='l2')
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--model_names', type=str, nargs='+', default=['PreResNet20', "PreResNet56", 'VGG16BN', "WideResNet28x10"])
    parser.add_argument('--n_repeats', type=int, default=5)
    parser.add_argument('--skip_compute', action='store_true', help="Load existing .npy results instead of recomputing")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    subset_sizes = [500, 1000, 3000, 5000, 8000, 10000] 
    result_out_dir = f'results/{args.dataset}/{args.mode}_{args.seed}/'
    os.makedirs(result_out_dir, exist_ok=True)
    
    all_results = {}
    methods = ["MLE-loo", "risk-loo", "equal_bin"]
    if args.mode == "classwise":
        methods += ["adapt_bin"]

    for model_name in args.model_names:
        print(f"\n>>> Evaluating {model_name}...")
        res_path = os.path.join(result_out_dir, f"{model_name}_seed{args.seed}_{args.ce_type}_{args.mode}.npy")
        
        if args.skip_compute and os.path.exists(res_path):
            res = np.load(res_path, allow_pickle=True).item()
        else:
            train_logits, train_labels = load_data_compatible(args.path_to_trained_models, args.dataset, model_name, args.seed, return_train=True)
            test_logits, test_labels = load_data_compatible(args.path_to_trained_models, args.dataset, model_name, args.seed, return_train=False)
            logits = torch.cat([train_logits, test_logits], dim=0)
            labels = torch.cat([train_labels, test_labels], dim=0)
            print(f"Merged Data: Train({train_logits.shape[0]}) + Test({test_logits.shape[0]}) = Total({logits.shape[0]}) samples.")
            if logits is None: continue
            res = run_estimation_experiment(logits, labels, subset_sizes, args.n_repeats, methods, args.ce_type, args.mode, device)
            np.save(res_path, res)
        all_results[model_name] = res

    fig_dir = f"figs/{args.dataset}/{args.mode}_{args.seed}/"
    os.makedirs(fig_dir, exist_ok=True)
    for m in ["ce"]: # , "risk_mae"
        plot_comparison(all_results, m, args.ce_type, mode=args.mode, save_path=os.path.join(fig_dir, f"{args.ce_type}_{args.mode}_{m}.png"))


if __name__ == "__main__":
    main()