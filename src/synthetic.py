import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import argparse
import sys


root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_path)

from kde.utils import EPS_PROB
from kde.bandwidth import empirical_risk
from src.utils import get_calibration_estimators, print_experiment_results

def sample_from_simplex(n_classes, size=1):
    if n_classes == 2:
        u = np.random.rand(size)
        u = np.vstack([1 - u, u]).T
    else:
        u = np.random.rand(size, n_classes - 1)
        u.sort(axis=-1)
        _0s = np.zeros(shape=(size, 1))
        _1s = np.ones(shape=(size, 1))
        u = np.hstack([u, _1s]) - np.hstack([_0s, u])
    return u.flatten() if size == 1 else u

def temp_scale(scores, temperature):
    logits = torch.log(scores) + torch.log(torch.tensor(10.0))
    return logits / temperature

@torch.no_grad()
def sample_points(num_samples, num_classes, temp1, temp2):
    samples, labels = [], []
    for _ in range(num_samples):
        s = sample_from_simplex(num_classes, 1)
        logit = temp_scale(torch.tensor(s), temp1).unsqueeze(0)
        scores = torch.softmax(logit, dim=1)[0]
        samples.append(scores.numpy())
        labels.append(np.random.choice(num_classes, p=scores.numpy()))

    p1 = torch.tensor(np.array(samples)).float()
    targets = torch.tensor(np.array(labels).astype('int64'))
    p2 = torch.softmax(temp_scale(p1, temp2), dim=1)
    return (p1, p2, targets)

@torch.no_grad()
def compute_true_calibration_error(true_p, pred_p, ce_type="kl", mode="canonical"):
    t = true_p.to(torch.float64)
    p = pred_p.to(torch.float64)
    p_safe = p.clamp(EPS_PROB, 1.0 - EPS_PROB)
    t_safe = t.clamp(EPS_PROB, 1.0 - EPS_PROB)
    if mode == "binary":
        if t.size(1) == 1: t_safe = torch.cat([1.0 - t_safe, t_safe], dim=1)
        if p.size(1) == 1: p_safe = torch.cat([1.0 - p_safe, p_safe], dim=1)
        mode = "canonical"
    if mode == "canonical":
        if ce_type == "kl":
            kl = torch.xlogy(t_safe, t_safe) - torch.xlogy(t_safe, p_safe)
            return torch.nan_to_num(kl.sum(dim=1).mean(), nan=0.0).item()
        return torch.sum((t_safe - p_safe) ** 2, dim=1).mean().item()
    elif mode == "classwise":
        if ce_type == "kl":
            kl_pos = torch.xlogy(t_safe, t_safe) - torch.xlogy(t_safe, p_safe)
            one_minus_t = 1.0 - t_safe
            one_minus_p = 1.0 - p_safe
            kl_neg = torch.xlogy(one_minus_t, one_minus_t) - torch.xlogy(one_minus_t, one_minus_p)
            
            res = (kl_pos + kl_neg).sum(dim=1).mean()
            return torch.nan_to_num(res, nan=0.0).item()
            
        return 2.0 * torch.sum((t_safe - p_safe) ** 2, dim=1).mean().item()

def run_synthetic_experiment(pool_size=100000, sizes=(500, 2000, 5000), repeats=5, 
                           num_classes=3, temp1=1.0, temp2=0.6, mode="canonical", 
                           ce_type="l2", methods=("MLE-loo",), device="cuda"):
    torch.manual_seed(40); np.random.seed(40)
    g = torch.Generator(device=device).manual_seed(123)
    p1, p2, targets = sample_points(pool_size, num_classes, temp1, temp2)
    f_p, y_p = p2.to(device), targets.to(device)
    
    true_ce = float(compute_true_calibration_error(p1, p2, ce_type, mode))
    true_risk = empirical_risk(f_p, y_p, ce_type, mode=mode).item()

    method_specs = []
    for m in methods:
        if m == "adapt_bin":
            for nb in [10, 15]: method_specs.append((f"{m}_n{nb}", m, nb))
        elif m == "equal_bin":
            if mode == "classwise":
                for nb in [15, 20]: method_specs.append((f"{m}_n{nb}", m, nb))
            if mode == "canonical":
                for nb in [10, 15]: method_specs.append((f"{m}_n{nb}", m, nb))
        else:
            method_specs.append((m, m, None))

    estimators = get_calibration_estimators(method_specs, ce_type, mode)
    results = {k: {m: [] for m in ["N", "ce", "ce_std", "ce_mae", "ce_mae_std", "risk", "risk_std", "risk_mae", "bw"]} for k in estimators}
    results.update({"true_ce": true_ce, "true_risk": true_risk, "repeats": repeats, "_method_order": list(estimators.keys())})

    for N in sizes:
        batch = {k: {"ce": [], "risk": [], "bw": []} for k in estimators}
        for _ in range(repeats):
            idx = torch.randperm(pool_size, generator=g, device=device)[:N]
            fs, ys = f_p[idx], y_p[idx]
            for key, run_est in estimators.items():
                out = run_est(fs, ys)
                batch[key]["ce"].append(out["ce"])
                batch[key]["risk"].append(out["risk"])
                batch[key]["bw"].append(out.get("bw", 0))

        for key in estimators:
            ce_v, r_v = np.array(batch[key]["ce"]), np.array(batch[key]["risk"])
            res = results[key]
            res["N"].append(N)
            res["ce"].append(ce_v.mean())
            res["ce_mae"].append(np.abs(ce_v - true_ce).mean())
            res["ce_mae_std"].append(np.abs(ce_v - true_ce).std())
            res["ce_std"].append(ce_v.std())
            res["risk"].append(r_v.mean())
            res["risk_mae"].append(np.abs(r_v - true_risk).mean())
            res["risk_std"].append(r_v.std())
            res["bw"].append(np.mean(batch[key]["bw"]))
    return results

def map_method_label(m):
    mapping = {"MLE-loo": "KDE(MLE)", "risk-loo": "KDE(RA)", "equal_bin": "Bin(Equal)", "adapt_bin": "Bin(Adapt)", "krr": "KRR"}
    if "_n" in m:
        base, _, nb = m.partition("_n")
        return f"{mapping.get(base, base)} ({nb})"
    return mapping.get(m, m)


label_cfg = {
    "kl": {
        "canonical": {"tex": r"$CE_{KL}$", "name": "KL Canonical Calib. Error", "risk": "Canonical Log Loss (KL)"},
        "classwise": {"tex": r"$CWCE_{KL}$", "name": "Classwise KL Calib. Error", "risk": "Classwise Log Loss"}
    },
    "l2": {
        "canonical": {"tex": r"$CE_2^2$", "name": "L2 Canonical Calib. Error", "risk": "Canonical Brier Score ($L_2$)"},
        "classwise": {"tex": r"$CWCE_2^2$", "name": "Classwise L2 Calib. Error", "risk": "Classwise Brier Score"}
    }
}


def plot_synthetic_results(all_nc_results, ce_type, mode, save_dir, interval="ci95", plot_mae=False):
    bg_color = "#f5f5f7"
    colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#8c564b', '#b07aa1', '#ff9da7']
    cfg = label_cfg.get(ce_type, label_cfg[ce_type])[mode]
    risk_name = {"kl": "Log Loss (KL)", "l2": "Brier Score ($L_2$)"}.get(ce_type, "Risk")
    risk_name = f"Classwise {risk_name}" if mode == "classwise" else f"Canonical {risk_name}"

    os.makedirs(save_dir, exist_ok=True)
    nc_list = sorted(all_nc_results.keys())

    for metric_type in ["ce"]: # , "risk"
        fig, axes = plt.subplots(1, len(nc_list), figsize=(6 * len(nc_list), 6), facecolor=bg_color)
        axes = [axes] if len(nc_list) == 1 else axes
        legend_data = []

        for idx, nc in enumerate(nc_list):
            ax = axes[idx]
            ax.set_facecolor(bg_color)
            ax.grid(True, color="white", lw=1.2, zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", which="major", labelsize=11)
            res = all_nc_results[nc]
            n_reps = res.get("repeats", 1)
            method_order = res.get("_method_order", [])
            ground_truth = res.get("true_ce", None) if metric_type == "ce" else res.get("true_risk", None)
            if plot_mae:
                gt = ax.axhline(0.0, ls=":", color="black", lw=2, zorder=4)
                if idx == 0:
                    legend_data.append((gt, "Ideal (0)"))
            else:
                l_gt = ax.axhline(y=ground_truth, ls='--', color="#555555", lw=2, zorder=4, label="Ground Truth")
                if idx == 0: legend_data.append((l_gt, "Ground Truth"))

            y_key = f"{metric_type}_mae" if plot_mae else metric_type
            for i, m in enumerate(method_order):
                r = res.get(m, {})
                if y_key not in r or "N" not in r or len(r["N"]) == 0:
                    continue

                x = np.asarray(r["N"])
                y = np.asarray(r[y_key])
                sidx = np.argsort(x)
                x, y = x[sidx], y[sidx]

                color = colors[i % len(colors)]
                lw = 2.5 if "risk" in m.lower() else 1.8
                line, = ax.plot(x, y, "o-", lw=lw, color=color, zorder=3)
                if idx == 0:
                    legend_data.append((line, map_method_label(m)))

                std = None
                for sk in (f"{y_key}_std", f"{metric_type}_std", "std"):
                    if sk in r and len(r[sk]) > 0:
                        std = np.asarray(r[sk])[sidx]
                        break
                if std is not None and interval == "ci95":
                    factor = 1.96 / np.sqrt(max(n_reps, 1))
                    ax.fill_between(x, y - std * factor, y + std * factor,
                                    color=color, alpha=0.15, lw=0, zorder=2)

            if metric_type == "ce":
                ax.set_title(f"Classes: {nc}", fontsize=14, fontweight="bold", pad=15)
                y_label = f"MAE of {cfg['tex']}" if plot_mae else cfg["tex"]
                if idx == 0:
                    ax.set_ylabel(y_label, fontsize=15)
            else:
                y_label = f"MAE of {risk_name}" if plot_mae else risk_name
                ax.set_title(f"Target: {risk_name}\n(Classes: {nc})", fontsize=14, fontweight="bold", pad=15)
                if idx == 0:
                    ax.set_ylabel(y_label, fontsize=15)

            ax.set_xlabel("Number of samples (N)", fontsize=13)
            ax.set_yscale("log")
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5, integer=True))
        if legend_data:
            leg = fig.legend([h for h, _ in legend_data], [t for _, t in legend_data],
                            loc="lower center", ncol=len(legend_data),
                            bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=14)
        for line in leg.get_lines():
            line.set_linewidth(3.0)

        plt.tight_layout(rect=[0, 0.032, 1, 1])
        dpi = 200
        plt.savefig(os.path.join(save_dir, f"{metric_type}.png"),
                    dpi=dpi, bbox_inches="tight", facecolor=bg_color)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="canonical", choices=["canonical", "classwise"])
    parser.add_argument("--ce_type", type=str, default="l2", choices=["kl", "l2"])
    parser.add_argument("--num_classes_list", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--temp1", type=float, default=1.0)
    parser.add_argument("--temp2", type=float, default=0.8)
    parser.add_argument('--skip_compute', action='store_true', help="Load existing .npy results instead of recomputing")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_nc_results = {}
    
    res_out_dir = f"./results/synthetic/t1_{args.temp1}t2_{args.temp2}"
    os.makedirs(res_out_dir, exist_ok=True)

    save_dir = f"./figs/synthetic/t1_{args.temp1}t2_{args.temp2}/{args.mode}_{args.ce_type}"
    os.makedirs(save_dir, exist_ok=True)
    methods = ["MLE-loo", "risk-loo", "equal_bin"]
    if args.mode == "classwise":
        methods += ["adapt_bin"]
    sizes = (300, 1000, 3000, 5000, 10000, 15000, 20000)
    for nc in args.num_classes_list:
        res_filename = f"nc{nc}_{args.mode}_{args.ce_type}.npy"
        res_path = os.path.join(res_out_dir, res_filename)
        if args.skip_compute and os.path.exists(res_path):
            print(f"\n[Skip] Loading existing results for NC={nc} from: {res_path}")
            all_nc_results[nc] = np.load(res_path, allow_pickle=True).item()
        else:
            print(f"\n>>> Running Experiment: Mode={args.mode} | NC={nc} | CE={args.ce_type}")
            results = run_synthetic_experiment(
                pool_size=200000,
                sizes=sizes,
                repeats=args.n_repeats,
                num_classes=nc,
                temp1=args.temp1,
                temp2=args.temp2,
                mode=args.mode,
                ce_type=args.ce_type,
                methods=methods,
                device=device
            )
            np.save(res_path, results)
            all_nc_results[nc] = results
            print(f"Results saved to: {res_path}")

        print_experiment_results(all_nc_results[nc])

    plot_synthetic_results(all_nc_results, args.ce_type, args.mode, save_dir)
