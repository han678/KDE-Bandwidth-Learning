import sys
import numpy as np
import torch
import torch.nn.functional as F
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_path)

from bin_method.bin_estimators import get_ece_bin, get_risk_bin
from kde.bandwidth import select_bandwidth
from kde.kde_estimators import get_risk_kde

def get_calibration_estimators(method_specs, ce_type="l2", mode="classwise"):
    estimators = {}
    for key, base_method, nb in method_specs:
        if "bin" in base_method:
            is_adapt = (base_method == "adapt_bin")
            def estimator_fn(f, y, _nb=nb, _ia=is_adapt):
                ce_val = get_ece_bin(f, y, n_bins=_nb, mode=mode, ce_type=ce_type, adaptive=_ia)
                risk = get_risk_bin(f, y, n_bins=_nb, mode=mode, ce_type=ce_type, adaptive=_ia)
                return {"ce": float(ce_val.item()), "risk": float(risk.item()), "bw": 0.0}
        else:
            def estimator_fn(f, y, _bm=base_method):
                bw = select_bandwidth(f, y, mode, ce_type=ce_type, method=_bm)
                cal, ref = get_risk_kde(f, y, bw, mode, ce_type, return_components=True)
                if isinstance(bw, torch.Tensor):
                    bw_val = bw.cpu().tolist() if bw.numel() > 1 else bw.item()
                else:
                    bw_val = bw
                ce_val = float(cal.item())
                return {
                    "ce": max(0, ce_val) if ce_type == "l2" else ce_val,
                    "risk": float((cal + ref).item()),
                    "bw": bw_val
                }
        
        estimators[key] = estimator_fn
    return estimators

def print_experiment_results(res):
    true_ce = res.get("true_ce", 0.0)
    true_risk = res.get("true_risk", 0.0)
    method_order = res.get("_method_order", [])
    if not method_order: return

    print(f"\n{'#'*82}")
    print(f"Ground Truth -> CE: {true_ce:.4e} | Risk: {true_risk:.4f}")
    print(f"{'#'*82}")

    sample_sizes = res[method_order[0]]["N"]
    for i, N_val in enumerate(sample_sizes):
        print(f"\n[ N = {N_val} ]")
        print(f"{'Method':<22} | {'CE (MAE)':<22} | {'Risk (MAE)':<22} | {'BW':<8}")
        print("-" * 82)
        for m in method_order:
            r = res[m]
            ce_str = f"{r['ce'][i]:.2e} ({r['ce_mae'][i]:.2e})"
            risk_str = f"{r['risk'][i]:.4f} ({r['risk_mae'][i]:.4f})"
            bw = r['bw'][i]
            bw_str = f"{bw[0]:.4f}*" if isinstance(bw, list) else f"{bw:.4f}"
            print(f"{m:<22} | {ce_str:<22} | {risk_str:<22} | {bw_str:<8}")