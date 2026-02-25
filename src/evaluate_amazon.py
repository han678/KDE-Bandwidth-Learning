import argparse
import json
import os
import torch
import numpy as np
import sys
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_path)

from src.evaluate_cifar import (
    get_logits_and_labels, 
    run_estimation_experiment, 
    plot_comparison
)


def load_data_amazon(path_to_root, model_name):
    """
    Load data from path_to_root/amazon_{model_name}/target_agg.json
    """
    folder_name = f'amazon_{model_name}'
    path_to_folder = os.path.join(path_to_root, folder_name)
    json_path = os.path.join(path_to_folder, 'target_agg.json')
    
    if not os.path.exists(json_path):
        print(f"[Skip] File not found: {json_path}")
        return None, None
        
    with open(json_path, 'r') as file:
        preds_dict = json.load(file)
    
    logits, labels = get_logits_and_labels(preds_dict, logits_key='y_logits', labels_key='y_true')
    return logits, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_trained_models', type=str, default='train_models/Amazon')
    parser.add_argument('--dataset', type=str, default='amazon')
    parser.add_argument('--mode', type=str, default='canonical', choices=['canonical', 'classwise'])
    parser.add_argument('--ce_type', type=str, default='l2', choices=['kl', 'l2'])
    parser.add_argument('--model_names', type=str, nargs='+', default=['bert', 'distill_bert', 'roberta', 'distill_roberta'])
    parser.add_argument('--n_repeats', type=int, default=5) 
    parser.add_argument('--output_path', type=str, default='figs/')
    parser.add_argument('--skip_compute', action='store_true', help="Load existing .npy results instead of recomputing")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    methods = ["MLE-loo", "risk-loo", "equal_bin"]
    if args.mode == "classwise":
        methods += ["adapt_bin"]
    subset_sizes = [500, 1000, 3000, 5000, 10000, 15000, 20000]
    
    result_out_dir = f'results/{args.dataset}/{args.mode}/'
    os.makedirs(result_out_dir, exist_ok=True)

    for model_name in tqdm(args.model_names, desc="Processing Amazon Models"):
        prefix = f"{model_name}_{args.ce_type}"
        results_file_path = os.path.join(result_out_dir, f"{prefix}.npy")
        
        res = None
        if args.skip_compute and os.path.exists(results_file_path):
            try:
                res = np.load(results_file_path, allow_pickle=True).item()
            except Exception as e:
                print(f"[{model_name}] Load failed: {e}")

        if res is None:
            logits, labels = load_data_amazon(args.path_to_trained_models, model_name)
            if logits is None: continue 
            
            print(f"\nEvaluating {model_name} (Mode: {args.mode})...")
            res = run_estimation_experiment(
                logits, labels, subset_sizes, args.n_repeats, methods, 
                ce_type=args.ce_type, mode=args.mode, device=device
            )
            np.save(results_file_path, res)
    
    result_out_dir = f'results/{args.dataset}/{args.mode}/'
    os.makedirs(result_out_dir, exist_ok=True)
    all_results = {}
    for model_name in ['bert', 'distill_bert', 'roberta',  'distill_roberta']:
        prefix = f"{model_name}_{args.ce_type}"
        results_file_path = os.path.join(result_out_dir, f"{prefix}.npy")
        res = None
        if os.path.exists(results_file_path):
            try:
                res = np.load(results_file_path, allow_pickle=True).item()
            except Exception as e:
                print(f"[{model_name}] Load failed: {e}")
        if res is not None:
            all_results[model_name] = res
        else:
            print(f"[Warning] No valid results found for {model_name}, skipping from plot.")
    if not all_results:
        print("\n[Error] No results available to plot. Check data paths.")
        return

    fig_dir = os.path.join(args.output_path, args.dataset, args.mode)
    os.makedirs(fig_dir, exist_ok=True)
    
    print(f"\nGenerating combined plots in {fig_dir}...")
    
    plot_comparison(all_results,  metric="ce", ce_type=args.ce_type, mode=args.mode, save_path=os.path.join(fig_dir, f"{args.ce_type}_ce.png"))

if __name__ == "__main__":
    main()