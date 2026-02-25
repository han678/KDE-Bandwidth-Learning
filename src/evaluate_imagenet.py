import argparse
import os
import random
import numpy as np
import torch
import sys
cache_path = "/leonardo_work/EUHPC_D25_055/hz/refine/hf_cache"
os.environ['HF_HOME'] = cache_path
os.environ['HF_HUB_OFFLINE'] = '1'
import timm

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_path)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x
from src.evaluate_cifar import (
    run_estimation_experiment, 
    plot_comparison
)
from utils.loaders import prepare_dataset

def get_logits_from_model(model, loader, device):
    all_logits = []
    all_targets = []
    
    print("Running inference to collect ImageNet logits...")
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Inference"):
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu()) 
            all_targets.append(targets.cpu())
    full_logits = torch.cat(all_logits, dim=0)
    full_targets = torch.cat(all_targets, dim=0)
    
    return full_logits, full_targets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_names', type=str, nargs='+', 
                        default=['vit_small_patch16_224', 'vit_base_patch16_224', 
                                 "swin_small_patch4_window7_224", "deit_base_patch16_224"])
    parser.add_argument('--data_dir', type=str, default='/leonardo_work/EUHPC_D25_055/hz/refine/data/ILSVRC2012')
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--mode', type=str, default='classwise', choices=['canonical', 'classwise'])
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--output_path', type=str, default='figs/')
    parser.add_argument('--cache_dir', type=str, default='train_models/ImageNet')
    parser.add_argument('--ce_type', type=str, default='l2', choices=['kl', 'l2'])
    parser.add_argument('--n_repeats', type=int, default=5) 
    parser.add_argument('--skip_compute', action='store_true', default=False)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    methods = ["MLE-loo", "risk-loo", "equal_bin"]
    if args.mode == "classwise":
        methods += ["adapt_bin"]
    subset_sizes = [5000, 10000, 15000, 20000]  
    
    results_out_path = f"results/{args.dataset}/{args.mode}"
    os.makedirs(results_out_path, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True) 
    
    loader = None 
    for model_name in args.model_names:
        print(f"\n{'='*60}\nProcessing ImageNet Model: {model_name}\n{'='*60}")
        res_file_path = os.path.join(results_out_path, f"{model_name}_{args.ce_type}.npy") 
        
        res = None
        if args.skip_compute and os.path.exists(res_file_path):
            print(f"[Skip] Loading results from {res_file_path}")
            try:
                res = np.load(res_file_path, allow_pickle=True).item()
            except Exception as e:
                print(f"Load failed: {e}")

        if res is None:
            logits_cache_file = os.path.join(args.cache_dir, f"{model_name}.pth")
            logits, labels = None, None
            
            if os.path.exists(logits_cache_file):
                print(f"Loading logits from cache: {logits_cache_file}")
                data = torch.load(logits_cache_file, map_location="cpu")
                logits, labels = data['logits'], data['labels']
            
            if logits is None:
                if loader is None:
                    loader = prepare_dataset(args.dataset, batch_size=128, load_train=False, 
                                             shuffle=False, num_workers=args.num_workers, data_dir=args.data_dir)
                
                print(f"Creating model: {model_name}")
                model = timm.create_model(model_name, pretrained=True).to(device).eval()
                logits, labels = get_logits_from_model(model, loader, device)
                torch.save({'logits': logits, 'labels': labels}, logits_cache_file)
                del model
                torch.cuda.empty_cache()

            print(f"Starting estimation (N_total={logits.shape[0]}, K=1000)...")
            res = run_estimation_experiment(
                logits, labels, subset_sizes, args.n_repeats, methods, 
                ce_type=args.ce_type, mode=args.mode, device=device
            )
            np.save(res_file_path, res)

    all_models_results = {}
    target_models = ['vit_small_patch16_224', 'vit_base_patch16_224', 
                     "swin_small_patch4_window7_224", "deit_base_patch16_224"]
    
    print("\nAggregation: Collecting all available ImageNet results for combined plotting...")
    for model_name in target_models:
        res_file_path = os.path.join(results_out_path, f"{model_name}_{args.ce_type}.npy")
        if os.path.exists(res_file_path):
            try:
                res = np.load(res_file_path, allow_pickle=True).item()
                all_models_results[model_name] = res
            except Exception as e:
                print(f"[Plotting] Failed to load {model_name}: {e}")

    if not all_models_results:
        print("No results to plot. Exiting.")
        return

    base_fig_path = os.path.join(args.output_path, args.dataset, args.mode)
    os.makedirs(base_fig_path, exist_ok=True)
    
    print("\nGenerating final ImageNet comparison plots...")
    for m in ["ce"]:
        save_name = f"{args.mode}_{args.ce_type}_{m}.png"
        plot_comparison(all_models_results, m, args.ce_type, args.mode, os.path.join(base_fig_path, save_name))
    print(f"Done. Plots saved in: {base_fig_path}")

if __name__ == "__main__":
    main()