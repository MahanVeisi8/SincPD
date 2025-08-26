import argparse
from pathlib import Path
from sincpd.training.trainer import run_training, run_eval
from sincpd.pruning.kmeans_prune import run_pruning

def main():
    p = argparse.ArgumentParser("SincPD CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train a model using a YAML config")
    p_train.add_argument("--config", type=Path, required=True)

    p_eval = sub.add_parser("eval", help="Evaluate a checkpoint on split")
    p_eval.add_argument("--ckpt", type=Path, required=True)
    p_eval.add_argument("--split", type=str, default="val")

    p_prune = sub.add_parser("prune", help="Prune a trained model with K-Means")
    p_prune.add_argument("--ckpt", type=Path, required=True)

    args = p.parse_args()
    if args.cmd == "train":
        run_training(args.config)
    elif args.cmd == "eval":
        run_eval(args.ckpt, split=args.split)
    elif args.cmd == "prune":
        run_pruning(args.ckpt)
    else:
        p.print_help()
