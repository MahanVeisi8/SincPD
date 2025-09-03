#!/usr/bin/env python3
import argparse, os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sincpd.datasets import load_physionet_flat

def main():
    ap = argparse.ArgumentParser(description='Build NPZ splits from a flat-folder PhysioNet-style dataset + demographics file')
    ap.add_argument('--trials_dir', required=True)
    ap.add_argument('--demographics', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--filename_col', default='filename')
    ap.add_argument('--label_col', default='label')
    ap.add_argument('--positive_values', nargs='*', default=None)
    ap.add_argument('--win_len', type=int, default=8000)
    ap.add_argument('--step', type=int, default=None)
    ap.add_argument('--merge_lr', action='store_true')
    ap.add_argument('--test_size', type=float, default=0.2)
    ap.add_argument('--val_size', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    X, y = load_physionet_flat(args.trials_dir, args.demographics, args.filename_col, args.label_col, None, args.positive_values, args.merge_lr, args.win_len, args.step)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=args.test_size+args.val_size, random_state=args.seed, stratify=y if len(np.unique(y))>1 else None)
    rel_val = args.val_size / (args.test_size + args.val_size) if (args.test_size + args.val_size) > 0 else 0.0
    if rel_val > 0:
        X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=1-rel_val, random_state=args.seed, stratify=y_tmp if len(np.unique(y_tmp))>1 else None)
    else:
        X_val, y_val, X_test, y_test = np.empty((0,)), np.empty((0,)), X_tmp, y_tmp
    out_dir = Path(args.out_dir) / 'data'
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / 'train.npz', X=X_train, y=y_train)
    if X_val.size:
        np.savez_compressed(out_dir / 'val.npz', X=X_val, y=y_val)
    np.savez_compressed(out_dir / 'test.npz', X=X_test, y=y_test)
    print('Saved:', out_dir)

if __name__ == '__main__':
    main()
