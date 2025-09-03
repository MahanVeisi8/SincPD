import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable
from ..preprocess import merge_lr, window_standardize

"""
Flat-folder PhysioNet-style loader for Parkinson's gait:
- All trial files are in ONE folder (no per-subject subdirs).
- A demographics file (Excel/CSV) maps each trial (by filename or subject) to labels.
- We support flexible column names via arguments.
"""

def _read_tabular(path: Path, expected_cols: int = 16) -> np.ndarray:
    try:
        df = pd.read_csv(path, sep=',')
        if df.shape[1] < expected_cols:
            df = pd.read_csv(path, sep='\t')
    except Exception:
        df = pd.read_csv(path, sep='\t')
    if all(str(c).startswith('Unnamed') for c in df.columns):
        df = pd.read_csv(path, sep=',', header=None)
        if df.shape[1] < expected_cols:
            df = pd.read_csv(path, sep='\t', header=None)
    arr = df.iloc[:, :expected_cols].to_numpy(dtype=np.float32, copy=True)
    return arr

def _read_demographics(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in ['.xls', '.xlsx']:
        return pd.read_excel(path)
    elif suf in ['.csv', '.tsv']:
        sep = ',' if suf == '.csv' else '\t'
        return pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f'Unsupported demographics file type: {suf}')

def load_physionet_flat(
    trials_dir: str,
    demographics_path: str,
    filename_col: str = 'filename',
    label_col: str = 'label',
    subject_col: Optional[str] = None,
    positive_values: Optional[Iterable] = None,
    merge_lr_channels: bool = True,
    win_len: int = 8000,
    step: Optional[int] = None,
    min_len: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load gait trials from a flat folder using a demographics file.
    Args:
        trials_dir: directory containing all trial files (.csv/.tsv/.txt)
        demographics_path: path to demographics .xls/.xlsx/.csv containing at least filename_col and label_col
        filename_col: column name in demographics that matches the trial filename (with or without extension)
        label_col: column with target label (binary or multi-class integer; if not integer, map via positive_values)
        subject_col: optional column with subject ID (not used in this function yet)
        positive_values: optional iterable mapping to label=1 for binary classification
        merge_lr_channels: if True, reduce 16->8 channels via L-R
        win_len/step: windowing parameters
        min_len: skip trials shorter than this
    Returns:
        X: (N, T, C), y: (N,)
    """
    trials_dir = Path(trials_dir)
    demo = _read_demographics(Path(demographics_path)).copy()
    def norm_name(s: str) -> str:
        s = str(s)
        base = Path(s).name
        return base.rsplit('.', 1)[0].lower()
    demo['_fname_key'] = demo[filename_col].astype(str).map(norm_name)
    labels_map: Dict[str, int] = {}
    if positive_values is not None:
        pset = set([str(v).lower() for v in positive_values])
        for _, r in demo.iterrows():
            key = r['_fname_key']
            val = r[label_col]
            lab = 1 if str(val).lower() in pset else 0
            labels_map[key] = lab
    else:
        for _, r in demo.iterrows():
            key = r['_fname_key']
            val = r[label_col]
            try:
                labels_map[key] = int(val)
            except Exception:
                labels_map[key] = 1 if str(val).strip().lower() not in {'0','false','no',''} else 0
    trial_files = sorted([p for p in trials_dir.iterdir() if p.suffix.lower() in ['.csv','.tsv','.txt']])
    X_chunks: List[np.ndarray] = []
    y_chunks: List[int] = []
    for tf in trial_files:
        key = norm_name(tf.name)
        if key not in labels_map:
            continue
        arr = _read_tabular(tf, expected_cols=16)
        if arr.shape[0] < min_len:
            continue
        if merge_lr_channels:
            arr = merge_lr(arr)
        Xw = window_standardize(arr, y=None, win_len=win_len, step=step)
        if Xw.shape[0] == 0:
            continue
        lab = int(labels_map[key])
        yw = np.full((Xw.shape[0],), lab, dtype=np.int64)
        X_chunks.append(Xw)
        y_chunks.append(yw)
    if not X_chunks:
        C = 8 if merge_lr_channels else 16
        return np.empty((0, win_len, C), dtype=np.float32), np.empty((0,), dtype=np.int64)
    X = np.concatenate(X_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)
    return X, y
