from pathlib import Path
import torch, numpy as np
from sklearn.cluster import KMeans
from sincpd.utils.io import load_ckpt, save_ckpt
from sincpd.models.sinc import SincPDNet

def run_pruning(ckpt_path: Path):
    """Toy pruning: cluster filters by (f2 - f1) bandwidth and keep cluster centers.
    This is a placeholder for your full pipeline.
    """
    model = SincPDNet(n_channels=8, input_len=2048)
    load_ckpt(model, ckpt_path, map_location="cpu")
    sinc = model.front[0]  # SincConv1d
    with torch.no_grad():
        bw = torch.abs(sinc.f2 - sinc.f1).cpu().numpy().reshape(-1, 1)
        k = min(8, len(bw))
        kmeans = KMeans(n_clusters=k, n_init="auto").fit(bw)
        # Keep first element of each cluster as a naive selection (demo only)
        keep_idx = []
        for c in range(k):
            idx = np.where(kmeans.labels_ == c)[0][0]
            keep_idx.append(int(idx))
        keep_idx = sorted(set(keep_idx))
        # Reduce filters
        sinc.f1 = torch.nn.Parameter(sinc.f1[keep_idx])
        sinc.f2 = torch.nn.Parameter(sinc.f2[keep_idx])
        model.head = torch.nn.Linear(model.head.in_features // (model.front[0].out_channels // len(keep_idx)), 1)
    out = Path(str(ckpt_path).replace(".pt", "_pruned.pt"))
    save_ckpt(model, out)
    print(f"Pruned checkpoint saved to: {out}")
