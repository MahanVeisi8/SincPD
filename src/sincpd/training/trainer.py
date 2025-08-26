from pathlib import Path
import yaml, torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import trange

from sincpd.utils.seed import set_global_seed
from sincpd.utils.io import save_ckpt, load_ckpt
from sincpd.utils.plotting import save_curve
from sincpd.data.datasets import make_dummy_dataset, make_loaders
from sincpd.models.sinc import SincPDNet
from sincpd.training.optim import make_optimizer
from sincpd.eval.metrics import binary_classification_report

def _device(tag: str):
    if tag == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return tag

def run_training(cfg_path: Path):
    cfg = yaml.safe_load(open(cfg_path))
    set_global_seed(cfg.get("seed", 1337))

    # Data (replace with real loader as needed)
    train_set, val_set = make_dummy_dataset(n=512, T=cfg["input_len"], C=cfg["n_channels"])
    train_loader, val_loader = make_loaders(train_set, val_set, batch_size=cfg["batch_size"])

    device = _device(cfg.get("device", "auto"))
    model = SincPDNet(cfg["n_channels"], cfg["input_len"], cfg["model"]["n_filters"], cfg["model"]["kernel_size"], cfg["model"]["sample_rate"], task=cfg.get("task", "classification")).to(device)
    opt = make_optimizer(model, lr=cfg["lr"])

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_losses = []
    for epoch in trange(cfg["epochs"], desc="epochs"):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            if cfg.get("task", "classification") == "regression":
                loss = F.l1_loss(p, y)
            else:
                loss = F.binary_cross_entropy(p, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        save_curve(train_losses, out_dir / "train_loss.png", title="Train loss", ylabel="BCE")

    # Save model
    save_ckpt(model, out_dir / "model.pt")

    # Eval
    model.eval()
    probs, gts = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            p = model(x).detach().cpu()
            probs.append(p.numpy())
            gts.append(y.numpy())
    import numpy as np
    probs = np.concatenate(probs)
    gts = np.concatenate(gts)
    rep = binary_classification_report(gts, probs)
    with open(out_dir / "metrics.json", "w") as f:
        import json; json.dump(rep, f, indent=2)
    print("Validation metrics:", rep)

def run_eval(ckpt: Path, split: str = "val"):
    # For now, load dummy val set and evaluate
    _, val_set = make_dummy_dataset()
    from sincpd.models.sinc import SincPDNet
    from sincpd.data.datasets import make_loaders
    loader = make_loaders(_, val_set, batch_size=64)[1]
    model = SincPDNet(n_channels=8, input_len=2048)
    load_ckpt(model, ckpt)
    model.eval()
    import numpy as np, torch
    probs, gts = [], []
    with torch.no_grad():
        for x, y in loader:
            p = model(x).detach().cpu()
            probs.append(p.numpy()); gts.append(y.numpy())
    probs = np.concatenate(probs); gts = np.concatenate(gts)
    from sincpd.eval.metrics import binary_classification_report
    rep = binary_classification_report(gts, probs)
    print(rep)

def demo_training():
    # A self-contained minimal run without YAML
    import tempfile, yaml
    cfg = {
        "seed": 1337, "device": "auto", "input_len": 1024, "n_channels": 8, "batch_size": 32,
        "epochs": 2, "lr": 1e-3, "out_dir": "runs/demo",
        "model": {"n_filters": 8, "kernel_size": 65, "sample_rate": 100.0},
    }
    p = Path("runs/demo"); p.mkdir(parents=True, exist_ok=True)
    with open(p / "config_used.yaml", "w") as f: yaml.safe_dump(cfg, f)
    import json; print("Config:", json.dumps(cfg, indent=2))
    # inline train
    set_global_seed(cfg["seed"])
    train_set, val_set = make_dummy_dataset(n=128, T=cfg["input_len"], C=cfg["n_channels"])
    train_loader, val_loader = make_loaders(train_set, val_set, batch_size=cfg["batch_size"])
    device = _device(cfg["device"])
    model = SincPDNet(cfg["n_channels"], cfg["input_len"], cfg["model"]["n_filters"], cfg["model"]["kernel_size"], cfg["model"]["sample_rate"]).to(device)
    opt = make_optimizer(model, lr=cfg["lr"])
    for epoch in range(cfg["epochs"]):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            loss = torch.nn.functional.binary_cross_entropy(p, y)
            opt.zero_grad(); loss.backward(); opt.step()
    from sincpd.utils.io import save_ckpt
    save_ckpt(model, Path(cfg["out_dir"]) / "model.pt")
    print("Demo training finished.")
