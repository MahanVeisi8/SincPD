from pathlib import Path
import torch

def save_ckpt(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

def load_ckpt(model, path: Path, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    return model
