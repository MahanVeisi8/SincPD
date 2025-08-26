import torch

def make_optimizer(model, lr: float = 1e-3):
    return torch.optim.Adam(model.parameters(), lr=lr)
