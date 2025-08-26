import torch
from sincpd.models.sinc import SincConv1d

def test_sinc_forward():
    layer = SincConv1d(out_channels=4, kernel_size=65, sample_rate=100.0)
    x = torch.randn(2, 1, 256)
    y = layer(x)
    assert y.shape[1] == 4
