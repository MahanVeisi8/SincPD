from sincpd.data.datasets import make_dummy_dataset, make_loaders

def test_make_loaders():
    tr, va = make_dummy_dataset(n=64, T=256, C=4)
    tl, vl = make_loaders(tr, va, batch_size=8)
    xb, yb = next(iter(tl))
    assert xb.shape[1:] == (256, 4)
