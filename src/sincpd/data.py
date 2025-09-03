import numpy as np

def load_npz(path):
    d=np.load(path)
    X,y=d['X'],d['y']
    if y.dtype.kind not in 'iu':
        y=(y>0.5).astype(np.int64)
    return X.astype(np.float32), y.astype(np.int64)
