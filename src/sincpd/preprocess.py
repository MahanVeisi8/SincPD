import numpy as np

def merge_lr(x16):
    x = np.asarray(x16)
    if x.ndim==2:
        L,R = x[:,:8], x[:,8:]
        return L-R
    elif x.ndim==3:
        L,R = x[:,:,:8], x[:,:,8:]
        return L-R
    else:
        raise ValueError('Expected (T,16) or (N,T,16)')

def window_standardize(x, y=None, win_len=8000, step=None, eps=1e-6):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim==2: x=x[None,...]
    N,T,C = x.shape
    if step is None: step = win_len
    chunks, ys = [], []
    for n in range(N):
        t0=0
        while t0+win_len<=T:
            seg = x[n,t0:t0+win_len,:]
            mu = seg.mean(axis=0, keepdims=True)
            std = seg.std(axis=0, keepdims=True)+eps
            chunks.append((seg-mu)/std)
            if y is not None: ys.append(int(y[n]) if np.ndim(y)>0 else int(y))
            t0+=step
    Xw = np.stack(chunks,axis=0) if chunks else np.empty((0,win_len,C),dtype=np.float32)
    if y is None: return Xw
    return Xw, np.array(ys,dtype=np.int64)