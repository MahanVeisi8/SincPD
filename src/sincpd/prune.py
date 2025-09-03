import argparse, os
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
from .layers import SincConv1D
from .models import build_binary_model
from .data import load_npz

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=30)
    ap.add_argument('--train', default='')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch', type=int, default=32)
    args=ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model=load_model(args.model, custom_objects={'SincConv1D': SincConv1D})
    sinc=None
    for lyr in model.layers:
        if isinstance(lyr, SincConv1D) or 'sinc' in lyr.name.lower():
            sinc=lyr; break
    if sinc is None: raise RuntimeError('No SincConv1D layer found.')
    low=np.abs(sinc.low_hz.numpy()); band=np.abs(sinc.band_hz.numpy())
    P=np.stack([low,band],axis=1)
    k=min(args.k, P.shape[0])
    km=KMeans(n_clusters=k, n_init=10, random_state=0).fit(P)
    centroids=km.cluster_centers_
    T,C=model.input_shape[1:]
    pruned=build_binary_model(input_shape=(T,C), fs=sinc.sample_rate, sinc_filters=k, sinc_kernel=sinc.kernel_size)
    _=pruned.layers[1].low_hz
    pruned.layers[1].low_hz.assign(centroids[:,0])
    pruned.layers[1].band_hz.assign(centroids[:,1])
    if args.train:
        Xtr,ytr=load_npz(args.train)
        pruned.fit(Xtr,ytr,epochs=args.epochs,batch_size=args.batch,verbose=1)
    pruned.save(os.path.join(args.out,'model_pruned.keras'))
    print('Saved pruned model to', args.out)

if __name__=='__main__': main()