import argparse, os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from .data import load_npz
from .models import build_binary_model

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--val', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--fs', type=float, default=100.0)
    ap.add_argument('--sinc_filters', type=int, default=100)
    ap.add_argument('--sinc_kernel', type=int, default=101)
    args=ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    Xtr,ytr=load_npz(args.train)
    Xva,yva=(None,None)
    if args.val: Xva,yva=load_npz(args.val)
    T,C=Xtr.shape[1:]
    model=build_binary_model(input_shape=(T,C), fs=args.fs, sinc_filters=args.sinc_filters, sinc_kernel=args.sinc_kernel)
    ckpt=ModelCheckpoint(os.path.join(args.out,'model.keras'), monitor='val_accuracy', save_best_only=True, mode='max')
    early=EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    log=CSVLogger(os.path.join(args.out,'train.csv'))
    model.fit(Xtr,ytr, validation_data=(Xva,yva) if Xva is not None else None, epochs=args.epochs, batch_size=args.batch, callbacks=[ckpt,early,log])
    model.save(os.path.join(args.out,'final_model.keras'))
    print('Saved to', args.out)

if __name__=='__main__': main()