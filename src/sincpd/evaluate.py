import argparse
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from .layers import SincConv1D
from .data import load_npz

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True)
    args=ap.parse_args()
    X,y=load_npz(args.data)
    model=load_model(args.model, custom_objects={'SincConv1D': SincConv1D})
    yhat=(model.predict(X)>0.5).astype(int).ravel()
    print(confusion_matrix(y,yhat))
    print(classification_report(y,yhat,digits=3))

if __name__=='__main__': main()