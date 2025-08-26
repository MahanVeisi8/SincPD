import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def binary_classification_report(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"acc": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}
