import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

def evaluate_predictions(y_true, preds, classes):
    for name, res in preds.items():
        y_pred = res['y_pred']
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")
        print(classification_report(y_true, y_pred, target_names=classes))
