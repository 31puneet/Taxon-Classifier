import os
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

def train_and_cross_validate(X, y, classes, output_dir="../models"):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
    }
    predictions = {}
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        y_pred = cross_val_predict(model, X, y, cv=cv, method='predict')
        y_proba = np.zeros((len(y), len(classes)))
        try:
             y_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')
        except Exception:
             pass    
        model.fit(X, y)
        model_filename = name.replace(" ", "_").lower() + ".pkl"
        joblib.dump(model, os.path.join(output_dir, model_filename))
        predictions[name] = {'y_pred': y_pred, 'y_proba': y_proba, 'model_obj': model}
        
    return predictions

if __name__ == "__main__":
    X_path = '../data/X_features.npy'
    y_path = '../data/y_labels.npy'

    if os.path.exists(X_path) and os.path.exists(y_path):
        X = np.load(X_path)
        y = np.load(y_path, allow_pickle=True)
        classes = np.unique(y)
        train_and_cross_validate(X, y, classes)
