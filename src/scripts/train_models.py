import argparse
import os
import time
import json
import numpy as np
import joblib
import warnings
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from lightgbm import LGBMClassifier
from tqdm import tqdm

warnings.filterwarnings('ignore')

try:
    from lightgbm import early_stopping as lgb_early_stopping
    from lightgbm import log_evaluation as lgb_log_evaluation
except ImportError:
    from lightgbm.callback import early_stopping as lgb_early_stopping
    from lightgbm.callback import log_evaluation as lgb_log_evaluation


def load_data(data_dir, k):
    x_path = os.path.join(data_dir, f"X_fastq_k{k}.npy")
    y_path = os.path.join(data_dir, f"y_fastq_k{k}.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"Feature files not found for k={k}. "
            f"Expected:\n  {x_path}\n  {y_path}\n"
            f"Run extract_kmers.py --k {k} first."
        )

    X_dense = np.load(x_path)
    y = np.load(y_path)

    X = csr_matrix(X_dense)
    del X_dense

    return X, y


def get_model():
    return LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        colsample_bytree=0.1,
        subsample=0.8,
        class_weight='balanced',
        feature_pre_filter=True,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )


def cross_validate(X, y_enc, le):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    y_pred = np.zeros_like(y_enc)

    for fold, (train_idx, test_idx) in enumerate(tqdm(cv.split(X, y_enc), total=3, desc="CV Folds")):
        model = get_model()
        model.fit(
            X[train_idx], y_enc[train_idx],
            eval_set=[(X[test_idx], y_enc[test_idx])],
            callbacks=[
                lgb_early_stopping(20),
                lgb_log_evaluation(-1),
            ],
        )
        y_pred[test_idx] = model.predict(X[test_idx])

    acc = accuracy_score(y_enc, y_pred)
    macro_f1 = f1_score(y_enc, y_pred, average='macro')

    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  Macro-F1:  {macro_f1:.4f}")
    print(f"\n{classification_report(y_enc, y_pred, target_names=le.classes_)}")

    return y_pred, acc, macro_f1


def train_final_model(X, y_enc):
    model = get_model()
    model.fit(X, y_enc)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM taxon classifier")
    parser.add_argument('--k', type=int, default=15, help="K-mer size used during extraction")
    args = parser.parse_args()

    k = args.k
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "../../data"))
    models_dir = os.path.abspath(os.path.join(script_dir, "../../models"))
    os.makedirs(models_dir, exist_ok=True)

    print(f"======= Training FASTQ Classifier (k={k}) =======\n")

    t_start = time.time()
    X, y = load_data(data_dir, k)
    t_load = time.time() - t_start

    print(f"  Samples:   {X.shape[0]}")
    print(f"  Features:  {X.shape[1]:,}")
    print(f"  Sparsity:  {1 - X.nnz / (X.shape[0] * X.shape[1]):.6f}")
    print(f"  Classes:   {np.unique(y).tolist()}")
    print(f"  Load time: {t_load:.2f}s\n")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print("------- 3-Fold Stratified Cross-Validation -------")
    t_cv_start = time.time()
    _, acc, macro_f1 = cross_validate(X, y_enc, le)
    t_cv = time.time() - t_cv_start

    print("------- Training Final Model (all data) -------")
    t_train_start = time.time()
    final_model = train_final_model(X, y_enc)
    t_train = time.time() - t_train_start

    model_path = os.path.join(models_dir, f"lightgbm_fastq_k{k}.pkl")
    encoder_path = os.path.join(models_dir, f"label_encoder_fastq_k{k}.pkl")
    joblib.dump(final_model, model_path)
    joblib.dump(le, encoder_path)

    t_total = time.time() - t_start

    stats = {
        "k": k,
        "samples": X.shape[0],
        "features": X.shape[1],
        "sparsity": round(1 - X.nnz / (X.shape[0] * X.shape[1]), 6),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "time_load_s": round(t_load, 2),
        "time_cv_s": round(t_cv, 2),
        "time_train_s": round(t_train, 2),
        "time_total_s": round(t_total, 2),
    }
    stats_path = os.path.join(models_dir, f"train_stats_k{k}.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Model saved:   {model_path}")
    print(f"  Encoder saved: {encoder_path}")
    print(f"  Stats saved:   {stats_path}")
    print(f"\n  CV time:       {t_cv:.2f}s")
    print(f"  Train time:    {t_train:.2f}s")
    print(f"  Total time:    {t_total:.2f}s")


if __name__ == "__main__":
    main()
