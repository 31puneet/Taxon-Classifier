import argparse
import os
import sys
import glob
import numpy as np
import joblib
import warnings
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer

warnings.filterwarnings('ignore')

N_FEATURES = 1048576


def parse_fastq(filepath, n_reads=10000):
    seqs = []
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num % 4 == 1:
                seq = line.strip()
                if len(seq) > 50:
                    seqs.append(seq)
                if len(seqs) >= n_reads:
                    break
    return seqs


def seqs_to_vector(seqs, k):
    vectorizer = HashingVectorizer(
        analyzer='char', ngram_range=(k, k),
        n_features=N_FEATURES, norm=None, alternate_sign=False
    )
    seq_text = " ".join(seqs)
    hashed = vectorizer.transform([seq_text])
    vec = np.zeros(N_FEATURES, dtype=np.float32)
    total = sum(hashed.data)
    if total > 0:
        for idx, val in zip(hashed.indices, hashed.data):
            vec[idx] = val / total
    return vec


def main():
    parser = argparse.ArgumentParser(description="Evaluate taxon classifier on test FASTQ files")
    parser.add_argument('--k', type=int, default=15, help="K-mer size (must match training)")
    parser.add_argument('files', nargs='*', help="FASTQ files to evaluate")
    args = parser.parse_args()

    k = args.k
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(script_dir, "../../models"))
    test_dir = os.path.abspath(os.path.join(script_dir, "../../data/test_samples"))

    model_path = os.path.join(models_dir, f"lightgbm_fastq_k{k}.pkl")
    encoder_path = os.path.join(models_dir, f"label_encoder_fastq_k{k}.pkl")

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print(f"Run train_models.py --k {k} first.")
        return

    model = joblib.load(model_path)
    le = joblib.load(encoder_path)

    if args.files:
        files = []
        for f in args.files:
            path = f if os.path.isabs(f) else os.path.join(test_dir, f)
            files.append(path)
    else:
        files = sorted(glob.glob(os.path.join(test_dir, "*.fastq")))

    print(f"======= Evaluating with k={k} model =======")
    print(f"Model classes: {le.classes_.tolist()}\n")

    correct, total = 0, 0
    for filepath in files:
        filename = os.path.basename(filepath)
        true_species = "_".join(filename.split("_")[:-1])

        seqs = parse_fastq(filepath)

        vec = seqs_to_vector(seqs, k)
        X_test = csr_matrix(vec.reshape(1, -1))

        pred_idx = model.predict(X_test)[0]
        pred_label = le.inverse_transform([pred_idx])[0]
        confidence = model.predict_proba(X_test)[0][pred_idx] * 100

        is_correct = pred_label == true_species
        if is_correct:
            correct += 1
        total += 1

        status = "CORRECT" if is_correct else "WRONG"
        print(f"  File: {filename}")
        print(f"    Reads:      {len(seqs)}")
        print(f"    True:       {true_species}")
        print(f"    Predicted:  {pred_label} ({confidence:.1f}%)")
        print(f"    [{status}]\n")

    if total > 0:
        print(f"  Overall: {correct}/{total} correct ({correct/total*100:.1f}%)")


if __name__ == "__main__":
    main()
