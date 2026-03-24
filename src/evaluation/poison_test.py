import argparse
import os
import random
import numpy as np
import joblib
import glob
import warnings
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer

warnings.filterwarnings('ignore')

N_FEATURES = 1048576
BASES = ['A', 'T', 'C', 'G']


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


def poison_sequence(seq, error_rate):
    poisoned = list(seq)
    for i in range(len(poisoned)):
        if random.random() < error_rate:
            original = poisoned[i]
            poisoned[i] = random.choice([b for b in BASES if b != original])
    return "".join(poisoned)


def poison_reads(seqs, error_rate):
    return [poison_sequence(s, error_rate) for s in seqs]


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
    parser = argparse.ArgumentParser(description="Noise injection stress test")
    parser.add_argument('--k', type=int, default=15, help="K-mer size (must match training)")
    args = parser.parse_args()

    k = args.k
    print(f"=== POISON TEST: Noise Injection Stress Test (k={k}) ===\n")

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

    fastq_files = sorted(glob.glob(os.path.join(test_dir, "*.fastq")))
    if not fastq_files:
        print("No FASTQ files found. Download test samples first.")
        return

    error_rates = [0.0, 0.01, 0.05, 0.10, 0.20, 0.30]
    random.seed(42)

    print(f"{'Error Rate':<12} ", end="")
    for fp in fastq_files:
        name = "_".join(os.path.basename(fp).split("_")[:-1])
        print(f"{'[' + name + ']':<22} ", end="")
    print("Accuracy")
    print("-" * 100)

    all_seqs = {}
    for fp in fastq_files:
        all_seqs[fp] = parse_fastq(fp)

    for rate in error_rates:
        correct = 0
        total = 0
        row = f"{rate*100:>5.0f}%        "

        for fp in fastq_files:
            true_name = "_".join(os.path.basename(fp).split("_")[:-1])
            seqs = all_seqs[fp]

            if rate > 0:
                seqs = poison_reads(seqs, rate)

            vec = seqs_to_vector(seqs, k)
            X_test = csr_matrix(vec.reshape(1, -1))

            pred_idx = model.predict(X_test)[0]
            pred_label = le.inverse_transform([pred_idx])[0]
            confidence = model.predict_proba(X_test)[0][pred_idx] * 100

            is_correct = pred_label == true_name
            if is_correct:
                correct += 1
            total += 1

            status = "PASS" if is_correct else "FAIL"
            row += f"{pred_label:<15} {confidence:>5.1f}% {status}  "

        acc = correct / total * 100
        row += f"  {acc:.0f}%"
        print(row)

    print(f"\n=== Poison Test Complete ===")


if __name__ == "__main__":
    main()
