import sys
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
from Bio import SeqIO
from sklearn.feature_extraction.text import HashingVectorizer

K = 6
N_FEATURES = 4096
N_READS = 1000

def predict_species(fastq_path, model_dir="../models"):
    vectorizer = HashingVectorizer(
        analyzer='char', ngram_range=(K, K),
        n_features=N_FEATURES, norm=None, alternate_sign=False
    )
    seqs = []
    for i, record in enumerate(SeqIO.parse(fastq_path, "fastq")):
        if i >= N_READS:
            break
        seqs.append(str(record.seq))

    if not seqs:
        print("No reads found.")
        return

    print(f"Read {len(seqs)} sequences from {fastq_path}")

    seq_text = " ".join(seqs)
    hashed = vectorizer.transform([seq_text])
    vec = np.zeros(N_FEATURES)
    total = sum(hashed.data)
    if total > 0:
        for idx, val in zip(hashed.indices, hashed.data):
            vec[idx] = val / total

    model = joblib.load(f"{model_dir}/lightgbm_fastq.pkl")
    le = joblib.load(f"{model_dir}/label_encoder_fastq.pkl")

    pred = model.predict(vec.reshape(1, -1))
    proba = model.predict_proba(vec.reshape(1, -1))[0]
    species = le.inverse_transform(pred)[0]

    print(f"\nPredicted species: {species}")
    print("\nConfidence:")
    for name, p in zip(le.classes_, proba):
        print(f"  {name}: {p:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m evaluation.test <path_to_fastq>")
        sys.exit(1)
    predict_species(sys.argv[1])
