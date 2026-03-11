import os
import urllib.request
import zlib
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

K = 6
N_FEATURES = 4096
N_READS = 1000
SAMPLES_PER_SPECIES = 25

SPECIES = {
    'gambiae_complex': 7165,
    'arabiensis': 7179,
    'funestus': 62324,
    'stephensi': 30069,
}

def get_fastq_urls(tax_id, limit=50):
    url = (
        f"https://www.ebi.ac.uk/ena/portal/api/search?"
        f"result=read_run&"
        f"query=tax_eq({tax_id})%20AND%20library_strategy=%22WGS%22&"
        f"fields=run_accession,fastq_ftp,read_count&"
        f"limit={limit}&"
        f"format=tsv"
    )
    resp = urllib.request.urlopen(url)
    lines = resp.read().decode().strip().split('\n')
    header = lines[0].split('\t')
    urls = []
    for line in lines[1:]:
        fields = line.split('\t')
        row = dict(zip(header, fields))
        if int(row.get('read_count', 0)) > 100000:
            ftp_list = row['fastq_ftp'].split(';')
            pick = [u for u in ftp_list if u.endswith('_1.fastq.gz')]
            if not pick:
                pick = ftp_list
            urls.append('https://' + pick[0])
    return urls

def stream_reads(url, n_reads=1000):
    try:
        resp = urllib.request.urlopen(url, timeout=60)
        raw = resp.read(3 * 1024 * 1024)
        resp.close()
        dec = zlib.decompressobj(zlib.MAX_WBITS | 32)
        text = dec.decompress(raw).decode('utf-8', errors='ignore')
        lines = text.split('\n')
        seqs = []
        i = 0
        while i + 3 < len(lines) and len(seqs) < n_reads:
            if lines[i].startswith('@'):
                seq = lines[i + 1].strip()
                if len(seq) > 50:
                    seqs.append(seq)
                i += 4
            else:
                i += 1
        return seqs
    except:
        return []

def seqs_to_vector(seqs, vectorizer):
    seq_text = " ".join(seqs)
    hashed = vectorizer.transform([seq_text])
    vec = np.zeros(N_FEATURES)
    total = sum(hashed.data)
    if total > 0:
        for idx, val in zip(hashed.indices, hashed.data):
            vec[idx] = val / total
    return vec

def main():
    print("=== FASTQ-based Model Training ===\n")
    vectorizer = HashingVectorizer(
        analyzer='char', ngram_range=(K, K),
        n_features=N_FEATURES, norm=None, alternate_sign=False
    )
    X_all, y_all = [], []

    for species, tax_id in SPECIES.items():
        print(f"--- {species} (tax_id={tax_id}) ---")
        urls = get_fastq_urls(tax_id, limit=SAMPLES_PER_SPECIES * 3)
        print(f"  Found {len(urls)} URLs")
        done = 0
        for url in urls:
            if done >= SAMPLES_PER_SPECIES:
                break
            fname = url.split('/')[-1]
            print(f"  [{done+1}/{SAMPLES_PER_SPECIES}] {fname}...", end=" ", flush=True)
            seqs = stream_reads(url, N_READS)
            if len(seqs) >= 100:
                vec = seqs_to_vector(seqs, vectorizer)
                X_all.append(vec)
                y_all.append(species)
                done += 1
                print(f"OK ({len(seqs)} reads)")
            else:
                print(f"skipped ({len(seqs)} reads)")
        print(f"  Got {done} samples\n")

    X = np.array(X_all)
    y = np.array(y_all)
    print(f"Feature matrix: {X.shape}")

    os.makedirs("../data", exist_ok=True)
    np.save("../data/X_fastq.npy", X)
    np.save("../data/y_fastq.npy", y)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print("\nTraining LightGBM...")
    model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y_enc, cv=cv)

    print(f"\nAccuracy: {accuracy_score(y_enc, y_pred):.4f}")
    print(f"Macro-F1: {f1_score(y_enc, y_pred, average='macro'):.4f}")
    print(classification_report(y_enc, y_pred, target_names=le.classes_))

    model.fit(X, y_enc)
    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/lightgbm_fastq.pkl")
    joblib.dump(le, "../models/label_encoder_fastq.pkl")
    print("Model saved! Now test with: python -m evaluation.test <fastq_file>")

if __name__ == "__main__":
    main()
