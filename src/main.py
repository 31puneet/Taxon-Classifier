import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

import malariagen_data
from scripts.extract_kmers import run_extraction
from scripts.train_models import train_and_cross_validate
from evaluation.evaluate_models import evaluate_predictions

SAMPLES_PER_SPECIES = 125
RANDOM_SEED = 42

def generate_sample_list():
    print("Fetching metadata for sampling...")
    ag3 = malariagen_data.Ag3(pre=True)
    af1 = malariagen_data.Af1(pre=True)
    ag_meta = ag3.sample_metadata(sample_sets="3.0")
    af_meta = af1.sample_metadata()

    np.random.seed(RANDOM_SEED)
    selected = []
    for sp in ['gambiae', 'coluzzii', 'arabiensis']:
        sp_df = ag_meta[ag_meta['taxon'] == sp]
        sampled = sp_df.sample(n=SAMPLES_PER_SPECIES, random_state=RANDOM_SEED)
        sampled = sampled[['sample_id', 'taxon']].copy()
        sampled['source'] = 'ag3'
        selected.append(sampled)

    fun_df = af_meta[af_meta['taxon'] == 'funestus']
    sampled_fun = fun_df.sample(n=SAMPLES_PER_SPECIES, random_state=RANDOM_SEED)
    sampled_fun = sampled_fun[['sample_id', 'taxon']].copy()
    sampled_fun['source'] = 'af1'
    selected.append(sampled_fun)

    samples_df = pd.concat(selected, ignore_index=True)
    os.makedirs('../data', exist_ok=True)
    csv_path = '../data/sample_accessions.csv'
    samples_df[['sample_id', 'taxon']].to_csv(csv_path, index=False)
    print(f"Saved {len(samples_df)} samples to {csv_path}")
    return samples_df

def load_or_extract_data():
    X_path = '../data/X_features_colab.npy'
    y_path = '../data/y_labels_colab.npy'
    csv_path = '../data/sample_accessions.csv'

    if os.path.exists(X_path) and os.path.exists(y_path):
        print("Loading cached features...")
        X = np.load(X_path)
        y = np.load(y_path, allow_pickle=True)
        return X, y

    print("No cached features found, starting extraction...")
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        df = pd.read_csv(csv_path)
        if 'source' not in df.columns:
            df['source'] = df['taxon'].apply(lambda t: 'af1' if t == 'funestus' else 'ag3')
    else:
        df = generate_sample_list()

    X, y = run_extraction(df, output_dir="../data")
    return X, y

def main():
    print("=== Taxon Classifier Pipeline ===")
    X, y_labels = load_or_extract_data()
    print(f"Feature matrix shape: {X.shape}")

    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    classes = le.classes_

    os.makedirs("../models", exist_ok=True)
    joblib.dump(le, '../models/label_encoder.pkl')

    preds = train_and_cross_validate(X, y, classes, output_dir="../models")
    evaluate_predictions(y, preds, classes)
    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    main()
