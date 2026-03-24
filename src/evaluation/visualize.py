import os
import json
import random
import glob
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import HashingVectorizer
import warnings

warnings.filterwarnings('ignore')

N_FEATURES = 1048576
COLORS = {'funestus': '#FF6B6B', 'gambiae_complex': '#4ECDC4', 'stephensi': '#45B7D1'}

def _get_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "../../data"))
    models_dir = os.path.abspath(os.path.join(script_dir, "../../models"))
    test_dir = os.path.join(data_dir, "test_samples")
    return data_dir, models_dir, test_dir

def _parse_fastq(filepath, n_reads=10000):
    seqs = []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i % 4 == 1:
                s = line.strip()
                if len(s) > 50:
                    seqs.append(s)
                if len(seqs) >= n_reads:
                    break
    return seqs

def _seqs_to_vector(seqs, k):
    vectorizer = HashingVectorizer(
        analyzer='char', ngram_range=(k, k),
        n_features=N_FEATURES, norm=None, alternate_sign=False
    )
    hashed = vectorizer.transform([" ".join(seqs)])
    vec = np.zeros(N_FEATURES, dtype=np.float32)
    total = sum(hashed.data)
    if total > 0:
        for idx, val in zip(hashed.indices, hashed.data):
            vec[idx] = val / total
    return vec


def _poison_reads(seqs, error_rate):
    bases = ['A', 'T', 'C', 'G']
    poisoned = []
    for seq in seqs:
        chars = list(seq)
        for j in range(len(chars)):
            if random.random() < error_rate:
                chars[j] = random.choice([b for b in bases if b != chars[j]])
        poisoned.append("".join(chars))
    return poisoned

def plot_tsne(k=15, perplexity=30, random_state=42):
    data_dir, _, _ = _get_paths()
    X = np.load(os.path.join(data_dir, f"X_fastq_k{k}.npy"))
    y = np.load(os.path.join(data_dir, f"y_fastq_k{k}.npy"))

    print("Running TruncatedSVD (1M -> 50 dims)...")
    svd = TruncatedSVD(n_components=50, random_state=random_state)
    X_svd = svd.fit_transform(X)

    print("Running t-SNE (50 -> 2 dims)...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    X_2d = tsne.fit_transform(X_svd)

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(6, 5))

    for species in np.unique(y):
        mask = y == species
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=COLORS.get(species, '#999'), label=species,
                   s=30, alpha=0.8, edgecolors='white', linewidth=0.5)

    ax.set_title(f"t-SNE Visualization of K-mer Profiles (k={k})", fontsize=14, fontweight='bold')
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(fontsize=11, loc='best')
    plt.tight_layout()
    plt.show()
    return fig

def plot_k_comparison(k_values=None):
    if k_values is None:
        k_values = [15, 21]

    _, models_dir, _ = _get_paths()
    stats = []
    for k in k_values:
        path = os.path.join(models_dir, f"train_stats_k{k}.json")
        if not os.path.exists(path):
            print(f"Stats not found for k={k}: {path}")
            return
        with open(path) as f:
            stats.append(json.load(f))

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    k_labels = [f"k={s['k']}" for s in stats]
    bar_colors = ['#4ECDC4', '#45B7D1', '#FF6B6B']

    ax = axes[0]
    x = np.arange(len(k_labels))
    width = 0.35
    acc_vals = [s['accuracy'] for s in stats]
    f1_vals = [s['macro_f1'] for s in stats]
    ax.bar(x - width/2, acc_vals, width, label='Accuracy', color='#4ECDC4', edgecolor='white')
    ax.bar(x + width/2, f1_vals, width, label='Macro-F1', color='#45B7D1', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(k_labels)
    ax.set_ylim(0.90, 1.01)
    ax.set_ylabel("Score")
    ax.set_title("Classification Performance", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    for i, (a, f) in enumerate(zip(acc_vals, f1_vals)):
        ax.text(i - width/2, a + 0.002, f"{a:.3f}", ha='center', fontsize=9)
        ax.text(i + width/2, f + 0.002, f"{f:.3f}", ha='center', fontsize=9)

    ax = axes[1]
    times = [s['time_total_s'] / 3600 for s in stats]
    bars = ax.bar(k_labels, times, color=['#4ECDC4', '#45B7D1'][:len(k_labels)], edgecolor='white')
    ax.set_ylabel("Hours")
    ax.set_title("Total Training Time", fontsize=13, fontweight='bold')
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{t:.1f}h", ha='center', fontsize=10)

    ax = axes[2]
    ratios = [4**s['k'] / N_FEATURES for s in stats]
    ratio_labels = [f"{r:.0e}" for r in ratios]
    bars = ax.bar(k_labels, ratios, color=['#4ECDC4', '#45B7D1'][:len(k_labels)], edgecolor='white')
    ax.set_ylabel("Compression Ratio (4^k / 2^20)")
    ax.set_title("Hash Compression Ratio", fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    for bar, label in zip(bars, ratio_labels):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.5, label, ha='center', fontsize=10)

    plt.suptitle("K-mer Size Comparison", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    return fig

def plot_poison_comparison(k_values=None, error_rates=None, n_reads=10000, seed=42):
    if k_values is None:
        k_values = [15, 21]
    if error_rates is None:
        error_rates = [0.0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    _, models_dir, test_dir = _get_paths()
    fastq_files = sorted(glob.glob(os.path.join(test_dir, "*.fastq")))
    if not fastq_files:
        print("No FASTQ files found in data/test_samples/")
        return

    all_seqs = {}
    species_names = []
    for fp in fastq_files:
        name = "_".join(os.path.basename(fp).split("_")[:-1])
        species_names.append(name)
        all_seqs[name] = _parse_fastq(fp, n_reads)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, len(k_values), figsize=(6 * len(k_values), 5), sharey=True)
    if len(k_values) == 1:
        axes = [axes]

    rates_pct = [r * 100 for r in error_rates]

    for ax, k in zip(axes, k_values):
        model_path = os.path.join(models_dir, f"lightgbm_fastq_k{k}.pkl")
        encoder_path = os.path.join(models_dir, f"label_encoder_fastq_k{k}.pkl")
        if not os.path.exists(model_path):
            print(f"Model not found for k={k}")
            continue

        model = joblib.load(model_path)
        le = joblib.load(encoder_path)
        random.seed(seed)

        for name in species_names:
            confs = []
            for rate in error_rates:
                seqs = all_seqs[name]
                if rate > 0:
                    seqs = _poison_reads(seqs, rate)

                vec = _seqs_to_vector(seqs, k)
                X_test = csr_matrix(vec.reshape(1, -1))
                pred = le.inverse_transform(model.predict(X_test))[0]
                conf = model.predict_proba(X_test)[0].max() * 100
                confs.append(conf if pred == name else 0)

            ax.plot(rates_pct, confs, 'o-', color=COLORS.get(name, '#999'),
                    label=name, linewidth=2, markersize=6)

        ax.axvline(x=1, color='green', linestyle='--', alpha=0.5, label='Illumina error rate')
        ax.set_xlabel("Base Substitution Error Rate (%)", fontsize=12)
        ax.set_ylabel("Prediction Confidence (%)", fontsize=12)
        ax.set_title(f"Noise Robustness (k={k})", fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)

    plt.suptitle("Poison Test: k=15 vs k=21", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    return fig

def plot_depth_curve(k=15, depths=None):
    if depths is None:
        depths = [50, 100, 250, 500, 1000, 2500, 5000, 10000]

    _, models_dir, test_dir = _get_paths()
    model = joblib.load(os.path.join(models_dir, f"lightgbm_fastq_k{k}.pkl"))
    le = joblib.load(os.path.join(models_dir, f"label_encoder_fastq_k{k}.pkl"))

    fastq_files = sorted(glob.glob(os.path.join(test_dir, "*.fastq")))
    if not fastq_files:
        print("No FASTQ files found in data/test_samples/")
        return

    results = {}
    for fp in fastq_files:
        name = "_".join(os.path.basename(fp).split("_")[:-1])
        all_seqs = _parse_fastq(fp, max(depths))

        confs = []
        for d in depths:
            seqs = all_seqs[:d]
            vec = _seqs_to_vector(seqs, k)
            X_test = csr_matrix(vec.reshape(1, -1))
            pred = le.inverse_transform(model.predict(X_test))[0]
            conf = model.predict_proba(X_test)[0].max() * 100
            confs.append(conf if pred == name else 0)
        results[name] = confs

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6, 5))

    for name, confs in results.items():
        ax.plot(depths, confs, 'o-', color=COLORS.get(name, '#999'),
                label=name, linewidth=2, markersize=6)

    ax.axvline(x=1000, color='red', linestyle='--', alpha=0.6)
    ax.annotate('Min threshold\n(1000 reads)', xy=(1000, 50), fontsize=10, color='red', ha='center')
    ax.set_xlabel("Number of Reads", fontsize=12)
    ax.set_ylabel("Prediction Confidence (%)", fontsize=12)
    ax.set_title(f"Read Depth vs Classification Confidence (k={k})", fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    plt.tight_layout()
    plt.show()
    return fig
