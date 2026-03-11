import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.model_selection import learning_curve, StratifiedKFold
from itertools import product

def plot_tsne(X, y, classes):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X)

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6']
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, name in enumerate(classes):
        mask = (y == i)
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=colors[i % len(colors)], label=name,
                   alpha=0.7, s=30, edgecolors='white', linewidth=0.3)
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('t-SNE of 6-mer Frequencies')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrices(y_true, preds, classes):
    model_names = list(preds.keys())
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, model_names):
        cm = confusion_matrix(y_true, preds[name]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes, ax=ax,
                    annot_kws={"size": 8})
        ax.set_title(name, fontsize=10)
        ax.set_ylabel('True Label', fontsize=8)
        ax.set_xlabel('Predicted Label', fontsize=8)
        ax.tick_params(labelsize=7)
    plt.suptitle('Confusion Matrices', fontsize=11)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(importances, top_n=20, k=6):
    bases = 'ACGT'
    kmer_names = [''.join(p) for p in product(bases, repeat=k)]
    top_idx = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(range(top_n), importances[top_idx], color='steelblue')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([kmer_names[i] for i in top_idx], fontsize=7, fontfamily='monospace')
    ax.set_xlabel('Importance', fontsize=9)
    ax.set_title(f'Top {top_n} Discriminative {k}-mers', fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_recall_heatmap(y_true, preds, classes):
    from sklearn.metrics import recall_score
    names = list(preds.keys())
    recall_data = []
    for name in names:
        y_pred = preds[name]['y_pred']
        per_class = recall_score(y_true, y_pred, average=None)
        recall_data.append(per_class)
    recall_matrix = np.array(recall_data)
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(recall_matrix, annot=True, fmt='.2f', cmap='YlGn',
                xticklabels=classes, yticklabels=names, ax=ax,
                vmin=0.8, vmax=1.0, annot_kws={"size": 9})
    ax.set_title('Per-Class Recall by Model', fontsize=10)
    ax.set_xlabel('Species', fontsize=9)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.show()

def load_models_and_predict(X, y, classes, model_dir="../models"):
    model_list = ['Random Forest', 'LightGBM']
    preds = {}
    for name in model_list:
        fname = name.replace(" ", "_").lower() + ".pkl"
        path = os.path.join(model_dir, fname)
        if os.path.exists(path):
            model = joblib.load(path)
            y_pred = model.predict(X)
            try:
                y_proba = model.predict_proba(X)
            except Exception:
                y_proba = np.zeros((len(y), len(classes)))
            preds[name] = {'y_pred': y_pred, 'y_proba': y_proba, 'model_obj': model}
        else:
            print(f"{name} model not found at {path}, run main.py first.")
    return preds
