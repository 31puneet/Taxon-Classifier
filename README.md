#  Taxon-Classifier

A FASTQ-based machine learning classifier that identifies **Anopheles** mosquito species directly from raw sequencing reads - no alignment or variant calling required.

## What is it?

Taxonomic identification of *Anopheles* mosquitoes is critical for malaria control but challenging because many species are morphologically identical. Misidentification leads to wasted sequencing resources and flawed epidemiological insights.

This tool classifies mosquito samples into **three major species groups** using only raw FASTQ reads:

| Species | Region |
|---|---|
| *An. gambiae* complex | Sub-Saharan Africa |
| *An. funestus* | Sub-Saharan Africa |
| *An. stephensi* | South/Southeast Asia |

## Features

- **No alignment needed** — works directly on raw FASTQ reads streamed from [ENA](https://www.ebi.ac.uk/ena)
- **K-mer frequency profiles** — uses `HashingVectorizer` to extract character n-gram features
- **LightGBM classifier** — fast, handles sparse high-dimensional data natively
- **Multiple k-mer sizes** — tested with k=15 (98.3% accuracy) and k=21 (97.5% accuracy)
- **Noise robust** — maintains 100% accuracy up to 10% base substitution error rate
- **Low read depth** — reliable classification with as few as 1,000 reads.

## Results

| Metric | k=15 | k=21 |
|---|---|---|
| Accuracy | **98.3%** | 97.5% |
| Macro-F1 | **0.983** | 0.975 |
| Compression Ratio | 10³:1 | 10⁶:1 |

## How it works

```
Raw FASTQ reads → K-mer extraction → Hashing (2²⁰ bins) → LightGBM → Species prediction
```

1. **Extract**: Stream FASTQ reads from ENA, extract k-mer frequency profiles using `HashingVectorizer`
2. **Train**: Train a LightGBM model with stratified cross-validation
3. **Evaluate**: Predict species from unseen FASTQ files with confidence scores




## Visualizations

Open `notebooks/visualization.ipynb` in Jupyter to generate plots:

```python
from evaluation.visualize import plot_tsne, plot_k_comparison, plot_poison_comparison, plot_depth_curve

plot_tsne(k=15)               # t-SNE species clustering
plot_k_comparison()           # k=15 vs k=21 comparison
plot_poison_comparison()      # Noise robustness curves
plot_depth_curve(k=15)        # Read depth vs confidence
```
