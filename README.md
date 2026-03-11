# Taxon Classifier

This is a prototype for a FASTQ-based mosquito species classifier. It takes in raw sequencing reads and tells you which Anopheles species the sample belongs to — gambiae complex, arabiensis, funestus, or stephensi. No variant calling needed, just k-mer counting and a LightGBM model.

## How to run

    cd /mnt/d/taxon/Taxon-Classifier
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cd src
    python -m evaluation.test ../data/gambiae_test.fastq


This will read 1000 sequences from the FASTQ file, extract 6-mer frequencies using a hashing vectorizer (4096 bins), and predict the species along with confidence scores.

## Retraining

The FASTQ model can be retrained from scratch. It streams reads directly from ENA so no manual downloads are needed. Takes about 10-15 minutes.

    cd src
    python -m scripts.train_on_fastq

## Plots

The visualization notebook (notebooks/visualization.ipynb) has t-SNE, confusion matrices, feature importance, and model comparison plots.

## About the two models

There are two saved models in the models/ folder. lightgbm.pkl was trained on BAM reads pulled from MalariaGEN and gets 99% F1 on cross-validation. lightgbm_fastq.pkl was trained on FASTQ reads from ENA. The BAM model works great for evaluation but doesn't predict well on raw FASTQ files because the original features were extracted on Colab with slightly different parameters. The FASTQ model uses the same extraction code as test.py so predictions are consistent and correct.

