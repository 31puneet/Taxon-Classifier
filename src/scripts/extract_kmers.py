import argparse
import os
import csv
import io
import zlib
import random
import socket
import urllib.request
import numpy as np
import warnings
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.feature_extraction.text import HashingVectorizer
from tqdm import tqdm

warnings.filterwarnings('ignore')

N_FEATURES = 1048576
N_READS = 10000
STREAM_BYTES = 5 * 1024 * 1024

ENA_TAX_IDS = {
    'funestus': 62324,
    'stephensi': 30069,
}


def accession_to_fastq_url(acc):
    dir1 = acc[:6]
    if len(acc) <= 9:
        return f"https://ftp.sra.ebi.ac.uk/vol1/fastq/{dir1}/{acc}/{acc}_1.fastq.gz"
    dir2 = acc[9:].zfill(3)
    return f"https://ftp.sra.ebi.ac.uk/vol1/fastq/{dir1}/{dir2}/{acc}/{acc}_1.fastq.gz"


def get_urls_from_catalog(csv_path, limit):
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        sample_runs = {}
        for row in reader:
            sid = row['sample_id']
            if sid not in sample_runs:
                sample_runs[sid] = []
            sample_runs[sid].append(row['ena_run'])

    sample_ids = list(sample_runs.keys())
    random.shuffle(sample_ids)

    urls = []
    for sid in sample_ids[:limit * 3]:
        acc = sample_runs[sid][0]
        urls.append(accession_to_fastq_url(acc))
    return urls


def get_urls_from_ena(tax_id, limit):
    url = (
        f"https://www.ebi.ac.uk/ena/portal/api/search?"
        f"result=read_run&"
        f"query=tax_eq({tax_id})%20AND%20library_strategy=%22WGS%22&"
        f"fields=run_accession,fastq_ftp,read_count&"
        f"limit={limit * 3}&"
        f"format=tsv"
    )
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        lines = resp.read().decode().strip().split('\n')

    if len(lines) < 2:
        return []

    header = lines[0].split('\t')
    urls = []
    for line in lines[1:]:
        fields = line.split('\t')
        row = dict(zip(header, fields))
        if int(row.get('read_count', 0)) > 500000:
            ftp_list = row['fastq_ftp'].split(';')
            pick = [u for u in ftp_list if u.endswith('_1.fastq.gz')]
            if not pick:
                pick = ftp_list
            if pick:
                urls.append('https://' + pick[0])
    return urls


def stream_reads(url, n_reads=10000):
    socket.setdefaulttimeout(30)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read(STREAM_BYTES)

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
    except Exception:
        return []


def process_single_url(args):
    url, species, k = args
    seqs = stream_reads(url, N_READS)

    if len(seqs) < (N_READS * 0.5):
        return None, None

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
    return vec, species


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=15)
    parser.add_argument('--samples', type=int, default=40)
    args = parser.parse_args()

    k = args.k
    samples_per = args.samples
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "../../data"))
    os.makedirs(data_dir, exist_ok=True)

    ag3_csv = os.path.join(data_dir, "ena_runs.csv")

    species_config = [
        ('gambiae_complex', 'catalog', ag3_csv),
        ('funestus', 'ena', ENA_TAX_IDS['funestus']),
        ('stephensi', 'ena', ENA_TAX_IDS['stephensi']),
    ]

    X_all, y_all = [], []
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    executor = ProcessPoolExecutor(max_workers=max_workers)

    for species, source, ref in species_config:
        if source == 'catalog':
            urls = get_urls_from_catalog(ref, samples_per)
        else:
            urls = get_urls_from_ena(ref, samples_per)

        if not urls:
            continue

        work_items = [(url, species, k) for url in urls]
        valid_samples = 0
        future_to_url = {executor.submit(process_single_url, item): item for item in work_items}

        with tqdm(total=samples_per, desc=f"{species:>18}") as pbar:
            for future in as_completed(future_to_url):
                if valid_samples >= samples_per:
                    break
                try:
                    vec, spec = future.result()
                    if vec is not None:
                        X_all.append(vec)
                        y_all.append(spec)
                        valid_samples += 1
                        pbar.update(1)
                except Exception:
                    pass

        for f in future_to_url:
            f.cancel()

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all)

    np.save(os.path.join(data_dir, f"X_fastq_k{k}.npy"), X)
    np.save(os.path.join(data_dir, f"y_fastq_k{k}.npy"), y)

    os._exit(0)


if __name__ == "__main__":
    main()