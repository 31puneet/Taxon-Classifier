import os
import urllib.request
import gzip
import malariagen_data
import pysam
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.feature_extraction.text import HashingVectorizer

K_VALUE = 6
N_FEATURES = 4096
SAMPLE_FRACTION = 0.05
N_READS_STEPHENSI = 1000

def kmer_counts_to_frequency_vector(counts_dict, n_features):
    vector = np.zeros(n_features)
    total_kmers = sum(counts_dict.values())
    if total_kmers == 0:
        return vector
    for idx, count in counts_dict.items():
        vector[idx] = count / total_kmers
    return vector

def download_and_extract_stephensi(vectorizer, output_dir="../data/stephensi_fastq"):
    # Only runs if cached .npy files are not present
    os.makedirs(output_dir, exist_ok=True)
    ena_url = (
        "https://www.ebi.ac.uk/ena/portal/api/search?"
        "result=read_run&"
        "query=tax_eq(30069)%20AND%20library_strategy=%22WGS%22&"
        "fields=run_accession,instrument_model,read_count,fastq_ftp&"
        "limit=125&"
        "format=tsv"
    )
    response = urllib.request.urlopen(ena_url)
    lines = response.read().decode().strip().split('\n')
    header = lines[0].split('\t')
    stephensi_features = []
    stephensi_labels = []
    for line in lines[1:]:
        fields = line.split('\t')
        row = dict(zip(header, fields))
        if int(row.get('read_count', 0)) > 100000:
            ftp_urls = row['fastq_ftp'].split(';')
            single_url = [u for u in ftp_urls if not u.endswith('_1.fastq.gz') and not u.endswith('_2.fastq.gz')]
            paired_url = [u for u in ftp_urls if u.endswith('_1.fastq.gz')]
            url = single_url[0] if single_url else (paired_url[0] if paired_url else ftp_urls[0])
            url = 'https://' + url
            acc = row['run_accession']
            gz_path = os.path.join(output_dir, f"{acc}.fastq.gz")
            fq_path = os.path.join(output_dir, f"{acc}.fastq")
            
            try:
                if not os.path.exists(gz_path) and not os.path.exists(fq_path):
                    urllib.request.urlretrieve(url, gz_path)
                
                if os.path.exists(gz_path) and not os.path.exists(fq_path):
                    with gzip.open(gz_path, 'rt') as f_in, open(fq_path, 'w') as f_out:
                        line_count = 0
                        for fl in f_in:
                            f_out.write(fl)
                            line_count += 1
                            if line_count >= N_READS_STEPHENSI * 4:
                                break
                    os.remove(gz_path)
                
                if os.path.exists(fq_path):
                    seqs = []
                    reads_processed = 0
                    for record in SeqIO.parse(fq_path, "fastq"):
                        seqs.append(str(record.seq))
                        reads_processed += 1
                        if reads_processed >= N_READS_STEPHENSI:
                            break
                            
                    if seqs:
                        seq_text = " ".join(seqs)
                        hashed_counts = vectorizer.transform([seq_text])
                        vector_dict = dict(zip(hashed_counts.indices, hashed_counts.data))
                        freq_vec = kmer_counts_to_frequency_vector(vector_dict, N_FEATURES)
                        
                        stephensi_features.append(freq_vec)
                        stephensi_labels.append('stephensi')
                        
                    os.remove(fq_path)
            except Exception:
                pass
                
    return stephensi_features, stephensi_labels

def run_extraction(df_sampled, output_dir="../data"):
    ag3 = malariagen_data.Ag3("gs://vo_agam_release/")
    af1 = malariagen_data.Af1("gs://vo_afun_release/")
    
    vectorizer = HashingVectorizer(analyzer='char', ngram_range=(K_VALUE, K_VALUE),
                                   n_features=N_FEATURES, norm=None, alternate_sign=False)
    
    X_features_list = []
    y_labels_list = []
 
    for i, row in df_sampled.iterrows():
        sample_id = row['sample_id']
        source = row['source']
        taxon = row['taxon']
        
        try:
            bam_url = None
            if source == 'ag3':
                bam_url = ag3.sample_bams(sample_id)['alignments'][0]
            elif source == 'af1':
                bam_url = af1.sample_bams(sample_id)['alignments'][0]
                
            if bam_url:
                with pysam.AlignmentFile(bam_url, "rb") as bam:
                    total_reads = bam.mapped + bam.unmapped
                    sampled_read_count = int(total_reads * SAMPLE_FRACTION)
                    
                    seqs = []
                    for j, read in enumerate(bam.fetch(until_eof=True)):
                        if j >= sampled_read_count:
                            break
                        seqs.append(read.query_sequence)
                    
                    if seqs:
                        seq_text = " ".join(seqs)
                        hashed_counts = vectorizer.transform([seq_text])
                        vector_dict = dict(zip(hashed_counts.indices, hashed_counts.data))
                        freq_vec = kmer_counts_to_frequency_vector(vector_dict, N_FEATURES)
                        
                        X_features_list.append(freq_vec)
                        y_labels_list.append(taxon)
        except Exception:
            pass
            
    step_features, step_labels = download_and_extract_stephensi(vectorizer, output_dir)
    
    X_features_list.extend(step_features)
    y_labels_list.extend(step_labels)
    
    X_features = np.array(X_features_list)
    y_labels = np.array(y_labels_list)
            
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_features.npy'), X_features)
    np.save(os.path.join(output_dir, 'y_labels.npy'), y_labels)
    
    return X_features, y_labels
