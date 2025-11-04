"""
I/O utilities for Tempest.

Handles reading and writing various file formats used in the pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import json


def load_pwm(pwm_file: str) -> np.ndarray:
    """
    Load a Position Weight Matrix from file.
    
    Expected format (tab-separated):
    pos  base  prob
    1    A     0.944
    1    C     0.026
    ...
    
    Args:
        pwm_file: Path to PWM file
        
    Returns:
        PWM matrix of shape (length, 4) for bases [A, C, G, T]
    """
    df = pd.read_csv(pwm_file, sep='\t')
    
    # Get PWM length
    pwm_length = df['pos'].max()
    
    # Initialize matrix
    pwm = np.zeros((pwm_length, 4))
    
    # Base to index mapping
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Fill in probabilities
    for _, row in df.iterrows():
        pos = int(row['pos']) - 1  # Convert to 0-indexed
        base = row['base']
        prob = float(row['prob'])
        
        if base in base_to_idx:
            pwm[pos, base_to_idx[base]] = prob
    
    return pwm


def save_pwm(pwm: np.ndarray, output_file: str):
    """
    Save a PWM to file.
    
    Args:
        pwm: PWM matrix of shape (length, 4)
        output_file: Output file path
    """
    bases = ['A', 'C', 'G', 'T']
    rows = []
    
    for pos in range(pwm.shape[0]):
        for base_idx, base in enumerate(bases):
            rows.append({
                'pos': pos + 1,
                'base': base,
                'prob': pwm[pos, base_idx]
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, sep='\t', index=False)


def load_acc_priors(priors_file: str, model_name: Optional[str] = None) -> Tuple[List[str], List[float]]:
    """
    Load ACC sequence priors from file.
    
    Expected format (tab-separated):
    model_name  sequence  frequency
    model1      ACCGGG    0.45
    model1      ACCGGC    0.35
    ...
    
    Args:
        priors_file: Path to priors file
        model_name: Model name to filter by (optional)
        
    Returns:
        Tuple of (sequences, frequencies)
    """
    df = pd.read_csv(priors_file, sep='\t', comment='#')
    
    if model_name is not None:
        df = df[df['model_name'] == model_name]
        
        if df.empty:
            raise ValueError(f"No ACC priors found for model '{model_name}'")
    
    # Check for N/A marker (skip priors)
    if (df['sequence'] == 'N/A').any():
        return None, None
    
    sequences = df['sequence'].tolist()
    frequencies = df['frequency'].tolist()
    
    # Normalize frequencies
    freq_sum = sum(frequencies)
    if abs(freq_sum - 1.0) > 0.01:
        frequencies = [f / freq_sum for f in frequencies]
    
    return sequences, frequencies


def load_barcodes(barcode_file: str) -> List[str]:
    """
    Load barcode sequences from file.
    
    Expected format: One barcode per line
    
    Args:
        barcode_file: Path to barcode file
        
    Returns:
        List of barcode sequences
    """
    with open(barcode_file, 'r') as f:
        barcodes = [line.strip() for line in f if line.strip()]
    return barcodes


def load_fastq(fastq_file: str, max_reads: Optional[int] = None) -> Iterator[SeqRecord]:
    """
    Load sequences from FASTQ file.
    
    Args:
        fastq_file: Path to FASTQ file
        max_reads: Maximum number of reads to load (None for all)
        
    Yields:
        SeqRecord objects
    """
    count = 0
    for record in SeqIO.parse(fastq_file, 'fastq'):
        yield record
        count += 1
        if max_reads is not None and count >= max_reads:
            break


def load_fasta(fasta_file: str, max_reads: Optional[int] = None) -> Iterator[SeqRecord]:
    """
    Load sequences from FASTA file.
    
    Args:
        fasta_file: Path to FASTA file
        max_reads: Maximum number of reads to load (None for all)
        
    Yields:
        SeqRecord objects
    """
    count = 0
    for record in SeqIO.parse(fasta_file, 'fasta'):
        yield record
        count += 1
        if max_reads is not None and count >= max_reads:
            break


def save_annotations_json(annotations: List[Dict], output_file: str):
    """
    Save annotations to JSON file.
    
    Args:
        annotations: List of annotation dictionaries
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)


def save_annotations_tsv(annotations: List[Dict], output_file: str):
    """
    Save annotations to TSV file.
    
    Args:
        annotations: List of annotation dictionaries
        output_file: Output file path
    """
    # Flatten annotations for TSV format
    rows = []
    for annot in annotations:
        read_name = annot.get('read_name', '')
        read_length = annot.get('read_length', 0)
        
        for label, regions in annot.items():
            if label in ['read_name', 'read_length', 'architecture', 'reason', 'orientation']:
                continue
            
            if isinstance(regions, dict) and 'Starts' in regions:
                for start, end in zip(regions['Starts'], regions['Ends']):
                    rows.append({
                        'read_name': read_name,
                        'read_length': read_length,
                        'label': label,
                        'start': start,
                        'end': end,
                        'length': end - start
                    })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, sep='\t', index=False)


def save_annotations_gff(annotations: List[Dict], output_file: str, source: str = 'Tempest'):
    """
    Save annotations to GFF3 file.
    
    Args:
        annotations: List of annotation dictionaries
        output_file: Output file path
        source: Source name for GFF
    """
    with open(output_file, 'w') as f:
        f.write("##gff-version 3\n")
        
        for annot in annotations:
            seqid = annot.get('read_name', '')
            
            for label, regions in annot.items():
                if label in ['read_name', 'read_length', 'architecture', 'reason', 'orientation']:
                    continue
                
                if isinstance(regions, dict) and 'Starts' in regions:
                    for idx, (start, end) in enumerate(zip(regions['Starts'], regions['Ends'])):
                        # GFF uses 1-based coordinates
                        f.write(f"{seqid}\t{source}\t{label}\t{start+1}\t{end}\t.\t+\t.\t")
                        f.write(f"ID={label}_{idx};Name={label}\n")


def load_seq_orders(seq_orders_file: str, model_name: str) -> Tuple[List[str], List[str], List[str], List[str], str]:
    """
    Load sequence orders from TSV file (from original Tranquillyzer format).
    
    Args:
        seq_orders_file: Path to seq_orders.tsv
        model_name: Model name to load
        
    Returns:
        Tuple of (sequence_order, sequences, barcodes, UMIs, strand)
    """
    df = pd.read_csv(seq_orders_file, sep='\t')
    
    row = df[df['model_name'] == model_name]
    if row.empty:
        raise ValueError(f"Model '{model_name}' not found in {seq_orders_file}")
    
    row = row.iloc[0]
    
    # Parse lists from string format
    sequence_order = eval(row['sequence_order'])
    sequences = eval(row['sequences'])
    barcodes = eval(row['barcodes'])
    UMIs = eval(row['UMIs'])
    strand = row['strand']
    
    return sequence_order, sequences, barcodes, UMIs, strand


def ensure_dir(directory: str):
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_base_to_index() -> Dict[str, int]:
    """
    Get mapping from nucleotide bases to indices.
    
    Returns:
        Dictionary mapping bases to indices
    """
    return {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}


def get_index_to_base() -> Dict[int, str]:
    """
    Get mapping from indices to nucleotide bases.
    
    Returns:
        Dictionary mapping indices to bases
    """
    return {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}
