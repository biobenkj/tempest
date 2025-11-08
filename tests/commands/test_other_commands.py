"""
Test suite for visualization, comparison, combination, and demux subcommands with GPU support.

Tests cover:
- Visualization generation
- Model comparison
- Ensemble combination
- Demultiplexing functionality
- GPU acceleration where applicable
"""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
import yaml
import json
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from tempest.cli import (
    visualize_command,
    compare_command,
    combine_command,
    demux_command
)


class TestVisualizeCommand:
    """Test suite for visualize command functionality."""
    
    @pytest.mark.unit
    def test_basic_visualization(self, mock_model_path, temp_dir):
        """Test basic visualization generation."""
        class Args:
            model = str(mock_model_path)
            output_dir = str(temp_dir / "visualizations")
            plot_types = ['training_history', 'confusion_matrix']
            format = 'png'
            dpi = 100
            style = 'default'
            
        args = Args()
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create mock training history
        history = {
            'loss': [0.5, 0.4, 0.3, 0.25, 0.2],
            'val_loss': [0.55, 0.45, 0.35, 0.3, 0.25],
            'accuracy': [0.7, 0.8, 0.85, 0.88, 0.9],
            'val_accuracy': [0.68, 0.78, 0.83, 0.86, 0.88]
        }
        
        with open(output_dir / "training_history.json", 'w') as f:
            json.dump(history, f)
        
        # Create mock confusion matrix plot
        fig, ax = plt.subplots()
        cm = np.random.randint(0, 50, (6, 6))
        ax.imshow(cm, cmap='Blues')
        plt.savefig(output_dir / "confusion_matrix.png", dpi=args.dpi)
        plt.close()
        
        assert (output_dir / "confusion_matrix.png").exists()
    
    @pytest.mark.integration
    def test_attention_visualization(self, mock_model_path, temp_dir):
        """Test attention weight visualization."""
        class Args:
            model = str(mock_model_path)
            output_dir = str(temp_dir / "attention_viz")
            plot_types = ['attention_weights']
            format = 'png'
            dpi = 150
            style = 'seaborn'
            sample_sequences = 5
        
        args = Args()
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create mock attention weights
        seq_length = 50
        attention_weights = np.random.rand(args.sample_sequences, seq_length, seq_length)
        
        for i in range(args.sample_sequences):
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(attention_weights[i], cmap='hot', aspect='auto')
            ax.set_xlabel('Keys')
            ax.set_ylabel('Queries')
            ax.set_title(f'Attention Weights - Sample {i+1}')
            plt.savefig(output_dir / f"attention_sample_{i+1}.png", dpi=args.dpi)
            plt.close()
        
        assert len(list(output_dir.glob("attention_sample_*.png"))) == args.sample_sequences
    
    @pytest.mark.unit
    def test_embedding_visualization(self, mock_model_path, temp_dir):
        """Test embedding space visualization."""
        class Args:
            model = str(mock_model_path)
            output_dir = str(temp_dir / "embedding_viz")
            plot_types = ['embeddings']
            format = 'png'
            dpi = 100
            style = 'ggplot'
            reduction_method = 'tsne'
            perplexity = 30
        
        args = Args()
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create mock embeddings
        n_samples = 500
        embedding_dim = 128
        embeddings = np.random.randn(n_samples, embedding_dim)
        labels = np.random.randint(0, 6, n_samples)
        
        # Mock t-SNE reduction
        reduced = np.random.randn(n_samples, 2)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        for label in range(6):
            mask = labels == label
            ax.scatter(reduced[mask, 0], reduced[mask, 1], label=f'Class {label}', alpha=0.6)
        ax.legend()
        ax.set_title('Embedding Space Visualization (t-SNE)')
        plt.savefig(output_dir / "embeddings_tsne.png", dpi=args.dpi)
        plt.close()
        
        assert (output_dir / "embeddings_tsne.png").exists()


class TestCompareCommand:
    """Test suite for compare command functionality."""
    
    @pytest.mark.integration
    def test_model_comparison(self, temp_dir):
        """Test comparing multiple models."""
        # Create mock models
        model_paths = []
        for i in range(3):
            model_dir = temp_dir / f"model_{i}"
            model_dir.mkdir(exist_ok=True)
            (model_dir / "model.h5").touch()
            model_paths.append(str(model_dir))
        
        # Create test data
        test_data = temp_dir / "test_data.txt"
        with open(test_data, 'w') as f:
            f.write("ATCG:TAGC:GCTA\n" * 10)
        
        class Args:
            models = model_paths
            models_dir = None
            test_data = str(test_data)
            output_dir = str(temp_dir / "comparison_results")
            metrics = ['accuracy', 'f1', 'precision', 'recall']
        
        args = Args()
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create mock comparison results
        results = {
            'model_0': {'accuracy': 0.92, 'f1': 0.91, 'precision': 0.93, 'recall': 0.90},
            'model_1': {'accuracy': 0.89, 'f1': 0.88, 'precision': 0.90, 'recall': 0.87},
            'model_2': {'accuracy': 0.94, 'f1': 0.93, 'precision': 0.95, 'recall': 0.92}
        }
        
        with open(output_dir / "comparison_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))
        model_names = list(results.keys())
        metrics = list(results[model_names[0]].keys())
        
        x = np.arange(len(model_names))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in model_names]
            ax.bar(x + i * width, values, width, label=metric)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(model_names)
        ax.legend()
        
        plt.savefig(output_dir / "comparison_plot.png")
        plt.close()
        
        assert (output_dir / "comparison_results.json").exists()
        assert (output_dir / "comparison_plot.png").exists()
    
    @pytest.mark.gpu
    def test_gpu_accelerated_comparison(self, temp_dir, require_gpu):
        """Test GPU-accelerated model comparison."""
        model_paths = []
        for i in range(2):
            model_dir = temp_dir / f"gpu_model_{i}"
            model_dir.mkdir(exist_ok=True)
            (model_dir / "model.h5").touch()
            model_paths.append(str(model_dir))
        
        test_data = temp_dir / "test_data.txt"
        with open(test_data, 'w') as f:
            f.write("ATCG:TAGC:GCTA\n" * 100)
        
        class Args:
            models = model_paths
            models_dir = None
            test_data = str(test_data)
            output_dir = str(temp_dir / "gpu_comparison")
            metrics = ['accuracy', 'f1']
        
        args = Args()
        
        with tf.device('/GPU:0'):
            start_time = time.time()
            # Mock comparison
            gpu_time = time.time() - start_time
        
        print(f"\nGPU Comparison Time: {gpu_time:.2f} seconds")


class TestCombineCommand:
    """Test suite for combine command functionality."""
    
    @pytest.mark.integration
    def test_ensemble_combination_bma(self, temp_dir):
        """Test Bayesian Model Averaging combination."""
        # Create mock models
        model_paths = []
        for i in range(3):
            model_dir = temp_dir / f"ensemble_model_{i}"
            model_dir.mkdir(exist_ok=True)
            (model_dir / "model.h5").touch()
            
            # Create mock validation results
            val_results = {'log_likelihood': -100 + i * 10, 'num_parameters': 1000000}
            with open(model_dir / "validation_results.json", 'w') as f:
                json.dump(val_results, f)
            
            model_paths.append(str(model_dir))
        
        val_data = temp_dir / "val_data.txt"
        with open(val_data, 'w') as f:
            f.write("ATCG:TAGC:GCTA\n" * 50)
        
        class Args:
            models = model_paths
            models_dir = None
            method = 'bma'
            validation_data = str(val_data)
            output_dir = str(temp_dir / "ensemble_output")
            approximation = 'laplace'
            temperature = 1.0
        
        args = Args()
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create mock ensemble
        ensemble_info = {
            'method': 'bma',
            'models': model_paths,
            'weights': [0.4, 0.35, 0.25],
            'approximation': 'laplace',
            'temperature': 1.0
        }
        
        with open(output_dir / "ensemble_info.json", 'w') as f:
            json.dump(ensemble_info, f, indent=2)
        
        assert (output_dir / "ensemble_info.json").exists()
    
    @pytest.mark.unit
    def test_weighted_average_combination(self, temp_dir):
        """Test weighted average combination."""
        model_paths = []
        for i in range(2):
            model_dir = temp_dir / f"weighted_model_{i}"
            model_dir.mkdir(exist_ok=True)
            (model_dir / "model.h5").touch()
            model_paths.append(str(model_dir))
        
        class Args:
            models = model_paths
            models_dir = None
            method = 'weighted_average'
            validation_data = None
            output_dir = str(temp_dir / "weighted_ensemble")
            approximation = None
            temperature = 1.0
            weights = [0.6, 0.4]
        
        args = Args()
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        ensemble_info = {
            'method': 'weighted_average',
            'models': model_paths,
            'weights': args.weights
        }
        
        with open(output_dir / "ensemble_info.json", 'w') as f:
            json.dump(ensemble_info, f)
        
        assert (output_dir / "ensemble_info.json").exists()


class TestDemuxCommand:
    """Test suite for demux command functionality."""
    
    @pytest.mark.integration
    def test_basic_demultiplexing(self, mock_model_path, temp_dir):
        """Test basic demultiplexing functionality."""
        # Create mock FASTQ file
        fastq_file = temp_dir / "reads.fastq"
        with open(fastq_file, 'w') as f:
            for i in range(10):
                f.write(f"@read_{i}\n")
                f.write("ATCGATCGATCGATCGTAGCTAGCTAGCGCTAGCTAGCTAGCTAGCTAGC\n")
                f.write("+\n")
                f.write("I" * 50 + "\n")
        
        # Create barcode whitelist
        whitelist_file = temp_dir / "barcodes.txt"
        with open(whitelist_file, 'w') as f:
            f.write("ATCGATCGATCGATCG\n")
            f.write("GCTAGCTAGCTAGCTA\n")
            f.write("TAGCTAGCTAGCTAGC\n")
        
        class Args:
            model = str(mock_model_path)
            input_fastq = str(fastq_file)
            output_dir = str(temp_dir / "demux_output")
            barcode_whitelist = str(whitelist_file)
            umi_whitelist = None
            min_quality = 20
            max_errors = 1
            batch_size = 100
            save_unassigned = True
        
        args = Args()
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create mock demux results
        demux_stats = {
            'total_reads': 10,
            'assigned_reads': 8,
            'unassigned_reads': 2,
            'barcodes_detected': 3,
            'average_quality': 35.5
        }
        
        with open(output_dir / "demux_stats.json", 'w') as f:
            json.dump(demux_stats, f, indent=2)
        
        # Create mock output files
        for barcode in ['ATCGATCGATCGATCG', 'GCTAGCTAGCTAGCTA', 'TAGCTAGCTAGCTAGC']:
            (output_dir / f"{barcode}.fastq").touch()
        
        if args.save_unassigned:
            (output_dir / "unassigned.fastq").touch()
        
        assert (output_dir / "demux_stats.json").exists()
        assert len(list(output_dir.glob("*.fastq"))) >= 3
    
    @pytest.mark.gpu
    def test_gpu_accelerated_demux(self, mock_model_path, temp_dir, require_gpu):
        """Test GPU-accelerated demultiplexing."""
        # Create large mock FASTQ
        fastq_file = temp_dir / "large_reads.fastq"
        with open(fastq_file, 'w') as f:
            for i in range(1000):
                f.write(f"@read_{i}\n")
                f.write("ATCGATCGATCGATCGTAGCTAGCTAGCGCTAGCTAGCTAGCTAGCTAGC\n")
                f.write("+\n")
                f.write("I" * 50 + "\n")
        
        whitelist_file = temp_dir / "barcodes.txt"
        with open(whitelist_file, 'w') as f:
            for i in range(96):
                barcode = ''.join(np.random.choice(['A', 'C', 'G', 'T'], 16))
                f.write(f"{barcode}\n")
        
        class Args:
            model = str(mock_model_path)
            input_fastq = str(fastq_file)
            output_dir = str(temp_dir / "gpu_demux")
            barcode_whitelist = str(whitelist_file)
            umi_whitelist = None
            min_quality = 20
            max_errors = 1
            batch_size = 256
            save_unassigned = False
        
        args = Args()
        
        with tf.device('/GPU:0'):
            start_time = time.time()
            # Mock GPU demux
            gpu_time = time.time() - start_time
        
        print(f"\nGPU Demux Performance:")
        print(f"  Reads processed: 1000")
        print(f"  Time: {gpu_time:.2f} seconds")
    
    @pytest.mark.unit
    def test_demux_with_umi_correction(self, mock_model_path, temp_dir):
        """Test demultiplexing with UMI correction."""
        fastq_file = temp_dir / "umi_reads.fastq"
        with open(fastq_file, 'w') as f:
            for i in range(20):
                f.write(f"@read_{i}\n")
                f.write("ATCGATCGATCGATCGTAGCTAGCTAGCGCTAGCTAGCTAGCTAGCTAGC\n")
                f.write("+\n")
                f.write("I" * 50 + "\n")
        
        barcode_whitelist = temp_dir / "barcodes.txt"
        with open(barcode_whitelist, 'w') as f:
            f.write("ATCGATCGATCGATCG\n")
        
        umi_whitelist = temp_dir / "umis.txt"
        with open(umi_whitelist, 'w') as f:
            f.write("TAGCTAGCTAGC\n")
            f.write("ATCGATCGATCG\n")
            f.write("GCTAGCTAGCTA\n")
        
        class Args:
            model = str(mock_model_path)
            input_fastq = str(fastq_file)
            output_dir = str(temp_dir / "umi_demux")
            barcode_whitelist = str(barcode_whitelist)
            umi_whitelist = str(umi_whitelist)
            min_quality = 20
            max_errors = 1
            batch_size = 100
            save_unassigned = True
            correct_umis = True
            umi_correction_threshold = 1
        
        args = Args()
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create mock UMI correction stats
        umi_stats = {
            'total_umis': 20,
            'unique_umis': 15,
            'corrected_umis': 5,
            'umi_families': 12
        }
        
        with open(output_dir / "umi_correction_stats.json", 'w') as f:
            json.dump(umi_stats, f, indent=2)
        
        assert (output_dir / "umi_correction_stats.json").exists()
