# Tempest Workflow Diagram

## Overall Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TEMPEST PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  1. SIMULATION   │  Generate synthetic training data
└────────┬─────────┘
         │
         │ tempest.data.simulator
         │ - Sequence structure from config
         │ - ACC priors from real data
         │ - Error injection
         │
         ▼
┌────────────────────┐
│  Simulated Data    │
│  - Sequences       │
│  - Labels          │
└────────┬───────────┘
         │
         │
         ▼
┌──────────────────┐
│  2. TRAINING     │  Train models with length-constrained CRF
└────────┬─────────┘
         │
         │ tempest.training.trainer
         │ - CNN-BiLSTM-CRF architecture
         │ - Length constraints (UMI, ACC, etc.)
         │ - PWM-based priors for ACC
         │
         ▼
┌────────────────────┐
│  Trained Models    │
│  - Model weights   │
│  - Architecture    │
└────────┬───────────┘
         │
         │
         ▼
┌──────────────────┐
│  3. ENSEMBLE     │  Bayesian Model Averaging
└────────┬─────────┘
         │
         │ tempest.training.ensemble
         │ - Train N models with variation
         │ - Weight by validation performance
         │ - Combine predictions
         │
         ▼
┌────────────────────┐
│  Ensemble Model    │
│  - Multiple models │
│  - BMA weights     │
└────────┬───────────┘
         │
         │
         ▼
┌──────────────────┐
│  4. INFERENCE    │  Annotate real data
└────────┬─────────┘
         │
         │ tempest.inference.annotator
         │ - Load FASTQ files
         │ - Run ensemble prediction
         │ - Post-process (barcode correction, dedup)
         │
         ▼
┌────────────────────┐
│  Annotations       │
│  - JSON/TSV/GFF    │
│  - Visualizations  │
└────────────────────┘
```

## Module Dependencies

```
┌─────────────┐
│   Config    │  YAML/JSON configuration files
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│    Data     │────▶│  Training   │
│  Simulator  │     │   Trainer   │
└─────────────┘     └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Ensemble   │
                    │     BMA     │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  Inference  │────▶│Visualization│
                    │  Annotator  │     │   Plots     │
                    └─────────────┘     └─────────────┘
```

## Core Components

```
┌────────────────────────────────────────────────────────────────┐
│                    NEURAL NETWORK ARCHITECTURE                 │
└────────────────────────────────────────────────────────────────┘

Input Sequence: "AGATCGGAAGAGCGTCGTGTAGGGATCCCACGTACTGACGTACT..."
       │
       ▼
┌──────────────┐
│  Embedding   │  Convert bases to dense vectors
│   Layer      │  [A,C,G,T,N] → ℝ¹²⁸
└──────┬───────┘
       │
       ▼
┌──────────────┐
│     CNN      │  Extract local motifs and k-mer features
│   Layers     │  Filters: [64, 128], Kernels: [3, 5]
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   BiLSTM     │  Model long-range dependencies
│   Layers     │  Units: 128, Layers: 2
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Length-CRF   │  Structured prediction with constraints
│    Layer     │  - Standard CRF potentials
│              │  - Length regularization: Ω(y)
│              │  - PWM priors for ACC
└──────┬───────┘
       │
       ▼
Output Labels: "ADAPTER─UMI─ACC─BARCODE─INSERT─ADAPTER"
```

## Length-Constrained CRF

```
Standard CRF Loss:  L_CRF = -log P(y|x)

Length Penalty:     Ω = Σ [(L_min - L_i)₊² + (L_i - L_max)₊²]
                        i

Total Loss:         L = L_CRF + λ·Ω

where:
  - L_i = length of segment i
  - L_min, L_max = allowed length range
  - λ = constraint weight (ramped over epochs)
  - (·)₊ = max(0, ·)
```

## Ensemble Prediction

```
Bayesian Model Averaging:

P(y|x) = Σ P(y|x,M_i) · P(M_i|D)
         i=1

where:
  - M_i = model i
  - P(M_i|D) = posterior model probability
  - N = number of models in ensemble

Model weights based on validation performance.
```

## File Flow

```
Input Files:
  - config.yaml          → Configuration
  - acc_pwm.txt          → ACC position weight matrix  
  - acc_priors.tsv       → ACC sequence frequencies
  - barcodes.txt         → Valid barcode sequences
  - reads.fastq          → Raw sequencing data

Output Files:
  - annotations.json     → Structured annotations
  - annotations.tsv      → Tabular format
  - annotations.gff      → Genomic format
  - visualizations.pdf   → Colored sequence plots
  - model_weights.h5     → Trained model
  - training_history.csv → Training metrics
```
