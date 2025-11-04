# Tempest Implementation Checklist

Track progress through the phased implementation of Tempest.

## Phase 1: Core Infrastructure (COMPLETE)

- [x] Project structure created
- [x] Configuration system (`tempest/utils/config.py`)
  - [x] `ModelConfig` - Model architecture parameters
  - [x] `SimulationConfig` - Data generation parameters
  - [x] `TrainingConfig` - Training hyperparameters
  - [x] `EnsembleConfig` - BMA parameters
  - [x] `InferenceConfig` - Annotation parameters
  - [x] `PWMConfig` - PWM detection settings
  - [x] `LengthConstraints` - Segment length constraints
  - [x] YAML/JSON load/save
  - [x] Type hints throughout
- [x] I/O utilities (`tempest/utils/io.py`)
  - [x] PWM loading/saving
  - [x] ACC priors loading
  - [x] Barcode loading
  - [x] FASTQ/FASTA readers
  - [x] Annotation export (JSON/TSV/GFF)
  - [x] Base encoding utilities
- [x] Example configuration (`example_config.yaml`)
- [x] Requirements file (`requirements.txt`)
- [x] Documentation
  - [x] Architecture design doc
  - [x] Workflow diagram
  - [x] README
  - [x] Usage examples
- [x] Testing
  - [x] Configuration system tested

## Phase 2: Model Architecture (NEXT)

- [ ] PWM module (`tempest/core/pwm.py`)
  - [ ] PWM scoring function
  - [ ] ACC detection using PWM
  - [ ] Threshold-based detection
  - [ ] Integration with CRF
  
- [ ] Length-Constrained CRF (`tempest/core/length_crf.py`)
  - [ ] Clean version from uploaded file
  - [ ] `LengthConstrainedCRF` layer class
  - [ ] `ModelWithLengthConstrainedCRF` wrapper
  - [ ] Vectorized length penalty computation
  - [ ] Constraint weight ramping
  - [ ] XLA compatibility
  
- [ ] Model builder (`tempest/core/models.py`)
  - [ ] Embedding layer
  - [ ] CNN layers (configurable filters/kernels)
  - [ ] BiLSTM layers (configurable units/layers)
  - [ ] CRF output layer
  - [ ] Full CNN-BiLSTM-CRF model
  - [ ] Model from config
  - [ ] Model serialization/deserialization

- [ ] Testing
  - [ ] PWM scoring test
  - [ ] Length-CRF test
  - [ ] Full model forward pass test

## Phase 3: Data Pipeline

- [ ] Sequence simulator (`tempest/data/simulator.py`)
  - [ ] Random sequence generation
  - [ ] ACC sequence sampling from priors
  - [ ] Barcode sampling
  - [ ] UMI generation
  - [ ] Sequence structure assembly
  - [ ] Error injection (substitutions, insertions, deletions)
  - [ ] Label generation
  - [ ] Batch generation
  
- [ ] Preprocessor (`tempest/data/preprocessor.py`)
  - [ ] FASTQ/FASTA reading
  - [ ] Sequence encoding
  - [ ] Quality filtering
  - [ ] Length filtering
  - [ ] Orientation detection
  
- [ ] Data generators (`tempest/data/generators.py`)
  - [ ] TensorFlow Dataset creation
  - [ ] Batching and padding
  - [ ] On-the-fly augmentation
  - [ ] Memory-efficient streaming

- [ ] Testing
  - [ ] Simulate 1000 sequences
  - [ ] Verify label consistency
  - [ ] Test error injection

## Phase 4: Training

- [ ] Single model trainer (`tempest/training/trainer.py`)
  - [ ] `ModelTrainer` class
  - [ ] Training loop
  - [ ] Validation loop
  - [ ] Callbacks (early stopping, reduce LR, checkpointing)
  - [ ] Metrics tracking
  - [ ] Logging
  - [ ] Model saving
  
- [ ] Ensemble trainer (`tempest/training/ensemble.py`)
  - [ ] `EnsembleTrainer` class
  - [ ] Multiple model training with variation
    - [ ] Architecture variation
    - [ ] Initialization variation
  - [ ] Model weight computation (BMA)
    - [ ] Uniform prior
    - [ ] Performance-based prior
  - [ ] Ensemble prediction
  - [ ] Ensemble serialization
  
- [ ] Training utilities
  - [ ] Custom metrics (per-label accuracy)
  - [ ] Learning rate schedules
  - [ ] Mixed precision training
  - [ ] Distributed training support

- [ ] Testing
  - [ ] Train single model on simulated data
  - [ ] Train ensemble (5 models)
  - [ ] Verify BMA weights

## Phase 5: Inference & Visualization

- [ ] Annotation pipeline (`tempest/inference/annotator.py`)
  - [ ] `SequenceAnnotator` class
  - [ ] FASTQ batch processing
  - [ ] Model prediction
  - [ ] Ensemble prediction (BMA)
  - [ ] Confidence scores
  - [ ] Annotation formatting
  
- [ ] Post-processing (`tempest/inference/postprocess.py`)
  - [ ] Barcode error correction
    - [ ] Hamming distance correction
    - [ ] Known barcode matching
  - [ ] UMI deduplication
  - [ ] Architecture classification
  - [ ] Quality filtering
  
- [ ] Visualization (`tempest/visualization/plots.py`)
  - [ ] **Retain visualization from Tranquillyzer**
  - [ ] Colored sequence plots
  - [ ] Multi-page PDF generation
  - [ ] Label color mapping
  - [ ] Read metrics visualization
  - [ ] Training history plots
  
- [ ] Export utilities
  - [ ] JSON export (full annotations)
  - [ ] TSV export (tabular format)
  - [ ] GFF3 export (genomic format)
  - [ ] Summary statistics

- [ ] Testing
  - [ ] Annotate 100 real reads
  - [ ] Verify barcode correction
  - [ ] Generate visualizations
  - [ ] Export in all formats

## Phase 6: CLI & Documentation

- [ ] Command-line interface
  - [ ] `tempest simulate` - Data simulation
  - [ ] `tempest train` - Model training
  - [ ] `tempest ensemble` - Ensemble training
  - [ ] `tempest annotate` - Sequence annotation
  - [ ] `tempest visualize` - Generate plots
  - [ ] Argument parsing
  - [ ] Progress bars
  
- [ ] Documentation
  - [ ] API documentation (sphinx)
  - [ ] Tutorial notebooks
  - [ ] Example workflows
  - [ ] Best practices guide
  - [ ] Troubleshooting guide
  
- [ ] Packaging
  - [ ] setup.py / pyproject.toml
  - [ ] PyPI package
  - [ ] Conda package
  - [ ] Docker image

## Success Criteria

By the end of all phases, users should be able to:

1. Configure experiments via YAML files
2. Simulate realistic training data with ACC priors
3. Train models with length constraints and PWM priors
4. Build ensemble models using BMA
5. Annotate real FASTQ files
6. Visualize annotations with colored sequences
7. Export annotations in multiple formats
8. Run entire pipeline via command line

