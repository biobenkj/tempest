"""
Configuration management for Tempest.

Provides a clean interface for loading and validating configurations
for simulation, training, and inference.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import yaml
import json
from pathlib import Path


@dataclass
class PWMConfig:
    """Configuration for PWM (Position Weight Matrix) detection."""
    pwm_file: Optional[str] = None
    use_pwm: bool = True
    pwm_threshold: float = 0.7
    
    @classmethod
    def from_dict(cls, config: dict):
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})


@dataclass
class LengthConstraints:
    """Length constraints for specific labels."""
    constraints: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    # Example: {'UMI': (8, 8), 'ACC': (6, 6), 'BARCODE': (16, 16)}
    
    constraint_weight: float = 5.0
    ramp_epochs: int = 5
    
    @classmethod
    def from_dict(cls, config: dict):
        constraints = config.get('constraints', {})
        # Convert list format to tuple if needed
        for k, v in constraints.items():
            if isinstance(v, list):
                constraints[k] = tuple(v)
        return cls(
            constraints=constraints,
            constraint_weight=config.get('constraint_weight', 5.0),
            ramp_epochs=config.get('ramp_epochs', 5)
        )


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    # Embedding
    vocab_size: int = 5  # A, C, G, T, N
    embedding_dim: int = 128
    
    # CNN
    use_cnn: bool = True
    cnn_filters: List[int] = field(default_factory=lambda: [64, 128])
    cnn_kernels: List[int] = field(default_factory=lambda: [3, 5])
    
    # BiLSTM
    use_bilstm: bool = True
    lstm_units: int = 128
    lstm_layers: int = 2
    dropout: float = 0.3
    
    # CRF
    num_labels: int = 10
    use_crf: bool = True
    
    # Length constraints
    length_constraints: Optional[LengthConstraints] = None
    
    # Training
    max_seq_len: int = 512
    batch_size: int = 32
    
    @classmethod
    def from_dict(cls, config: dict):
        # Handle nested length_constraints
        length_config = config.pop('length_constraints', None)
        if length_config:
            length_constraints = LengthConstraints.from_dict(length_config)
        else:
            length_constraints = None
        
        return cls(
            **{k: v for k, v in config.items() if k in cls.__annotations__},
            length_constraints=length_constraints
        )


@dataclass
class SimulationConfig:
    """Configuration for data simulation."""
    # Sequence structure
    sequence_order: List[str] = field(default_factory=list)
    # Example: ['ADAPTER', 'UMI', 'ACC', 'BARCODE', 'INSERT', 'ADAPTER']
    
    # Component sequences
    sequences: Dict[str, str] = field(default_factory=dict)
    # Example: {'ADAPTER': 'AGATCGGAAGAGC', 'INSERT': 'random'}
    
    # ACC priors
    acc_priors_file: Optional[str] = None
    acc_sequences: Optional[List[str]] = None
    acc_frequencies: Optional[List[float]] = None
    
    # Barcodes and UMIs
    barcode_file: Optional[str] = None
    umi_length: int = 8
    
    # Simulation parameters
    num_sequences: int = 10000
    insert_min_length: int = 50
    insert_max_length: int = 200
    error_rate: float = 0.05
    
    # Random seed
    random_seed: Optional[int] = None
    
    @classmethod
    def from_dict(cls, config: dict):
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Optimization
    learning_rate: float = 0.001
    optimizer: str = 'adam'
    epochs: int = 20
    
    # Data
    train_split: float = 0.8
    validation_split: float = 0.1
    
    # Callbacks
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    checkpoint_dir: str = './checkpoints'
    
    # Distributed training
    use_mixed_precision: bool = False
    use_xla: bool = True
    
    @classmethod
    def from_dict(cls, config: dict):
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})


@dataclass
class EnsembleConfig:
    """Configuration for ensemble modeling."""
    method: str = 'bma'  # 'bma' (Bayesian Model Averaging) or 'voting'
    num_models: int = 5
    
    # Model diversity
    vary_architecture: bool = True
    vary_initialization: bool = True
    
    # BMA specific
    prior_type: str = 'uniform'  # 'uniform' or 'performance'
    
    @classmethod
    def from_dict(cls, config: dict):
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})


@dataclass
class InferenceConfig:
    """Configuration for inference/annotation."""
    # Input
    input_format: str = 'fastq'  # 'fastq' or 'fasta'
    
    # Post-processing
    apply_barcode_correction: bool = True
    barcode_correction_distance: int = 1
    deduplicate: bool = True
    
    # Output
    output_format: str = 'json'  # 'json', 'tsv', or 'gff'
    min_confidence: float = 0.8
    
    @classmethod
    def from_dict(cls, config: dict):
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})


@dataclass
class TempestConfig:
    """Master configuration for Tempest."""
    model: ModelConfig
    simulation: Optional[SimulationConfig] = None
    training: Optional[TrainingConfig] = None
    ensemble: Optional[EnsembleConfig] = None
    inference: Optional[InferenceConfig] = None
    pwm: Optional[PWMConfig] = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)
    
    @classmethod
    def from_json(cls, json_path: str):
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config = json.load(f)
        return cls.from_dict(config)
    
    @classmethod
    def from_dict(cls, config: dict):
        """Load configuration from dictionary."""
        model_config = ModelConfig.from_dict(config.get('model', {}))
        
        simulation_config = None
        if 'simulation' in config:
            simulation_config = SimulationConfig.from_dict(config['simulation'])
        
        training_config = None
        if 'training' in config:
            training_config = TrainingConfig.from_dict(config['training'])
        
        ensemble_config = None
        if 'ensemble' in config:
            ensemble_config = EnsembleConfig.from_dict(config['ensemble'])
        
        inference_config = None
        if 'inference' in config:
            inference_config = InferenceConfig.from_dict(config['inference'])
        
        pwm_config = None
        if 'pwm' in config:
            pwm_config = PWMConfig.from_dict(config['pwm'])
        
        return cls(
            model=model_config,
            simulation=simulation_config,
            training=training_config,
            ensemble=ensemble_config,
            inference=inference_config,
            pwm=pwm_config
        )
    
    def to_yaml(self, output_path: str):
        """Save configuration to YAML file."""
        # Convert to dict, handling dataclasses
        config_dict = self._to_dict()
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def to_json(self, output_path: str):
        """Save configuration to JSON file."""
        config_dict = self._to_dict()
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def _to_dict(self):
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        
        def convert_tuples(obj):
            """Recursively convert tuples to lists for YAML serialization."""
            if isinstance(obj, dict):
                return {k: convert_tuples(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_tuples(item) for item in obj]
            else:
                return obj
        
        return convert_tuples(asdict(self))


def load_config(config_path: str) -> TempestConfig:
    """
    Load configuration from file (YAML or JSON).
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        TempestConfig instance
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if path.suffix in ['.yaml', '.yml']:
        return TempestConfig.from_yaml(config_path)
    elif path.suffix == '.json':
        return TempestConfig.from_json(config_path)
    else:
        raise ValueError(f"Unsupported configuration format: {path.suffix}")
