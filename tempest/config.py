#!/usr/bin/env python3
"""
Unified configuration module for Tempest.

This module consolidates all configuration classes, including the comprehensive
EnsembleConfig and BMAConfig from the inference module, providing a single
source of truth for configuration management.
"""

import yaml
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Union, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    max_seq_len: int = 500
    num_labels: int = 5
    embedding_dim: int = 64
    lstm_units: int = 128
    lstm_layers: int = 2
    dropout: float = 0.2
    use_cnn: bool = False
    cnn_filters: int = 64
    cnn_kernel_size: int = 3
    use_attention: bool = False
    attention_units: int = 64
    use_bilstm: bool = True
    batch_size: int = 32
    
    @classmethod
    def from_dict(cls, config: dict):
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})


@dataclass
class LengthConstraints:
    """Length constraints for segments."""
    min_length: int
    max_length: int
    
    @classmethod
    def from_dict(cls, config: dict):
        return cls(**config)


@dataclass
class PWMConfig:
    """PWM-specific configuration."""
    pwm_file: str
    temperature: float = 1.0
    min_entropy: float = 0.1
    diversity_boost: float = 1.0
    pattern: Optional[str] = None
    use_probabilistic: bool = True
    scoring_method: str = 'log_likelihood'
    min_score: float = -10.0
    score_weight: float = 0.5
    
    @classmethod
    def from_dict(cls, config: dict):
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})


@dataclass
class SimulationConfig:
    """Data simulation configuration."""
    num_sequences: int = 10000
    train_split: float = 0.8
    n_train: Optional[int] = None
    n_val: Optional[int] = None
    n_test: Optional[int] = None
    random_seed: int = 42
    
    sequence_order: Optional[List[str]] = None
    full_read_reverse_complement_prob: float = 0.0
    
    sequences: Optional[Dict[str, str]] = None
    whitelist_files: Optional[Dict[str, str]] = None
    pwm_files: Optional[Dict[str, str]] = None
    pwm: Optional[PWMConfig] = None
    
    segment_generation: Optional[Dict[str, Any]] = None
    sequence_lengths: Optional[Dict[str, Dict[str, int]]] = None
    
    transcript: Optional[Dict[str, Any]] = None
    polya_tail: Optional[Dict[str, Any]] = None
    error_injection: Optional[Dict[str, Any]] = None
    complexity: Optional[Dict[str, Any]] = None

    # invalid read generation up front instead of during hybrid training
    invalid_fraction: float = 0.0
    
    @classmethod
    def from_dict(cls, config: dict):
        # Handle nested PWM config
        if 'pwm' in config and isinstance(config['pwm'], dict):
            config['pwm'] = PWMConfig.from_dict(config['pwm'])
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = 'adam'
    early_stopping: Optional[Dict[str, Any]] = None
    use_class_weights: bool = False
    checkpoint: Optional[Dict[str, Any]] = None
    tensorboard: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, config: dict):
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})


@dataclass
class BMAConfig:
    """Enhanced configuration for Bayesian Model Averaging."""
    
    # Enable/disable BMA
    enabled: bool = True
    
    # Prior configuration
    prior_type: str = 'uniform'  # 'uniform', 'informative', 'adaptive'
    prior_weights: Optional[Dict[str, float]] = None
    
    # Approximation method
    approximation: str = 'bic'  # 'bic', 'laplace', 'variational', 'cross_validation'
    
    # Approximation parameters (nested structure from YAML)
    approximation_params: Optional[Dict[str, Dict[str, Any]]] = None
    
    # BIC parameters (legacy, for backward compatibility)
    bic_penalty_factor: float = 1.0
    
    # Laplace approximation parameters (legacy)
    laplace_num_samples: int = 1000
    laplace_damping: float = 0.01
    
    # Variational inference parameters (legacy)
    vi_num_iterations: int = 100
    vi_learning_rate: float = 0.01
    vi_convergence_threshold: float = 1e-4
    
    # Cross-validation parameters (legacy)
    cv_num_folds: int = 5
    cv_stratified: bool = True
    
    # Posterior settings
    temperature: float = 1.0
    compute_posterior_variance: bool = True
    normalize_posteriors: bool = True
    min_posterior_weight: float = 0.01
    
    # Model selection
    selection_criteria: Optional[Dict[str, Any]] = None
    use_model_averaging: bool = True
    evidence_threshold: float = 0.05
    
    @classmethod
    def from_dict(cls, config: dict):
        """Create BMAConfig from dictionary, handling nested structures."""
        # Extract selection criteria if nested
        if 'selection_criteria' in config:
            # Keep as dict for flexibility
            pass
        
        # Handle legacy parameters from approximation_params if present
        if 'approximation_params' in config:
            params = config.get('approximation_params', {})
            # Extract BIC params
            if 'bic' in params and 'penalty_factor' in params['bic']:
                config['bic_penalty_factor'] = params['bic']['penalty_factor']
            # Extract Laplace params
            if 'laplace' in params:
                if 'num_samples' in params['laplace']:
                    config['laplace_num_samples'] = params['laplace']['num_samples']
                if 'damping' in params['laplace']:
                    config['laplace_damping'] = params['laplace']['damping']
            # Extract VI params
            if 'variational' in params:
                if 'num_iterations' in params['variational']:
                    config['vi_num_iterations'] = params['variational']['num_iterations']
                if 'learning_rate' in params['variational']:
                    config['vi_learning_rate'] = params['variational']['learning_rate']
                if 'convergence_threshold' in params['variational']:
                    config['vi_convergence_threshold'] = params['variational']['convergence_threshold']
            # Extract CV params
            if 'cross_validation' in params:
                if 'num_folds' in params['cross_validation']:
                    config['cv_num_folds'] = params['cross_validation']['num_folds']
                if 'stratified' in params['cross_validation']:
                    config['cv_stratified'] = params['cross_validation']['stratified']
        
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})


@dataclass
class EnsembleConfig:
    """Complete ensemble configuration with BMA support."""
    
    # General settings
    enabled: bool = True
    num_models: int = 3
    voting_method: str = 'bayesian_model_averaging'  # or 'weighted_average', 'voting', 'stacking'
    
    # Model specifications (from YAML)
    models: Optional[List[Dict[str, Any]]] = None
    
    # BMA configuration (nested)
    bma_config: Optional[BMAConfig] = None
    
    # Weighted average configuration (from YAML)
    weighted_average_config: Optional[Dict[str, Any]] = None
    
    # Legacy weighted average fields (for backward compatibility)
    weighted_optimization: str = 'fixed'
    fixed_weights: Optional[Dict[str, float]] = None
    
    # Prediction aggregation
    prediction_aggregation: Optional[Dict[str, Any]] = None
    prediction_method: str = 'probability_averaging'  # Legacy field
    confidence_weighting: bool = True  # Legacy field
    apply_temperature_scaling: bool = False  # Legacy field
    
    # Calibration settings
    calibration: Optional[Dict[str, Any]] = None
    calibration_enabled: bool = True  # Legacy field
    calibration_method: str = 'isotonic'  # Legacy field
    use_separate_calibration_set: bool = True  # Legacy field
    calibration_split: float = 0.2  # Legacy field
    
    # Diversity settings
    diversity: Optional[Dict[str, Any]] = None
    enforce_diversity: bool = True  # Legacy field
    diversity_metric: str = 'disagreement'  # Legacy field
    min_diversity_threshold: float = 0.1  # Legacy field
    
    # Model variation settings (for ensemble diversity)
    vary_architecture: bool = True
    vary_initialization: bool = True
    vary_training: bool = False
    
    # Uncertainty settings
    uncertainty: Optional[Dict[str, Any]] = None
    compute_epistemic: bool = True  # Legacy field
    compute_aleatoric: bool = True  # Legacy field
    confidence_intervals: bool = True  # Legacy field
    interval_alpha: float = 0.05  # Legacy field
    
    # Evaluation settings
    evaluation: Optional[Dict[str, Any]] = None
    
    # Output settings
    output_dir: str = './ensemble_results'
    
    @classmethod
    def from_dict(cls, config: dict):
        """Create EnsembleConfig from dictionary, handling nested BMAConfig."""
        # Create a copy to avoid modifying original
        config = config.copy()
        
        # Handle nested BMA config
        if 'bma_config' in config and config['bma_config'] is not None:
            if isinstance(config['bma_config'], dict):
                config['bma_config'] = BMAConfig.from_dict(config['bma_config'])
        
        # Handle legacy fields from nested structures
        if 'weighted_average_config' in config and config['weighted_average_config']:
            wac = config['weighted_average_config']
            if 'optimization' in wac:
                config['weighted_optimization'] = wac['optimization']
            if 'fixed_weights' in wac:
                config['fixed_weights'] = wac['fixed_weights']
        
        if 'prediction_aggregation' in config and config['prediction_aggregation']:
            pa = config['prediction_aggregation']
            if 'method' in pa:
                config['prediction_method'] = pa['method']
            if 'confidence_weighting' in pa:
                config['confidence_weighting'] = pa['confidence_weighting']
            if 'apply_temperature_scaling' in pa:
                config['apply_temperature_scaling'] = pa['apply_temperature_scaling']
        
        if 'calibration' in config and config['calibration']:
            cal = config['calibration']
            if 'enabled' in cal:
                config['calibration_enabled'] = cal['enabled']
            if 'method' in cal:
                config['calibration_method'] = cal['method']
            if 'use_separate_calibration_set' in cal:
                config['use_separate_calibration_set'] = cal['use_separate_calibration_set']
            if 'calibration_split' in cal:
                config['calibration_split'] = cal['calibration_split']
        
        if 'diversity' in config and config['diversity']:
            div = config['diversity']
            if 'enforce_diversity' in div:
                config['enforce_diversity'] = div['enforce_diversity']
            if 'diversity_metric' in div:
                config['diversity_metric'] = div['diversity_metric']
            if 'min_diversity_threshold' in div:
                config['min_diversity_threshold'] = div['min_diversity_threshold']
        
        if 'uncertainty' in config and config['uncertainty']:
            unc = config['uncertainty']
            if 'compute_epistemic' in unc:
                config['compute_epistemic'] = unc['compute_epistemic']
            if 'compute_aleatoric' in unc:
                config['compute_aleatoric'] = unc['compute_aleatoric']
            if 'confidence_intervals' in unc:
                config['confidence_intervals'] = unc['confidence_intervals']
            if 'interval_alpha' in unc:
                config['interval_alpha'] = unc['interval_alpha']
        
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})


@dataclass
class InferenceConfig:
    """Configuration for inference/annotation."""
    input_format: str = 'fastq'
    apply_barcode_correction: bool = True
    barcode_correction_distance: int = 1
    deduplicate: bool = True
    output_format: str = 'json'
    min_confidence: float = 0.8
    
    @classmethod
    def from_dict(cls, config: dict):
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})


@dataclass
class HybridTrainingConfig:
    """Configuration for hybrid robustness training."""
    enabled: bool = False
    
    # Training phases
    warmup_epochs: int = 5
    discriminator_epochs: int = 10
    pseudolabel_epochs: int = 10
    
    # Invalid read generation
    invalid_ratio: float = 0.1
    segment_loss_prob: float = 0.3
    segment_dup_prob: float = 0.3
    truncation_prob: float = 0.2
    chimeric_prob: float = 0.1
    scrambled_prob: float = 0.1
    
    # Loss weights
    invalid_weight_initial: float = 0.1
    invalid_weight_max: float = 0.3
    adversarial_weight: float = 0.1
    
    # Pseudo-labeling
    confidence_threshold: float = 0.9
    confidence_decay: float = 0.95
    pseudo_weight: float = 0.5
    max_pseudo_examples: int = 1000
    
    # Discriminator
    discriminator_lr_factor: float = 0.1
    discriminator_hidden_dim: int = 64
    
    # Validation
    validate_architecture: bool = True
    min_unique_segments: int = 3
    max_segment_repetition: int = 2
    
    # Constraints (from YAML structure)
    constrained_decoding: Optional[Dict[str, Any]] = None
    length_constraints: Optional[Dict[str, Any]] = None
    whitelist_constraints: Optional[Dict[str, Any]] = None
    pwm_constraints: Optional[Dict[str, Any]] = None
    transition_constraints: Optional[Dict[str, Any]] = None
    
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
    hybrid: Optional[HybridTrainingConfig] = None
    
    # Additional top-level configs from YAML
    evaluation: Optional[Dict[str, Any]] = None
    visualization: Optional[Dict[str, Any]] = None
    logging: Optional[Dict[str, Any]] = None
    output: Optional[Dict[str, Any]] = None
    
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
        
        hybrid_config = None
        if 'hybrid' in config:
            hybrid_config = HybridTrainingConfig.from_dict(config['hybrid'])
        
        return cls(
            model=model_config,
            simulation=simulation_config,
            training=training_config,
            ensemble=ensemble_config,
            inference=inference_config,
            pwm=pwm_config,
            hybrid=hybrid_config,
            evaluation=config.get('evaluation'),
            visualization=config.get('visualization'),
            logging=config.get('logging'),
            output=config.get('output')
        )
    
    def to_yaml(self, output_path: str):
        """Save configuration to YAML file."""
        config_dict = self._to_dict()
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def to_json(self, output_path: str):
        """Save configuration to JSON file."""
        config_dict = self._to_dict()
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def _to_dict(self) -> Dict[str, Any]:
        """
        Convert the TempestConfig (and all nested dataclasses) to pure dictionaries.
        Ensures nested dataclasses are flattened recursively.
        """
        from dataclasses import asdict, is_dataclass

        def convert_obj(obj):
            """Recursively convert dataclasses, lists, tuples, and dicts to plain Python types."""
            if is_dataclass(obj):
                return {k: convert_obj(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, dict):
                return {k: convert_obj(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_obj(i) for i in obj]
            else:
                return obj

        return convert_obj(self)


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
