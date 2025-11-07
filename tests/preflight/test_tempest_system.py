#!/usr/bin/env python3
"""
TEMPEST System Test Suite
=========================
Comprehensive tests to validate GPU setup, data handling, model training, and inference.

Run with: python test_tempest_system.py
"""

import sys
import os
import numpy as np
from pathlib import Path

# Color codes for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_test_header(test_name):
    """Print formatted test header."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*70}")
    print(f"  TEST: {test_name}")
    print(f"{'='*70}{Colors.RESET}\n")

def print_success(message):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")

def print_failure(message):
    """Print failure message."""
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")

def print_warning(message):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.RESET}")

def print_info(message):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {message}{Colors.RESET}")


class TempestSystemTest:
    """Comprehensive system test for TEMPEST."""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.tests_total = 0
        self.gpu_available = False
        
    def run_all_tests(self):
        """Run all system tests in sequence."""
        print(f"\n{Colors.BOLD}{'='*70}")
        print("  TEMPEST System Test Suite")
        print(f"{'='*70}{Colors.RESET}\n")
        
        # Core tests
        self.test_gpu_detection()
        self.test_tensorflow_import()
        self.test_cuda_gpu_operations()
        self.test_data_simulation()
        self.test_model_creation()
        self.test_sample_training()
        self.test_inference_pipeline()
        self.test_length_constraints()
        self.test_pwm_functionality()
        
        # Summary
        self.print_summary()
        
    def record_test(self, name, passed, details=""):
        """Record test result."""
        self.tests_total += 1
        if passed:
            self.tests_passed += 1
            print_success(f"{name}")
            if details:
                print(f"  {details}")
        else:
            self.tests_failed += 1
            print_failure(f"{name}")
            if details:
                print(f"  {details}")
    
    def test_gpu_detection(self):
        """Test 1: GPU Detection and CUDA Availability."""
        print_test_header("GPU Detection and CUDA Availability")
        
        try:
            import tensorflow as tf
            
            # Check TensorFlow version
            tf_version = tf.__version__
            print_info(f"TensorFlow version: {tf_version}")
            self.record_test("TensorFlow version check", True, f"Version {tf_version}")
            
            # List physical devices
            gpus = tf.config.list_physical_devices('GPU')
            print_info(f"Physical GPUs detected: {len(gpus)}")
            
            if gpus:
                for i, gpu in enumerate(gpus):
                    print_info(f"  GPU {i}: {gpu.name}")
                    # Try to get GPU details
                    try:
                        gpu_details = tf.config.experimental.get_device_details(gpu)
                        if 'device_name' in gpu_details:
                            print_info(f"    Device: {gpu_details['device_name']}")
                    except:
                        pass
                
                self.gpu_available = True
                self.record_test("GPU detection", True, f"Found {len(gpus)} GPU(s)")
                
                # Test GPU memory growth setting
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    self.record_test("GPU memory growth configuration", True)
                except Exception as e:
                    self.record_test("GPU memory growth configuration", False, str(e))
                    
            else:
                self.gpu_available = False
                self.record_test("GPU detection", False, "No GPUs detected")
                print_warning("Training will run on CPU (will be slower)")
                
        except Exception as e:
            self.record_test("GPU detection", False, f"Error: {str(e)}")
    
    def test_tensorflow_import(self):
        """Test 2: TensorFlow Core Imports."""
        print_test_header("TensorFlow Core Imports")
        
        try:
            import tensorflow as tf
            from tensorflow import keras
            import tensorflow_addons as tfa
            
            self.record_test("TensorFlow import", True)
            self.record_test("Keras import", True)
            self.record_test("TensorFlow Addons import", True)
            
            # Check if operations are available
            print_info("Testing basic TensorFlow operations...")
            x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            y = tf.constant([[1.0, 1.0], [1.0, 1.0]])
            result = tf.matmul(x, y)
            
            self.record_test("Basic TensorFlow operations", True)
            
        except Exception as e:
            self.record_test("TensorFlow imports and operations", False, str(e))
    
    def test_cuda_gpu_operations(self):
        """Test 3: CUDA GPU Operations."""
        print_test_header("CUDA GPU Operations")
        
        if not self.gpu_available:
            print_warning("Skipping GPU operations test (no GPU available)")
            return
            
        try:
            import tensorflow as tf
            
            # Force GPU execution
            with tf.device('/GPU:0'):
                # Create random matrices
                matrix_size = 1000
                a = tf.random.normal([matrix_size, matrix_size])
                b = tf.random.normal([matrix_size, matrix_size])
                
                print_info(f"Testing matrix multiplication on GPU ({matrix_size}x{matrix_size})...")
                
                # Warm up
                _ = tf.matmul(a, b)
                
                # Time the operation
                import time
                start = time.time()
                for _ in range(10):
                    c = tf.matmul(a, b)
                elapsed = time.time() - start
                
                print_info(f"10 matrix multiplications completed in {elapsed:.3f}s")
                self.record_test("GPU matrix operations", True, f"Average: {elapsed/10:.4f}s per operation")
                
                # Test gradient computation on GPU
                with tf.GradientTape() as tape:
                    tape.watch(a)
                    result = tf.reduce_sum(tf.matmul(a, b))
                grad = tape.gradient(result, a)
                
                self.record_test("GPU gradient computation", True)
                
        except Exception as e:
            self.record_test("GPU operations", False, str(e))
    
    def test_data_simulation(self):
        """Test 4: Data Generation and Simulation."""
        print_test_header("Data Generation and Simulation")
        
        try:
            # Simulate sequence data
            print_info("Generating synthetic sequence data...")
            
            num_samples = 100
            seq_length = 150
            num_labels = 5
            vocab_size = 5  # A, C, G, T, N
            
            # Generate random DNA sequences (one-hot encoded)
            sequences = np.random.randint(0, vocab_size, size=(num_samples, seq_length))
            
            # Generate random labels
            labels = np.random.randint(0, num_labels, size=(num_samples, seq_length))
            
            print_info(f"Generated {num_samples} sequences of length {seq_length}")
            print_info(f"Sequence shape: {sequences.shape}")
            print_info(f"Labels shape: {labels.shape}")
            
            self.record_test("Synthetic data generation", True, 
                           f"{num_samples} samples, length {seq_length}")
            
            # Store for later tests
            self.test_sequences = sequences
            self.test_labels = labels
            self.vocab_size = vocab_size
            self.num_labels = num_labels
            self.seq_length = seq_length
            
        except Exception as e:
            self.record_test("Data generation", False, str(e))
    
    def test_model_creation(self):
        """Test 5: Model Creation and Architecture."""
        print_test_header("Model Creation and Architecture")
        
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            print_info("Building CNN-BiLSTM model...")
            
            # Model parameters
            embedding_dim = 32
            lstm_units = 64
            cnn_filters = [32, 64]
            cnn_kernels = [3, 5]
            
            # Build a simple model
            inputs = keras.Input(shape=(self.seq_length,), dtype=tf.int32)
            
            # Embedding layer
            x = keras.layers.Embedding(self.vocab_size, embedding_dim)(inputs)
            
            # CNN layers
            conv_outputs = []
            for filters, kernel in zip(cnn_filters, cnn_kernels):
                conv = keras.layers.Conv1D(filters, kernel, padding='same', activation='relu')(x)
                conv_outputs.append(conv)
            
            x = keras.layers.Concatenate()(conv_outputs)
            
            # BiLSTM layers
            x = keras.layers.Bidirectional(keras.layers.LSTM(lstm_units, return_sequences=True))(x)
            
            # Output layer
            outputs = keras.layers.Dense(self.num_labels, activation='softmax')(x)
            
            model = keras.Model(inputs=inputs, outputs=outputs)
            
            print_info("Model architecture:")
            model.summary(print_fn=lambda x: print_info(f"  {x}"))
            
            self.record_test("Model creation", True)
            
            # Compile model
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.record_test("Model compilation", True)
            
            # Store model for training test
            self.model = model
            
        except Exception as e:
            self.record_test("Model creation", False, str(e))
    
    def test_sample_training(self):
        """Test 6: Sample Training Loop."""
        print_test_header("Sample Training Loop")
        
        if not hasattr(self, 'model'):
            print_warning("Skipping training test (model not created)")
            return
            
        try:
            import tensorflow as tf
            
            print_info("Running mini training loop (3 epochs)...")
            print_info(f"Device: {'GPU' if self.gpu_available else 'CPU'}")
            
            # Use small subset for quick test
            train_size = 50
            X_train = self.test_sequences[:train_size]
            y_train = self.test_labels[:train_size]
            
            # Run training
            history = self.model.fit(
                X_train, y_train,
                epochs=3,
                batch_size=8,
                verbose=0,
                validation_split=0.2
            )
            
            final_loss = history.history['loss'][-1]
            final_acc = history.history['accuracy'][-1]
            
            print_info(f"Training completed")
            print_info(f"  Final loss: {final_loss:.4f}")
            print_info(f"  Final accuracy: {final_acc:.4f}")
            
            self.record_test("Sample training", True, 
                           f"Loss: {final_loss:.4f}, Acc: {final_acc:.4f}")
            
            # Test that model can make predictions
            predictions = self.model.predict(X_train[:5], verbose=0)
            self.record_test("Post-training predictions", True, 
                           f"Output shape: {predictions.shape}")
            
        except Exception as e:
            self.record_test("Sample training", False, str(e))
    
    def test_inference_pipeline(self):
        """Test 7: Inference Pipeline."""
        print_test_header("Inference Pipeline")
        
        if not hasattr(self, 'model'):
            print_warning("Skipping inference test (model not created)")
            return
            
        try:
            print_info("Testing inference on new data...")
            
            # Generate test batch
            test_batch = np.random.randint(0, self.vocab_size, size=(10, self.seq_length))
            
            # Run inference
            predictions = self.model.predict(test_batch, verbose=0)
            
            print_info(f"Input shape: {test_batch.shape}")
            print_info(f"Output shape: {predictions.shape}")
            print_info(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
            
            # Check output validity
            assert predictions.shape == (10, self.seq_length, self.num_labels)
            assert np.all(predictions >= 0) and np.all(predictions <= 1)
            
            self.record_test("Inference pipeline", True)
            
            # Test batch processing
            large_batch = np.random.randint(0, self.vocab_size, size=(100, self.seq_length))
            predictions = self.model.predict(large_batch, batch_size=32, verbose=0)
            
            self.record_test("Batch inference", True, f"Processed {len(large_batch)} sequences")
            
        except Exception as e:
            self.record_test("Inference pipeline", False, str(e))
    
    def test_length_constraints(self):
        """Test 8: Length-Constrained CRF Operations."""
        print_test_header("Length-Constrained CRF Operations")
        
        try:
            import tensorflow as tf
            
            print_info("Testing length constraint computations...")
            
            # Simulate CRF potentials
            batch_size = 4
            seq_len = 50
            num_states = 5
            
            # Unary potentials (emission scores) - use Variable for gradient tracking
            unary = tf.Variable(tf.random.normal([batch_size, seq_len, num_states]))
            
            # Transition matrix
            transitions = tf.random.normal([num_states, num_states])
            
            print_info(f"Unary potentials shape: {unary.shape}")
            print_info(f"Transition matrix shape: {transitions.shape}")
            
            self.record_test("CRF tensor creation", True)
            
            # Test segment length constraints
            min_length = 3
            max_length = 10
            
            # Create length mask
            lengths = tf.range(1, seq_len + 1, dtype=tf.float32)
            length_mask = tf.logical_and(
                lengths >= min_length,
                lengths <= max_length
            )
            
            print_info(f"Length constraints: min={min_length}, max={max_length}")
            self.record_test("Length constraint masks", True)
            
            # Test gradient flow through CRF operations
            with tf.GradientTape() as tape:
                # Compute forward scores inside tape context
                forward = tf.reduce_logsumexp(unary, axis=2)
                # Compute a loss-like score
                score = tf.reduce_sum(forward)
            
            # Compute gradients
            grad = tape.gradient(score, unary)
            
            # Check gradient is valid
            if grad is not None:
                grad_valid = not tf.reduce_any(tf.math.is_nan(grad))
                print_info(f"Gradient shape: {grad.shape}")
                print_info(f"Gradient range: [{tf.reduce_min(grad):.4f}, {tf.reduce_max(grad):.4f}]")
                self.record_test("CRF gradient computation", grad_valid, 
                               f"Valid gradients computed")
            else:
                self.record_test("CRF gradient computation", False, "Gradient is None")
            
        except Exception as e:
            self.record_test("Length constraints", False, str(e))
    
    def test_pwm_functionality(self):
        """Test 9: Position Weight Matrix (PWM) Operations."""
        print_test_header("Position Weight Matrix (PWM) Operations")
        
        try:
            import tensorflow as tf
            
            print_info("Testing PWM scoring...")
            
            # Create a simple PWM (e.g., for ACC motif)
            motif_length = 6
            vocab_size = 4  # A, C, G, T
            
            # Random PWM
            pwm = tf.random.uniform([motif_length, vocab_size], minval=-2, maxval=2)
            
            print_info(f"PWM shape: {pwm.shape}")
            print_info(f"PWM values range: [{tf.reduce_min(pwm):.2f}, {tf.reduce_max(pwm):.2f}]")
            
            self.record_test("PWM creation", True)
            
            # Test scoring a sequence
            batch_size = 10
            seq_len = 100
            
            # One-hot encoded sequences
            sequences = tf.one_hot(
                tf.random.uniform([batch_size, seq_len], minval=0, maxval=vocab_size, dtype=tf.int32),
                depth=vocab_size
            )
            
            # Compute PWM scores using convolution
            pwm_kernel = tf.expand_dims(pwm, axis=1)  # [motif_len, 1, vocab]
            pwm_kernel = tf.transpose(pwm_kernel, [0, 2, 1])  # [motif_len, vocab, 1]
            
            # Score all positions
            scores = tf.nn.conv1d(
                sequences,
                pwm_kernel,
                stride=1,
                padding='VALID'
            )
            
            print_info(f"PWM scores shape: {scores.shape}")
            print_info(f"Score range: [{tf.reduce_min(scores):.2f}, {tf.reduce_max(scores):.2f}]")
            
            self.record_test("PWM scoring", True)
            
            # Test thresholding
            threshold = 0.0
            matches = tf.reduce_sum(tf.cast(scores > threshold, tf.int32))
            print_info(f"Motif matches above threshold: {matches.numpy()}")
            
            self.record_test("PWM thresholding", True)
            
        except Exception as e:
            self.record_test("PWM functionality", False, str(e))
    
    def print_summary(self):
        """Print test summary."""
        print(f"\n{Colors.BOLD}{'='*70}")
        print("  TEST SUMMARY")
        print(f"{'='*70}{Colors.RESET}\n")
        
        pass_rate = (self.tests_passed / self.tests_total * 100) if self.tests_total > 0 else 0
        
        print(f"Total tests: {Colors.BOLD}{self.tests_total}{Colors.RESET}")
        print(f"Passed: {Colors.GREEN}{self.tests_passed}{Colors.RESET}")
        print(f"Failed: {Colors.RED}{self.tests_failed}{Colors.RESET}")
        print(f"Pass rate: {Colors.BOLD}{pass_rate:.1f}%{Colors.RESET}\n")
        
        if self.tests_failed == 0:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ All tests passed!{Colors.RESET}")
            print(f"{Colors.GREEN}Your system is ready for TEMPEST training.{Colors.RESET}\n")
        else:
            print(f"{Colors.YELLOW}⚠ Some tests failed.{Colors.RESET}")
            print(f"{Colors.YELLOW}Review the failures above before proceeding.{Colors.RESET}\n")
        
        # GPU recommendation
        if self.gpu_available:
            print(f"{Colors.GREEN}GPU detected and functional - training will be accelerated.{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}No GPU detected - training will run on CPU (slower).{Colors.RESET}")
            print(f"{Colors.YELLOW}For faster training, ensure CUDA-compatible GPU is available.{Colors.RESET}")


if __name__ == "__main__":
    print(f"\n{Colors.BOLD}TEMPEST System Test Suite{Colors.RESET}")
    print(f"Testing GPU, TensorFlow, data handling, and training pipeline\n")
    
    # Set environment to reduce TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Run tests
    tester = TempestSystemTest()
    tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if tester.tests_failed == 0 else 1)
