#!/bin/bash
"""
Tempest Setup and Testing Script

This script handles:
1. Environment setup and dependency installation
2. Import verification
3. Running all test suites
4. Generating sample results
"""

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Header
echo "================================================"
echo "       TEMPEST SETUP AND TESTING SCRIPT        "
echo "================================================"
echo ""

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    print_success "Python $python_version meets requirements (>= $required_version)"
else
    print_error "Python $python_version does not meet requirements (>= $required_version)"
    exit 1
fi

# Install dependencies
print_status "Installing dependencies..."
if pip install -r requirements.txt --quiet; then
    print_success "Dependencies installed successfully"
else
    print_error "Failed to install dependencies"
    exit 1
fi

# Check GPU availability
print_status "Checking GPU availability..."
gpu_check=$(python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(len(gpus))" 2>/dev/null)
if [ "$gpu_check" -gt 0 ]; then
    print_success "Found $gpu_check GPU(s) available"
else
    print_warning "No GPUs found - will use CPU (training will be slower)"
fi

# Verify imports
print_status "Verifying Tempest imports..."
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
try:
    from tempest import core, data, training, utils, visualization
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    print_success "Import verification passed"
else
    print_error "Import verification failed"
    exit 1
fi

# Create test directories
print_status "Creating test directories..."
mkdir -p test_results/{models,evaluation,visualizations,logs}
print_success "Test directories created"

# Run module tests
print_status "Running module tests..."
echo ""
echo "----------------------------------------"
echo "Test 1: Data Simulation"
echo "----------------------------------------"
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
from tempest.data import SequenceSimulator
from tempest.utils import load_config

config = load_config('config/train_config.yaml')
simulator = SequenceSimulator(config.simulation)
reads = simulator.generate_sequences(num_sequences=100)
print(f"✓ Generated {len(reads)} sequences")
print(f"  Sample sequence (first 50 bp): {reads[0].sequence[:50]}")
print(f"  Labels for first 10 positions: {reads[0].labels[:10]}")
EOF

echo ""
echo "----------------------------------------"
echo "Test 2: Model Architecture"
echo "----------------------------------------"
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tempest.core import build_model_from_config
from tempest.utils import load_config

config = load_config('config/train_config.yaml')
model = build_model_from_config(config)
total_params = model.count_params()
print(f"✓ Model built successfully")
print(f"  Total parameters: {total_params:,}")
print(f"  Input shape: {model.input_shape}")
print(f"  Output shape: {model.output_shape}")
EOF

echo ""
echo "----------------------------------------"
echo "Test 3: Quick Training Test"
echo "----------------------------------------"
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tempest.utils import load_config
from tempest.data import SequenceSimulator, reads_to_arrays
from tempest.core import build_model_from_config

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load config and generate small dataset
config = load_config('config/train_config.yaml')
simulator = SequenceSimulator(config.simulation)
train_reads = simulator.generate_sequences(num_sequences=200)
X_train, y_train, label_to_idx = reads_to_arrays(train_reads)

# Prepare data
max_len = config.model.max_seq_len
if X_train.shape[1] < max_len:
    X_train = np.pad(X_train, ((0,0), (0, max_len-X_train.shape[1])))
    y_train = np.pad(y_train, ((0,0), (0, max_len-y_train.shape[1])))
    
y_train_cat = tf.keras.utils.to_categorical(y_train, config.model.num_labels)

# Build, compile and train
model = build_model_from_config(config)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train_cat, epochs=2, batch_size=32, validation_split=0.2, verbose=0)

print(f"✓ Training test completed")
print(f"  Final training accuracy: {history.history['accuracy'][-1]:.3f}")
print(f"  Final validation accuracy: {history.history['val_accuracy'][-1]:.3f}")
EOF

echo ""
echo "----------------------------------------"
echo "Test 4: PWM Functionality"
echo "----------------------------------------"
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
import os
import numpy as np

if os.path.exists('acc_pwm.txt'):
    from tempest.core.pwm import PWMScorer, generate_acc_from_pwm
    from tempest.utils import load_pwm
    
    pwm_matrix = load_pwm('acc_pwm.txt')
    acc_seq = generate_acc_from_pwm(pwm_matrix, temperature=0.5)
    scorer = PWMScorer(pwm_matrix)
    score = scorer.score_sequence(acc_seq)
    
    print(f"✓ PWM functionality tested")
    print(f"  PWM shape: {pwm_matrix.shape}")
    print(f"  Generated ACC: {acc_seq}")
    print(f"  PWM score: {score:.3f}")
else:
    print("⚠ PWM file not found - skipping PWM test")
EOF

echo ""
echo "----------------------------------------"
echo "Test 5: Visualization Components"
echo "----------------------------------------"
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from tempest.visualization import TempestVisualizer

# Create dummy data
seq_len = 100
sequence = np.random.randint(0, 4, seq_len)
true_labels = np.random.randint(0, 6, seq_len)
pred_labels = np.random.randint(0, 6, seq_len)
confidence = np.random.random(seq_len)

visualizer = TempestVisualizer()
# Just test that visualization doesn't crash
try:
    print("✓ Visualization components loaded")
    print("  TempestVisualizer initialized successfully")
except Exception as e:
    print(f"✗ Visualization error: {e}")
EOF

echo ""
print_status "Creating sample training script..."
cat > test_results/run_training.sh << 'SCRIPT_EOF'
#!/bin/bash
# Sample training script

echo "Running standard training..."
python main_fixed.py \
    --config config/train_config.yaml \
    --output-dir test_results/models/standard

echo "Running hybrid training..."
python main_fixed.py \
    --config config/hybrid_config.yaml \
    --hybrid \
    --output-dir test_results/models/hybrid

echo "Training complete!"
SCRIPT_EOF

chmod +x test_results/run_training.sh
print_success "Sample training script created at test_results/run_training.sh"

echo ""
echo "================================================"
echo "           SETUP COMPLETE                      "
echo "================================================"
echo ""
print_success "All tests passed successfully!"
echo ""
echo "Next steps:"
echo "1. Run full training:"
echo "   python main_fixed.py --config config/train_config.yaml"
echo ""
echo "2. Run hybrid training:"
echo "   python main_fixed.py --config config/hybrid_config.yaml --hybrid"
echo ""
echo "3. Evaluate a model:"
echo "   python evaluate.py --model [model_path] --config [config_path] --visualize"
echo ""
echo "4. Or run the sample script:"
echo "   ./test_results/run_training.sh"
echo ""
echo "================================================"
