# Tempest Documentation Index

Complete guide to all Tempest documentation and code.

## 🎯 Start Here

**New to Tempest?** → [QUICK_REFERENCE.md](QUICK_REFERENCE.md)  
**Want to train a model?** → [TRAINING_GUIDE.md](TRAINING_GUIDE.md)  
**Need PWM help?** → [PWM_GUIDE.md](PWM_GUIDE.md)

## 📚 Documentation by Purpose

### Getting Started (Pick One)

| Document | Purpose | Read Time |
|----------|---------|-----------|
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | 30-second commands and API | 2 min |
| [GETTING_STARTED.md](GETTING_STARTED.md) | Step-by-step tutorial | 15 min |

### Training & Usage

| Document | Purpose | Read Time |
|----------|---------|-----------|
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | **Complete training guide** | 30 min |
| [PWM_GUIDE.md](PWM_GUIDE.md) | **PWM theory and usage** | 25 min |
| [train_config.yaml](train_config.yaml) | Configuration example | 5 min |

### Technical Reference

| Document | Purpose | Read Time |
|----------|---------|-----------|
| [TEMPEST_ARCHITECTURE.md](TEMPEST_ARCHITECTURE.md) | Design decisions | 15 min |
| [WORKFLOW_DIAGRAM.md](WORKFLOW_DIAGRAM.md) | Visual pipeline | 10 min |

### Development & Planning

| Document | Purpose | Read Time |
|----------|---------|-----------|
| [IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md) | Progress tracker | 5 min |
| [README.md](README.md) | Project overview | 10 min |

## Main Scripts

```
main.py              270 lines   Training pipeline
test_simulator.py    120 lines   Testing & validation
examples.py          150 lines   Configuration examples
```

### Configuration

```
train_config.yaml    ~100 lines  Training configuration
example_config.yaml  ~150 lines  Complete config example
acc_pwm.txt           26 lines   ACC PWM data
```

### Core Package

```
tempest/
├── core/
│   ├── pwm.py          430 lines   PWM scoring & generation
│   └── models.py       130 lines   Model architectures
├── data/
│   └── simulator.py    420 lines   Data simulation
└── utils/
    ├── config.py       350 lines   Configuration system
    └── io.py           280 lines   File I/O utilities
```

## Reading Paths

### Path 1: Quick Start (30 minutes)
1. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Get commands
2. Run `python test_simulator.py` - Verify setup
3. Run `python main.py --config train_config.yaml` - Train
4. Check [TRAINING_GUIDE.md](TRAINING_GUIDE.md) if issues

### Path 2: Deep Dive (2 hours)
1. [README.md](README.md) - Project overview
2. [TEMPEST_ARCHITECTURE.md](TEMPEST_ARCHITECTURE.md) - Design
3. [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - How to train
4. [PWM_GUIDE.md](PWM_GUIDE.md) - PWM theory
5. [LC_CRF.pdf](LC_CRF.pdf) - Length constrained semi-Markov CRF

### Path 3: Just the Code (1 hour)
1. [train_config.yaml](train_config.yaml) - See configuration
2. [main.py](main.py) - Understand pipeline
3. [tempest/data/simulator.py](tempest/data/simulator.py) - Data generation
4. [tempest/core/pwm.py](tempest/core/pwm.py) - PWM implementation

### Path 4: Troubleshooting
1. [TRAINING_GUIDE.md](TRAINING_GUIDE.md) → Troubleshooting section
2. [PWM_GUIDE.md](PWM_GUIDE.md) → Troubleshooting

## Documentation by Topic

### Configuration
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) → Configuration section
- [train_config.yaml](train_config.yaml) - Working example
- [tempest/utils/config.py](tempest/utils/config.py) - Implementation

### PWM (Position Weight Matrices)
- [PWM_GUIDE.md](PWM_GUIDE.md) - **Complete guide**
- [tempest/core/pwm.py](tempest/core/pwm.py) - Implementation
- [acc_pwm.txt](acc_pwm.txt) - Example data

### Data Simulation
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) → Simulation section
- [tempest/data/simulator.py](tempest/data/simulator.py) - Implementation
- [test_simulator.py](test_simulator.py) - Examples

### Model Architecture
- [TEMPEST_ARCHITECTURE.md](TEMPEST_ARCHITECTURE.md) - Design
- [WORKFLOW_DIAGRAM.md](WORKFLOW_DIAGRAM.md) - Visual diagrams
- [tempest/core/models.py](tempest/core/models.py) - Implementation

### Training
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - **Primary resource**
- [main.py](main.py) - Complete pipeline
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Commands

## Finding Information

### Common Questions

**Q: How do I start?**  
→ [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

**Q: How do I configure my experiment?**  
→ [TRAINING_GUIDE.md](TRAINING_GUIDE.md) → Configuration section

**Q: What is a PWM?**  
→ [PWM_GUIDE.md](PWM_GUIDE.md)

**Q: How do I customize the sequence structure?**  
→ [TRAINING_GUIDE.md](TRAINING_GUIDE.md) → Simulation section

**Q: Training is failing, what do I do?**  
→ [TRAINING_GUIDE.md](TRAINING_GUIDE.md) → Troubleshooting

**Q: What hyperparameters should I use?**  
→ [TRAINING_GUIDE.md](TRAINING_GUIDE.md) → Hyperparameter tuning

**Q: What files do I need?**  
→ [QUICK_REFERENCE.md](QUICK_REFERENCE.md) → Key Files

**Q: What's implemented and what's planned?**  
→ [IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md)

## By Experience Level

### Beginner (Never used Tempest)
1. [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
2. [GETTING_STARTED.md](GETTING_STARTED.md)
3. Run `test_simulator.py`
4. Run `main.py` with example config

### Intermediate (Know basics, want to customize)
1. [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Deep dive
2. [train_config.yaml](train_config.yaml) - Modify
3. [PWM_GUIDE.md](PWM_GUIDE.md) - Understand PWMs
4. Experiment with hyperparameters

### Advanced (Want to extend/modify)
1. [TEMPEST_ARCHITECTURE.md](TEMPEST_ARCHITECTURE.md)
2. Source code in `tempest/`
3. [IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md)

## Quick Commands

```bash
# Test everything
python test_simulator.py

# Train with default config
python main.py --config train_config.yaml

# Get help
python main.py --help

# View training history
cat checkpoints/training_history.csv
```

## Code Descriptions

| File | Description | Lines |
|------|-------------|-------|
| main.py | Training pipeline | 270 |
| tempest/core/pwm.py | PWM scoring & generation | 430 |
| tempest/data/simulator.py | Data simulation | 420 |
| tempest/utils/config.py | Configuration system | 350 |
| tempest/utils/io.py | File I/O | 280 |
| tempest/core/models.py | Model builder | 130 |
| test_simulator.py | Testing | 120 |
| examples.py | Config examples | 150 |

### Configuration

| File | Description |
|------|-------------|
| train_config.yaml | Training configuration |
| example_config.yaml | Full config example |
| acc_pwm.txt | ACC PWM data |
| requirements.txt | Python dependencies |

## Checklist for New Users

- [ ] Read QUICK_REFERENCE.md
- [ ] Run `python test_simulator.py`
- [ ] Run `python main.py --config train_config.yaml`
- [ ] Review TRAINING_GUIDE.md
- [ ] Customize train_config.yaml
- [ ] Train production model
- [ ] Read PWM_GUIDE.md for ACC details

Happy training!
