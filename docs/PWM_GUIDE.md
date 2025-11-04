# Position Weight Matrices (PWMs) in Tempest

Comprehensive guide to PWMs, ACC detection, and sequence generation.

## Table of Contents
1. [What is a PWM?](#what-is-a-pwm)
2. [PWM in Tempest](#pwm-in-tempest)
3. [ACC Sequences](#acc-sequences)
4. [Using PWMs](#using-pwms)
5. [Creating PWMs](#creating-pwms)
6. [Advanced Topics](#advanced-topics)

## What is a PWM?

A **Position Weight Matrix** (PWM) is a mathematical model that captures position-specific base preferences in biological sequences.

### Mathematical Definition

For a sequence motif of length L, a PWM is a matrix M of size L × 4:

```
M[i,j] = probability of base j at position i

where j ∈ {A, C, G, T}
```

### Example PWM

For the ACC motif (6 bases):

```
Position    A       C       G       T
   1      0.944   0.026   0.007   0.022   ← ~94% A
   2      0.011   0.834   0.007   0.148   ← ~83% C
   3      0.009   0.940   0.031   0.021   ← ~94% C
   4      0.015   0.316   0.657   0.013   ← ~66% G
   5      0.033   0.144   0.812   0.012   ← ~81% G
   6      0.165   0.160   0.660   0.015   ← ~66% G
```

This represents the **IUPAC pattern: ACCSSV**
- Position 1: A (strong preference)
- Position 2: C (strong preference)
- Position 3: C (strong preference)
- Positions 4-5: S = G or C (G preferred)
- Position 6: V = A, C, or G (G preferred)

### Visualization

```
     A  C  G  T
Pos1 █████████████████████████████████ 94%
     ██ 3%
     
Pos2 █ 1%
     █████████████████████████████ 83%
     █ 1%
     ███████████ 15%
     
Pos3 █ 1%
     █████████████████████████████████ 94%
     ██ 3%
     ██ 2%
```

## PWM in Tempest

### Why PWMs Matter

1. **Realistic simulation**: Generate ACC sequences that match real data
2. **Detection**: Find ACC sequences in reads with high accuracy
3. **Validation**: Verify ACC sequences match expected patterns
4. **Priors**: Inform the model about valid ACC sequences

### PWM File Format

Tempest uses tab-separated format:

```
pos	base	prob
1	A	0.944634482890082
1	C	0.026079325376165
1	G	0.007367199880516
1	T	0.021918991853238
2	A	0.010948997911672
2	C	0.833559568052585
...
```

**Requirements:**
- Tab-separated values
- Three columns: `pos`, `base`, `prob`
- Position starts at 1
- Probabilities for each position sum to 1.0

### Loading a PWM

```python
from tempest.utils import load_pwm
from tempest.core import PWMScorer

# Load PWM from file
pwm = load_pwm('acc_pwm.txt')
print(f"PWM shape: {pwm.shape}")  # (6, 4)

# Create scorer
scorer = PWMScorer(pwm, threshold=0.7)
```

## ACC Sequences

### What is ACC?

ACC is an **adapter core consensus** sequence found in certain sequencing protocols. It's a short (typically 6bp) sequence with specific base preferences at each position.

### ACC Characteristics

1. **Fixed length**: Usually 6 bases
2. **Degenerate**: Not all positions are fixed
3. **IUPAC pattern**: ACCSSV
   - ACC: Fixed first 3 bases
   - SS: G or C (positions 4-5)
   - V: A, C, or G (position 6)
4. **Variable**: Different variants exist with different frequencies

### Common ACC Sequences

Based on the PWM:

```
ACCGGG  (most common)      Score: 1.000
ACCGGC  (common)           Score: 0.946
ACCGCG  (common)           Score: 0.934
ACCCGG  (less common)      Score: 0.892
ACCCGC  (less common)      Score: 0.841
ACCGGA  (less common)      Score: 0.798
```

### Invalid ACC Sequences

```
TTTTTT  (no A's)           Score: 0.186
AAAAAA  (wrong positions)  Score: 0.338
GCCGGG  (wrong pos 1)      Score: 0.421
ACGGGG  (wrong pos 2)      Score: 0.512
```

## Using PWMs

### Scoring Sequences

```python
from tempest.core import PWMScorer
from tempest.utils import load_pwm

# Setup
pwm = load_pwm('acc_pwm.txt')
scorer = PWMScorer(pwm, threshold=0.7)

# Score a sequence
score = scorer.score_sequence('ACCGGG')
print(f"Score: {score:.3f}")  # 1.000

# Check if above threshold
if score >= scorer.threshold:
    print("Valid ACC sequence!")
```

### Finding ACC in Reads

```python
# Find best match in a long read
read = "AGATCGGAAGAGCGTAAGTGACCGGGCACGTACGTACG..."

# Search for ACC
result = scorer.detect_motif(read)

if result:
    start, end, score, match = result
    print(f"Found ACC at position {start}-{end}")
    print(f"Sequence: {match}")
    print(f"Score: {score:.3f}")
else:
    print("No ACC found above threshold")
```

### Generating ACC Sequences

```python
from tempest.core import generate_acc_from_pwm

# Generate 10 ACC sequences
acc_sequences = generate_acc_from_pwm(pwm, n=10, random_state=42)

for seq in acc_sequences:
    score = scorer.score_sequence(seq)
    print(f"{seq}  (score={score:.3f})")
```

### Validating Against IUPAC

```python
# Check if sequence matches IUPAC pattern
is_valid = scorer.validate_against_iupac('ACCGGG', 'ACCSSV')
print(f"Valid: {is_valid}")  # True

is_valid = scorer.validate_against_iupac('ACCTTT', 'ACCSSV')
print(f"Valid: {is_valid}")  # False (T not allowed in position 4-5)
```

## Creating PWMs

### From Real Data

If you have real ACC sequences with known positions:

```python
from tempest.core import compute_pwm_from_sequences

# Collect ACC sequences from your data
acc_sequences = [
    'ACCGGG',
    'ACCGGC',
    'ACCGCG',
    'ACCGGG',
    'ACCCGG',
    # ... more sequences
]

# Compute PWM
pwm = compute_pwm_from_sequences(acc_sequences)

# Save for future use
from tempest.utils import save_pwm
save_pwm(pwm, 'my_acc_pwm.txt')
```

### Manual Construction

For known IUPAC patterns:

```python
import numpy as np

# IUPAC: ACCSSV
# Position 1: A only
# Position 2: C only
# Position 3: C only
# Position 4: S (G or C) - say 70% G, 30% C
# Position 5: S (G or C) - say 80% G, 20% C
# Position 6: V (A,C,G) - say 10%, 15%, 75%

pwm = np.array([
    [0.95, 0.02, 0.02, 0.01],  # Position 1: ~95% A
    [0.01, 0.95, 0.02, 0.02],  # Position 2: ~95% C
    [0.01, 0.95, 0.02, 0.02],  # Position 3: ~95% C
    [0.01, 0.29, 0.69, 0.01],  # Position 4: S (~70% G)
    [0.01, 0.19, 0.79, 0.01],  # Position 5: S (~80% G)
    [0.10, 0.15, 0.74, 0.01],  # Position 6: V (mixed)
])

save_pwm(pwm, 'custom_acc_pwm.txt')
```

### From Multiple Sequence Alignment

```python
from Bio import AlignIO

# Load alignment
alignment = AlignIO.read('acc_alignment.fasta', 'fasta')

# Extract sequences
sequences = [str(record.seq) for record in alignment]

# Compute PWM
pwm = compute_pwm_from_sequences(sequences)
```

## Advanced Topics

### Log-Odds Scoring

Internally, PWM scoring uses log-odds ratios:

```
score(seq) = Σ log(P(base_i | position_i) / P(base_i | background))
             i=1..L

where:
  P(base_i | position_i) = PWM probability
  P(base_i | background) = 0.25 (uniform)
```

This gives higher scores to sequences that match the PWM better than random.

### Score Normalization

Scores are normalized to [0, 1]:

```
normalized_score = (score - min_score) / (max_score - min_score)

where:
  max_score = best possible score (take max at each position)
  min_score = worst possible score (take min at each position)
```

### Threshold Selection

Choosing the right threshold:

```python
from tempest.core import evaluate_pwm_performance

# Test different thresholds
true_sequences = load_true_acc_sequences()
false_sequences = load_non_acc_sequences()

for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    scorer = PWMScorer(pwm, threshold=threshold)
    metrics = evaluate_pwm_performance(scorer, true_sequences, false_sequences)
    
    print(f"\nThreshold: {threshold:.1f}")
    print(f"  Sensitivity: {metrics['true_positive_rate']:.3f}")
    print(f"  Specificity: {metrics['specificity']:.3f}")
    print(f"  F1 Score: {metrics['f1_score']:.3f}")
```

**Guidelines:**
- Threshold 0.7: Balanced (recommended)
- Threshold 0.8: Higher specificity, may miss some true ACCs
- Threshold 0.6: Higher sensitivity, more false positives

### PWM Information Content

Measure how informative each position is:

```python
import numpy as np

def information_content(pwm):
    """Compute information content per position."""
    ic = np.zeros(pwm.shape[0])
    for i in range(pwm.shape[0]):
        # Compute entropy
        p = pwm[i]
        h = -np.sum(p * np.log2(p + 1e-10))
        # Information = 2 - entropy
        ic[i] = 2 - h
    return ic

ic = information_content(pwm)
print("Information content per position:")
for i, value in enumerate(ic):
    print(f"  Position {i+1}: {value:.2f} bits")
```

**Interpretation:**
- 2 bits: Fully conserved (one base only)
- 0 bits: No information (all bases equally likely)

### Reverse Complement PWMs

For stranded data:

```python
def reverse_complement_pwm(pwm):
    """Create reverse complement PWM."""
    # Reverse positions
    rc_pwm = pwm[::-1]
    # Swap A↔T and C↔G
    rc_pwm = rc_pwm[:, [3, 2, 1, 0]]  # [T, G, C, A]
    return rc_pwm

# Create RC scorer
rc_pwm = reverse_complement_pwm(pwm)
rc_scorer = PWMScorer(rc_pwm)

# Score both strands
fwd_score = scorer.score_sequence(seq)
rev_score = rc_scorer.score_sequence(seq)
best_score = max(fwd_score, rev_score)
```

## Integration with Simulation

### Using PWM in Simulator

```python
from tempest.data import SequenceSimulator
from tempest.utils import load_config

config = load_config('train_config.yaml')

# Simulator automatically uses PWM if provided
simulator = SequenceSimulator(
    config.simulation,
    pwm_file='acc_pwm.txt'
)

# Generate reads with realistic ACC sequences
reads = simulator.generate(num_sequences=1000)
```

### Priority: Priors > PWM > Random

The simulator uses this priority:

1. **ACC priors** (if `acc_priors_file` specified)
   - Sample from known distribution
   - Best for matching real data exactly

2. **PWM** (if `pwm_file` specified)
   - Generate by sampling from PWM
   - Good balance of realism and diversity

3. **Random** (fallback)
   - Random 6bp sequence
   - Use only if no priors or PWM available

### Verifying PWM Usage

```python
# Generate reads
reads = simulator.generate(num_sequences=100)

# Extract ACC sequences
acc_sequences = []
for read in reads:
    if 'ACC' in read.label_regions:
        for start, end in read.label_regions['ACC']:
            acc_sequences.append(read.sequence[start:end])

# Score them
scorer = PWMScorer(pwm)
scores = scorer.score_multiple_sequences(acc_sequences)

print(f"Mean score: {scores.mean():.3f}")  # Should be >0.8
print(f"Min score: {scores.min():.3f}")
print(f"% above 0.7: {(scores >= 0.7).mean() * 100:.1f}%")
```

## Best Practices

1. **Start with provided PWM**: Use `acc_pwm.txt` if available
2. **Validate PWM**: Check that it generates realistic sequences
3. **Tune threshold**: Based on your false positive tolerance
4. **Monitor scores**: Track ACC scores during simulation
5. **Document**: Note PWM source and version
6. **Version control**: Track PWM files alongside code
7. **Test**: Verify PWM performance on held-out data

## Troubleshooting

### Issue: All ACC sequences look random

**Check:**
```python
# Verify PWM loaded correctly
pwm = load_pwm('acc_pwm.txt')
print(pwm)  # Should show position-specific probabilities

# Check if PWM is being used
scorer = PWMScorer(pwm)
test_scores = [
    scorer.score_sequence('ACCGGG'),  # Should be high
    scorer.score_sequence('TTTTTT'),  # Should be low
]
print(test_scores)
```

### Issue: PWM scores all ~0.5

**Cause:** PWM might be too flat (not informative)

**Solution:** Use a more specific PWM or adjust pseudocounts

### Issue: No ACC detected in reads

**Causes:**
1. Threshold too high
2. Reads don't actually contain ACC
3. ACC sequences have too many errors

**Solutions:**
```python
# Try lower threshold
scorer = PWMScorer(pwm, threshold=0.6)  # Was 0.7

# Search wider region
result = scorer.find_best_match(read, start=0, end=len(read))
```

## Example Workflows

### Workflow 1: Generate and Validate

```python
# Generate ACC sequences
accs = generate_acc_from_pwm(pwm, n=1000)

# Score them
scorer = PWMScorer(pwm)
scores = scorer.score_multiple_sequences(accs)

# Validate
print(f"Generated {len(accs)} sequences")
print(f"Mean score: {scores.mean():.3f}")
print(f"Std score: {scores.std():.3f}")
print(f"Min/Max: {scores.min():.3f} / {scores.max():.3f}")

# Show distribution
import matplotlib.pyplot as plt
plt.hist(scores, bins=50)
plt.xlabel('PWM Score')
plt.ylabel('Count')
plt.title('Generated ACC Scores')
plt.savefig('acc_score_distribution.png')
```

### Workflow 2: Detect in Real Data

```python
from tempest.utils import load_fastq

# Load real reads
for record in load_fastq('reads.fastq', max_reads=100):
    seq = str(record.seq)
    
    # Try to find ACC
    result = scorer.detect_motif(seq)
    
    if result:
        start, end, score, match = result
        print(f"{record.id}: Found {match} at {start}-{end} (score={score:.3f})")
```

### Workflow 3: Build Custom PWM

```python
# Collect your ACC sequences
my_accs = extract_acc_from_alignment('my_data.bam')

# Build PWM
my_pwm = compute_pwm_from_sequences(my_accs)

# Save
save_pwm(my_pwm, 'my_custom_pwm.txt')

# Test
scorer = PWMScorer(my_pwm)
for seq in my_accs[:10]:
    score = scorer.score_sequence(seq)
    print(f"{seq}: {score:.3f}")
```

## References

- Stormo, G. D. (2000). DNA binding sites: representation and discovery. *Bioinformatics*, 16(1), 16-23.
- Bailey, T. L., & Elkan, C. (1994). Fitting a mixture model by expectation maximization. *ISMB*, 2, 28-36.

## Summary

- **PWMs** model position-specific base preferences
- **ACC sequences** follow IUPAC pattern ACCSSV
- **Scoring** uses log-odds ratios normalized to [0, 1]
- **Generation** samples from PWM probabilities
- **Threshold 0.7** recommended for balanced performance
- **Integration** with simulation for realistic training data

For more examples, see:
- `test_simulator.py` - PWM usage examples
- `tempest/core/pwm.py` - Implementation details
- `TRAINING_GUIDE.md` - Integration with training pipeline
