# Comparison and Evaluation of Hyperplane Fitting Methods

English | [Chinese](README-Method-Comparison_zh.md)

This document explains how to use the provided evaluation framework to compare different hyperplane fitting methods, including **2D line fitting** and **3D plane fitting** task.


---
## 📁 Additional Project Structure

```
hyperplanes_fitting/
├── visualization/                # Visualization tools
├── results/                      # Supplementary test results
└── scripts-for-eval/             # Additional baselines and test methods
```


## Methods

The evaluation framework supports comparison of the following 8 methods:

| Method | Type | 2D | 3D | Model Count Mode |
|------|------|----|----|-------------|
| **Ours (Full Pipeline)** | Manifold Optimization | ✓ | ✓ | Auto-detection |
| **PARSAC (Known)** | Parallel Random Sample Consensus | ✓ | ✓ | Known |
| **PARSAC (Unknown)** | Parallel Random Sample Consensus | ✓ | ✓ | Auto-detection |
| **SupeRANSAC (Known)** | Sequential RANSAC | ✓ | ✓ | Known |
| **SupeRANSAC (Unknown)** | Sequential RANSAC | ✓ | ✓ | Auto-detection |
| **RANSAC** | Random Sample Consensus | ✓ | ✓ | Known |
| **K-Means** | Clustering + SVD | ✓ | ✓ | Known |
| **GMM** | Gaussian Mixture Model + SVD | ✓ | ✓ | Known |

### Explanation

- **Known Model Count Mode**: Uses the true number of hyperplanes as an input parameter.
- **Auto-detection Mode**: The algorithm automatically estimates the number of hyperplanes (based on score thresholds or stopping criteria).

---

## Quick Evaluation

### Full Evaluation

```bash
cd /path/to/hyperplanes_fitting-main
conda activate parsac

# Run 2D evaluation (all methods)
python scripts-for-eval/run_all_evaluations.py --dim 2

# Run 3D evaluation (all methods)
python scripts-for-eval/run_all_evaluations.py --dim 3

# Run all evaluations (2D + 3D)
python scripts-for-eval/run_all_evaluations.py --dim all
```

### Selective Evaluation

```bash
# Evaluate only specific methods
python scripts-for-eval/run_all_evaluations.py --dim 2 \
    --methods parsac_known parsac_unknown superansac_known

# Quiet mode (no detailed information printed)
python scripts-for-eval/run_all_evaluations.py --dim 2 --quiet

# Specify the number of samples
python scripts-for-eval/run_all_evaluations.py --dim 2 --num_samples 10
```

---

## Evaluation Results

### 2D Line Fitting Results

| Method | Total Cost | Cost Ratio | Model Count | Runtime |
|------|------------|------------|--------|---------|
| PARSAC (Known) | 5.57±0.30 | 0.93±0.03 | 4.0±0.0 | 0.016s |
| PARSAC (Unknown) | 5.57±0.30 | 0.93±0.03 | 4.0±0.0 | 0.021s |
| SupeRANSAC (Known) | 5.61±0.30 | 0.94±0.03 | 4.0±0.0 | 0.016s |
| SupeRANSAC (Unknown) | 5.61±0.30 | 0.94±0.03 | 4.0±0.0 | 0.017s |
| RANSAC | 188.57±147.28 | 31.42±23.93 | 4.0±0.0 | 0.023s |
| K-Means | 40.95±22.34 | 6.89±3.75 | 4.0±0.0 | 0.013s |
| GMM | 44.25±14.31 | 7.42±2.44 | 4.0±0.0 | 0.350s |

### 3D Plane Fitting Results

| Method | Total Cost | Cost Ratio | Model Count | Runtime |
|------|------------|------------|--------|---------|
| PARSAC (Known) | 5.50±0.30 | 0.92±0.03 | 4.0±0.0 | 0.018s |
| PARSAC (Unknown) | 5.56±0.39 | 0.93±0.06 | 4.0±0.0 | 0.023s |
| SupeRANSAC (Known) | 5.55±0.30 | 0.93±0.03 | 4.0±0.0 | 0.038s |
| SupeRANSAC (Unknown) | 5.59±0.32 | 0.93±0.03 | 4.0±0.0 | 0.038s |
| RANSAC | 6.14±0.61 | 1.02±0.08 | 4.0±0.0 | 0.085s |
| K-Means | 64.81±21.75 | 10.83±3.59 | 4.0±0.0 | 0.001s |
| GMM | 30.45±25.96 | 5.12±4.42 | 4.0±0.0 | 0.015s |

### Result Analysis

1. **PARSAC and SupeRANSAC** perform the best in both tasks
2. **Known VS Unknown Model Count**: The performance of the auto-detection mode slightly degrades
3. **RANSAC** performs poorly in the 2D task but improves significantly in the 3D task.
4. **K-Means and GMM** show moderate performance in hyperplane fitting tasks.

---

## Explanation of Evaluation Metrics

### Total Cost (TC)
Total distance from points to the nearest hyperplane:
$$TC = \sum_{i=1}^{N} \min_{j} d(x_i, H_j)$$

### Cost Ratio (CR)
Ratio of the result cost to the ground truth cost:
$$CR = \frac{TC_{result}}{TC_{ground\_truth}}$$

- CR = 1.0 indicates perfect consistency with the ground truth.
- CR < 1.0 indicates better performance than the ground truth.
- CR > 1.0 indicates worse performance than the ground truth.

### H-bar Distance
Distance in the hyperplane parameter space:
$$d_{H-bar} = \sum_{i} \min_{j} ||\hbar_i - g_j||$$

where $\hbar = d \cdot n$ is the H-bar representation.

### Model Count Error
Error in the number of estimated models:
$$MCE = |K_{result} - K_{ground\_truth}|$$

---

## Run Single Method

### PARSAC

```bash
# 2D with known model count
python scripts-for-eval/parsac/evaluate_parsac.py --known_count

# 2D with auto-detection
python scripts-for-eval/parsac/evaluate_parsac.py

# 3D with known model count
python scripts-for-eval/parsac/evaluate_parsac_3d.py --known_count

# 3D with auto-detection
python scripts-for-eval/parsac/evaluate_parsac_3d.py
```

### SupeRANSAC

```bash
# 2D with known model count
python scripts-for-eval/superansac/evaluate_superansac.py --known_count

# 2D with auto-detection
python scripts-for-eval/superansac/evaluate_superansac.py

# 3D with known model count
python scripts-for-eval/superansac/evaluate_superansac_3d.py --known_count

# 3D with auto-detection
python scripts-for-eval/superansac/evaluate_superansac_3d.py
```

### Traditional Methods (RANSAC, K-Means, GMM)

These methods are evaluated using the original implementations in the `compared_alg/` directory.

```bash
# Use the original evaluation framework
python compared_alg/evaluate.py

# Or use the unified evaluation script
python scripts-for-eval/run_all_evaluations.py --methods ransac kmeans gmm
```

---

## Results Location

Evaluation results are saved in the following directory:

```
results/
├── 2d/
│   ├── parsac_known/
│   │   ├── parsac_known_results.csv
│   │   └── parsac_known_summary.txt
│   ├── parsac_unknown/
│   ├── superansac_known/
│   ├── superansac_unknown/
│   ├── ransac/
│   ├── kmeans/
│   └── gmm/
│
└── 3d/
    ├── parsac_known/
    ├── parsac_unknown/
    ├── superansac_known/
    ├── superansac_unknown/
    ├── ransac/
    ├── kmeans/
    └── gmm/
```

Each method directory contains:
- `*_results.csv`: Detailed evaluation results for each sample.
- `*_summary.txt`: Statistical summary.

---

## Reference Documents

- [PARSAC/SupeRANSAC Implementation](../scripts-for-eval/README-PARSAC-SupeRANSAC.md): Detailed explanation of the algorithm implementation.
