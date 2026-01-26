# Evaluation Guide for PARSAC and SupeRANSAC

English | [Chinese](README-PARSAC-SupeRANSAC_zh.md)

This document explains how to evaluate the PARSAC and SupeRANSAC methods on our **2D line fitting** and **3D plane fitting** datasets to enable fair comparison with other baseline methods in the paper.


## ⚠️ Important Note: Implementation Differences

**The implementations of "PARSAC" and "SupeRANSAC" in this project are simplified adaptations for 2D line fitting tasks, not full reproductions of the original algorithms.**

### Why Can't We Use the Original Algorithms Directly?

| Original Algorithm | Design Purpose | Our Task | Differences |
|----------|----------|------------|------|
| **PARSAC** | Vanishing point detection, fundamental matrix/homography matrix estimation | 2D point cloud multi-line fitting | Completely different problem definitions |
| **SupeRANSAC** | Robust estimation for image feature matching | 2D point cloud multi-line fitting | Does not support general line fitting |

### Implementation Strategy

We adopted the following strategies to ensure fair comparison:

1. **PARSAC**: A geometric version implemented by borrowing its core ideas (parallel hypothesis generation, soft inlier weighting, greedy model selection)
2. **SupeRANSAC**: Sequential RANSAC implemented by borrowing its RANSAC variants (PROSAC sampling, MSAC/MAGSAC scoring)

**Confidence Assessment: 60-70%** - These implementations retain the core algorithmic ideas of the original methods but remove components dependent on deep learning.

---

## 🔬 Comparative Analysis of Original Algorithms vs. Our Implementations

### PARSAC Comparative Analysis

#### Original PARSAC Algorithm Pipeline

```
Input: Feature points/line segments + image features
    ↓
┌─────────────────────────────────────────┐
│ 1. Neural Network (CNNet)               │
│    - Input: Point features (N × input_dim) │
│    - Output: log_inlier_weights (sampling weights) │
│           log_sample_weights (instance weights) │
│    - 5-layer ResNet-style 1D convolution │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 2. Parallel Sampling (sampling.py)       │
│    - Importance sampling with learned weights │
│    - Generate M×S×K minimal sets         │
│    - minimal_solver computes hypotheses  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 3. Residual Calculation + Soft Inlier Counting │
│    - soft_inlier: sigmoid(β×(τ-d)/τ)    │
│    - Weighted inlier ratio as hypothesis score │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 4. Hypothesis Selection + Clustering (postprocessing.py) │
│    - Greedy selection: select hypothesis with maximum gain each time │
│    - Assign points to nearest hypothesis │
└─────────────────────────────────────────┘
    ↓
Output: Multiple model instances + point labels
```

#### Our SimplePARSACLineFitter Implementation

```
Input: 2D point cloud (N × 2)
    ↓
┌─────────────────────────────────────────┐
│ 1. Random Sampling (replaces neural network) │
│    - Uniform random sampling of point pairs │
│    - No learned weights                 │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 2. Hypothesis Generation                │
│    - Compute line from point pairs: n = (p2-p1)⊥ │
│    - Generate H candidate lines         │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 3. Soft Inlier Counting ✓ (same as original) │
│    - soft_inlier: 1/(1+exp(β×(d-τ)))    │
│    - Calculate total soft inlier score for each hypothesis │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 4. Greedy Selection ✓ (similar to original) │
│    - Select hypothesis with highest score │
│    - Penalize candidates similar to selected hypotheses │
│    - Mark assigned points               │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 5. Iterative Refinement                 │
│    - Cluster assignment + SVD refitting │
└─────────────────────────────────────────┘
    ↓
Output: Multiple lines + point labels
```

#### Key Difference Summary

| Component | Original PARSAC | Our Implementation | Fidelity |
|------|------------|-----------|-------|
| **Sampling Strategy** | Neural network-learned weights | Uniform random | ❌ 50% |
| **Hypothesis Generation** | minimal_solver | Point pair fitting | ✓ 90% |
| **Soft Inlier Function** | sigmoid(β(τ-d)/τ) | 1/(1+exp(β(d-τ))) | ✓ 95% |
| **Hypothesis Selection** | Greedy + deduplication | Greedy + similarity penalty | ✓ 85% |
| **Refinement** | SVD (VP only) | Iterative SVD | ✓ 90% |
| **Supported Problems** | VP/F/H matrices | 2D lines | ⚡ Adapted |

#### 🔍 Implementation Notes for Unknown Hyperplane Count Mode

**Original PARSAC's "unknown model count" feature:**

The original PARSAC paper does support scenarios with "unknown number of models". Its core mechanism is:
1. The neural network predicts **instance assignment probabilities** for each point
2. Via softmax normalization, each point can belong to K possible model instances
3. In the post-processing stage, low-confidence models are filtered via thresholding

**Our "unknown mode" implementation (`auto_detect=True`)：**

Since we removed the neural network component, we implemented an automatic detection mechanism based on **greedy score decay**:

```python
def _auto_detect_num_models(self, points, hypotheses, inlier_scores,
                            max_models=10, min_score_ratio=0.1, 
                            min_inliers_ratio=0.05):
    """
    Automatic detection strategy: greedily select hypotheses until termination conditions are met
    """
    N = len(points)
    selected_indices = []
    assigned_points = np.zeros(N, dtype=bool)
    first_score = None
    
    for _ in range(max_models):
        # Calculate scores for remaining unassigned points
        remaining_scores = np.sum(inlier_scores[~assigned_points, :], axis=0)
        remaining_scores[selected_indices] = -inf
        
        best_idx = np.argmax(remaining_scores)
        best_score = remaining_scores[best_idx]
        
        # Record the first model's score as reference
        if first_score is None:
            first_score = best_score
        
        # Termination condition 1: New model score is too low relative to the first
        if best_score < first_score * min_score_ratio:
            break
        
        # Termination condition 2: New model has too few inliers
        hard_inliers = inlier_scores[:, best_idx] > 0.5
        num_new_inliers = np.sum(hard_inliers & ~assigned_points)
        if num_new_inliers < N * min_inliers_ratio:
            break
        
        selected_indices.append(best_idx)
        assigned_points |= hard_inliers
    
    return len(selected_indices)
```

**Comparison with the original algorithm:**

| Feature | Original PARSAC | Our Implementation |
|------|------------|-----------|
| **Model Count Detection** | Neural network output + softmax | Greedy score decay |
| **Termination Conditions** | Confidence threshold | Score ratio + inlier ratio |
| **Theoretical Basis** | Learned prior distribution | Heuristic rules |
| **Applicable Scenarios** | Trained specific tasks | General geometric fitting |
| **Parameter Sensitivity** | Low (neural network learned) | Medium (requires tuning) |

**Parameter Explanation:**
- `max_models=10`: Upper limit of the maximum number of detectable models
- `min_score_ratio=0.1`: The score of the new model must be at least 10% of the first model
- `min_inliers_ratio=0.05`: The new model must cover at least 5% of remaining points

**Note:** This heuristic method works well in simple scenarios but may be less robust than the original neural network-based method in complex scenarios (e.g., large differences in model sizes, high noise).

---

### SupeRANSAC Comparative Analysis

#### Original SupeRANSAC Algorithm Pipeline

```
Input: Point correspondences + image size
    ↓
┌─────────────────────────────────────────┐
│ RANSAC Main Loop                        │
│ while (iter < max_iter && iter < adaptive_iter):
│   │
│   ├─ Sampler (SamplerType)              │
│   │  - Uniform / PROSAC / NAPSAC        │
│   │  - ImportanceSampler / ARSampler    │
│   │
│   ├─ Model Estimation (estimator)       │
│   │  - Homography / Fundamental / Essential │
│   │  - RigidTransform / AbsolutePose    │
│   │
│   ├─ Scoring (ScoringType)              │
│   │  - RANSAC / MSAC / MAGSAC / ACRANSAC │
│   │
│   ├─ Local Optimization (LocalOptimizationType) │
│   │  - LSQ / IteratedLSQ                │
│   │  - NestedRANSAC / GCRANSAC          │
│   │
│   └─ Termination Criterion (adaptive iterations) │
└─────────────────────────────────────────┘
    ↓
Output: Single best model + inlier mask
```

**Key Features**:
- Core implementation in C++ with Python bindings
- Designed exclusively for **single-model** estimation
- Supports combinations of multiple sampling, scoring, and optimization strategies
- Built-in spatial partitioning for acceleration

#### Our SequentialRANSAC2DLine Implementation

```
Input: 2D point cloud (N × 2)
    ↓
┌─────────────────────────────────────────┐
│ Sequential Multi-Model Loop             │
│ for model_idx in range(max_models):     │
│   │
│   ├─ RANSAC Single-Model Fitting        │
│   │  ├─ PROSAC Sampling ✓               │
│   │  ├─ Line fitting from point pairs   │
│   │  ├─ MSAC/MAGSAC Scoring ✓           │
│   │  └─ Adaptive iterations ✓           │
│   │
│   ├─ LSQ Refinement ✓                   │
│   │  └─ Refit with all inliers via SVD  │
│   │
│   └─ Remove inliers and proceed to next iteration │
└─────────────────────────────────────────┘
    ↓
Output: Multiple lines + point labels
```

#### Key Difference Summary

| Component | Original SupeRANSAC | Our Implementation | Fidelity |
|------|----------------|-----------|-------|
| **Design Goal** | Single-model estimation	 | Multi-model fitting | ⚡ Adapted |
| **Sampler** | Multiple options | PROSAC | ✓ 80% |
| **Scoring** | RANSAC/MSAC/MAGSAC | MSAC/MAGSAC | ✓ 90% |
| **Local Optimization** | LSQ/GCRANSAC/Nested | LSQ (SVD) | ✓ 70% |
| **Supported Models** | H/F/E/Rigid/Pose | 2D lines | ⚡ Adapted |
| **Implementation Language** | C++ + pybind11 | Python | ⚠️ Performance difference |
| **Multi-Model Strategy** | None (single-model) | Sequential | ⚡ Extended |

---

### 🌐 3D Plane Fitting Algorithm Implementation

We extended the 2D algorithms to 3D plane fitting, with key modifications to the hypothesis generation and residual calculation components.

#### SimplePARSACPlaneFitter (3D)

```
Input: 3D point cloud (N × 3)
    ↓
┌─────────────────────────────────────────┐
│ 1. Random Three-Point Sampling          │
│    - Uniform random sampling of 3 points │
│    - Corresponding to 2-point sampling in 2D │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 2. Plane Hypothesis Generation          │
│    - Compute normal vector via cross product of two vectors: │
│      v1 = p2 - p1, v2 = p3 - p1          │
│      n = v1 × v2 / ||v1 × v2||           │
│    - Compute distance: d = n · p1        │
│    - Generate H candidate planes         │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 3. Soft Inlier Counting (same as 2D)    │
│    - Distance from point to plane: |n·x - d| │
│    - soft_inlier: 1/(1+exp(β×(r-τ)))     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 4. Greedy Selection + SVD Refinement    │
│    - Select plane with highest score    │
│    - Refit inliers via SVD              │
└─────────────────────────────────────────┘
    ↓
Output: Multiple planes (n1, n2, n3, d) + point labels
```

**Key Code Snippet**:
```python
# 3D plane hypothesis generation
p1, p2, p3 = points[idx]  # Sample 3 points
v1 = p2 - p1
v2 = p3 - p1
normal = np.cross(v1, v2)  # Cross product to get normal vector
normal = normal / np.linalg.norm(normal)
d = np.dot(normal, p1)
```

#### SequentialRANSAC3DPlane

```
Input: 3D point cloud (N × 3)
    ↓
┌─────────────────────────────────────────┐
│ Sequential Multi-Model Loop             │
│ for plane_idx in range(max_planes):     │
│   │
│   ├─ RANSAC Single-Plane Fitting        │
│   │  ├─ 3-point sampling (vs. 2-point in 2D) │
│   │  ├─ Compute normal vector via cross product │
│   │  ├─ MSAC/MAGSAC Scoring             │
│   │  └─ Adaptive iterations             │
│   │
│   ├─ SVD Refinement                     │
│   │  └─ Refit plane with all inliers    │
│   │
│   └─ Remove inliers and proceed to next iteration │
└─────────────────────────────────────────┘
    ↓
Output: Multiple planes + point labels
```

#### 2D vs. 3D Implementation Difference Summary

| Component | 2D Line | 3D Plane | Modification |
|------|---------|---------|------|
| **Minimal Sampling** | 2 points | 3 points | Add 1 point |
| **Hypothesis Generation** | Perpendicular vector | Cross product | Different algorithms |
| **Model Parameters** | (a, b, c) or (n1, n2, d) | (n1, n2, n3, d) | Add 1 dimension |
| **Residual Calculation** | \|ax + by + c\| / √(a²+b²) | \|n·x - d\| | Similar formula |
| **SVD Refinement** | 2D covariance | 3D covariance | Dimension extension |

---

## 🔧 Environment Setup

### 1. Official Repository Reference

The official repository code is located in the `potential-repositories/` directory for reference only and does not need to be modified:

- **PARSAC**: `potential-repositories/parsac/`
- **SupeRANSAC**: `potential-repositories/superansac/`

### 2. Evaluation Environment Configuration

Our evaluation scripts have simplified dependencies on the original repositories and can run with only a basic Python environment:

```bash
# Create or activate environment
conda create -n parsac python=3.10
conda activate parsac

# Install dependencies
pip install numpy pandas scipy matplotlib scikit-learn
```

## 📁 Directory Structure

Files related to the newly added evaluation in this project are concentrated in the following three directories:

```
hyperplanes_fitting-main/
├── scripts-for-eval/               # Evaluation script directory
│   ├── evaluate_utils.py           # Unified evaluation tools (metrics consistent with the paper)
│   ├── compare_all_methods.py      # 2D multi-method comparison script
│   ├── compare_all_methods_3d.py   # 3D multi-method comparison script
│   ├── data_utils.py               # Data reading tools
│   ├── metrics.py                  # Evaluation metrics module
│   │
│   ├── ours/                       # Ours (manifold optimization) evaluation module
│   │   ├── __init__.py
│   │   └── evaluate_ours.py        # 2D/3D: Ours evaluation script
│   │
│   ├── parsac/                     # PARSAC evaluation module
│   │   ├── __init__.py
│   │   ├── line_fitter.py          # 2D: SimplePARSACLineFitter implementation
│   │   ├── evaluate_parsac.py      # 2D: PARSAC evaluation script
│   │   ├── plane_fitter_3d.py      # 3D: SimplePARSACPlaneFitter implementation
│   │   └── evaluate_parsac_3d.py   # 3D: PARSAC evaluation script
│   │
│   ├── superansac/                 # SupeRANSAC evaluation module
│   │   ├── __init__.py
│   │   ├── sequential_ransac.py    # 2D: Sequential RANSAC implementation
│   │   ├── evaluate_superansac.py  # 2D: SupeRANSAC evaluation script
│   │   ├── sequential_ransac_3d.py # 3D: Sequential RANSAC 3D implementation
│   │   └── evaluate_superansac_3d.py # 3D: SupeRANSAC evaluation script
│   │
│   └── compared_alg_3d/            # 3D comparison algorithm module
│       ├── __init__.py
│       ├── RANSAC_3D.py            # 3D RANSAC plane fitting
│       ├── K_Means_3D.py           # 3D K-Means + SVD fitting
│       └── GMM_3D.py               # 3D GMM + SVD fitting
│
├── results/                        # Result output directory
│   ├── 2d/                         # 2D line fitting results
│   │   ├── ours/                   # Ours 2D results
│   │   ├── parsac/                 # PARSAC 2D results
│   │   ├── superansac/             # SupeRANSAC 2D results
│   │   └── figures/                # 2D comparison charts
│   │
│   └── 3d/                         # 3D plane fitting results
│       ├── ours/                   # Ours 3D results
│       ├── parsac/                 # PARSAC 3D results
│       ├── superansac/             # SupeRANSAC 3D results
│       ├── ransac/                 # RANSAC 3D results
│       ├── kmeans/                 # K-Means 3D results
│       ├── gmm/                    # GMM 3D results
│       └── figures/                # 3D comparison charts
│
├── visualization/                  # Visualization script directory
│   ├── plot_results.py             # 2D result visualization tool
│   └── visualize_3d.py             # 3D visualization tool (plotly)
│
├── csv_dataset/                    # 2D dataset
├── csv_groundtruth/                # 2D ground truth
├── csv_dataset_3d/                 # 3D dataset
├── csv_groundtruth_3d/             # 3D ground truth
│
└── README-PARSAC-SupeRANSAC.md     # This document
```

**Note**: The original project structure (`algorithm/`, `compared_alg/`, `data/`, `csv_dataset/` etc.) remains unchanged.

## 🚀 Quick Start

### 1. Run PARSAC Evaluation

```bash
cd /path/to/hyperplanes_fitting-main

# Use known model count (recommended, consistent with paper settings)
python scripts-for-eval/parsac/evaluate_parsac.py --known_count

# Auto-detect model count
python scripts-for-eval/parsac/evaluate_parsac.py
```

### 2. Run SupeRANSAC Evaluation

```bash
# Use known model count
python scripts-for-eval/superansac/evaluate_superansac.py --known_count

# Auto-detect model count
python scripts-for-eval/superansac/evaluate_superansac.py
```

### 3. Run Multi-Method Comparison

```bash
# Compare all methods and generate charts
python scripts-for-eval/compare_all_methods.py \
    --methods parsac superansac ransac kmeans gmm \
    --run_eval --known_count
```

---

## 🌐 3D Plane Fitting Evaluation

### Overview

In addition to 2D line fitting, we also provide evaluation scripts for **3D plane fitting** to fit multiple planes in the $\mathbb{R}^3$ space.

### 1. Generate 3D Test Data

```bash
cd /path/to/hyperplanes_fitting-main

# Generate 3D data (4 planes, 100 points per plane, noise 0.1)
python data/csv_data_generator.py \
    --dim 3 \
    --data_dir csv_dataset_3d \
    --gt_dir csv_groundtruth_3d \
    --num_samples 20 \
    --num_hyperplanes 4 \
    --points_per_hyperplane 100 \
    --noise 0.1
```

**3D Data Format**:
- Data files (`csv_dataset_3d/X.csv`): Three columns of coordinate `x,y,z`
- Ground truth files (`csv_groundtruth_3d/X.csv`): Five columns `n1,n2,n3,d,totaldistance`

### 2. Run PARSAC 3D Evaluation

```bash
# Use known model count (recommended)
python scripts-for-eval/parsac/evaluate_parsac_3d.py --known_count

# Auto-detect model count
python scripts-for-eval/parsac/evaluate_parsac_3d.py

# Custom parameters
python scripts-for-eval/parsac/evaluate_parsac_3d.py \
    --data_dir csv_dataset_3d \
    --gt_dir csv_groundtruth_3d \
    --output_dir results/3d/parsac \
    --num_samples 20 \
    --num_hypotheses 500 \
    --inlier_threshold 0.2 \
    --known_count
```

### 3. Run SupeRANSAC 3D Evaluation

```bash
# Use known model count
python scripts-for-eval/superansac/evaluate_superansac_3d.py --known_count

# Auto-detect model count
python scripts-for-eval/superansac/evaluate_superansac_3d.py

# Custom parameters
python scripts-for-eval/superansac/evaluate_superansac_3d.py \
    --data_dir csv_dataset_3d \
    --gt_dir csv_groundtruth_3d \
    --output_dir results/3d/superansac \
    --num_samples 20 \
    --max_iterations 1000 \
    --inlier_threshold 0.3 \
    --known_count
```

### 4. 3D Visualization

```bash
# Visualize fitting results of a single sample
python visualization/visualize_3d.py \
    --data_file csv_dataset_3d/0.csv \
    --gt_file csv_groundtruth_3d/0.csv \
    --output_dir results/3d/figures \
    --method parsac  # or superansac

# Interactive HTML visualization (recommended)
python visualization/visualize_3d.py \
    --data_file csv_dataset_3d/0.csv \
    --gt_file csv_groundtruth_3d/0.csv \
    --format html
```

### 5. 3D Evaluation Parameter Explanation

#### PARSAC 3D (`evaluate_parsac_3d.py`)

| Parameter | Default Value | Description |
|------|--------|------|
| `--data_dir` | `csv_dataset_3d/` | 3D data directory |
| `--gt_dir` | `csv_groundtruth_3d/` | 3D ground truth directory |
| `--output_dir` | `results/3d/parsac/` | Output directory |
| `--num_hypotheses` | 500 | Number of candidate planes to generate |
| `--inlier_threshold` | 0.2 | Inlier determination threshold |
| `--known_count` | False | Use known number of planes |

#### SupeRANSAC 3D (`evaluate_superansac_3d.py`)

| Parameter | Default Value | Description |
|------|--------|------|
| `--data_dir` | `csv_dataset_3d/` | 3D data directory |
| `--gt_dir` | `csv_groundtruth_3d/` | 3D ground truth directory |
| `--output_dir` | `results/3d/superansac/` | Output directory |
| `--max_iterations` | 1000 | Maximum iterations for a single RANSAC |
| `--inlier_threshold` | 0.3 | Inlier determination threshold |
| `--min_inliers` | 15 | Minimum number of inliers required for a plane |
| `--known_count` | False | Use known number of planes |

### 6. 3D Algorithm Implementation Notes

#### SimplePARSACPlaneFitter (3D)

**Key Differences from the 2D Version**:
- **Hypothesis Generation**: Compute plane normal vector (cross product) from 3 points instead of computing a line from 2 points
- **Dimension**: Hypothesis format `(n1, n2, n3, d)` instead of `(a, b, c)`

```python
# 3D plane hypothesis generation
p1, p2, p3 = points[idx]  # Sample 3 points
v1, v2 = p2 - p1, p3 - p1
normal = np.cross(v1, v2)  # Cross product to get normal vector
normal = normal / np.linalg.norm(normal)
d = np.dot(normal, p1)
```

#### SequentialRANSAC3DPlane

**Key Differences from the 2D Version**:
- **Minimal Sampling**: 3 points to fit a plane instead of 2 points to fit a line
- **Distance Calculation**: Perpendicular distance from point to plane

```python
# 3D distance from point to plane
distances = np.abs(np.dot(points, normal) - d)
```

---

## 📖 Detailed Parameter Description

### PARSAC Evaluation Script (`evaluate_parsac.py`)

```bash
python scripts-for-eval/parsac/evaluate_parsac.py [OPTIONS]
```

| Parameter | Type | Default Value | Description |
|------|------|--------|------|
| `--data_dir` | str | `csv_dataset/` | Path to data directory |
| `--gt_dir` | str | `csv_groundtruth/` | Path to ground truth directory |
| `--output_dir` | str | `results/parsac/` | Path to output directory |
| `--num_samples` | int | 20 | Number of samples to evaluate |
| `--known_count` | flag | False | Use known number of models (number of lines in Ground Truth) |
| `--num_hypotheses` | int | 500 | Number of candidate hypotheses to generate (larger = more accurate but slower) |
| `--inlier_threshold` | float | 0.15 | Inlier determination threshold (point is inlier if distance to line < threshold) |
| `--num_iterations` | int | 3 | Number of iterations to take the optimal result |

**Example**:
```bash
# High-precision mode (more hypotheses)
python scripts-for-eval/parsac/evaluate_parsac.py \
    --known_count \
    --num_hypotheses 1000 \
    --inlier_threshold 0.1

# Fast mode
python scripts-for-eval/parsac/evaluate_parsac.py \
    --num_hypotheses 200 \
    --num_iterations 1
```

### SupeRANSAC Evaluation Script (`evaluate_superansac.py`)

```bash
python scripts-for-eval/superansac/evaluate_superansac.py [OPTIONS]
```

| Parameter | Type | Default Value | Description |
|------|------|--------|------|
| `--data_dir` | str | `csv_dataset/` | Path to data directory |
| `--gt_dir` | str | `csv_groundtruth/` | Path to ground truth directory |
| `--output_dir` | str | `results/parsac/` | Path to output directory |
| `--num_samples` | int | 20 | Number of samples to evaluate |
| `--known_count` | flag | False | Use known number of models (number of lines in Ground Truth) |
| `--max_iterations` | int | 1000 | Maximum iterations for a single RANSAC |
| `--inlier_threshold` | float | 0.3 | Inlier determination threshold |
| `--min_inliers` | int | 10 | Minimum number of inliers required for a line |

**Example**:
```bash
# Strict inlier threshold
python scripts-for-eval/superansac/evaluate_superansac.py \
    --known_count \
    --inlier_threshold 0.2 \
    --min_inliers 15

# Relaxed settings (suitable for noisy data)
python scripts-for-eval/superansac/evaluate_superansac.py \
    --inlier_threshold 0.5 \
    --min_inliers 8
```

### Multi-Method Comparison Script (`compare_all_methods.py`)

```bash
python scripts-for-eval/compare_all_methods.py [OPTIONS]
```

| Parameter | Type | Default Value | Description |
|------|------|--------|------|
| `--methods` | list | `['parsac', 'superansac', 'ransac', 'kmeans', 'gmm']` | List of methods to compare |
| `--results_dir` | str | `results/` | Directory to read existing results |
| `--output_dir` | str | `results/figures/` | Directory to output charts |
| `--run_eval` | flag | False | Automatically run evaluation if results do not exist |
| `--known_count` | flag | False | Use known model count during evaluation |

**Supported Methods**:
- `parsac`: PARSAC algorithm
- `superansac`: Sequential RANSAC (simulating SupeRANSAC)
- `ransac`: sklearn RANSAC
- `kmeans`: K-Means clustering
- `gmm`: Gaussian Mixture Model
- `ours`: Our manifold optimization method

**Example**:
```bash
# Compare only PARSAC and SupeRANSAC
python scripts-for-eval/compare_all_methods.py \
    --methods parsac superansac \
    --run_eval

# Compare all methods
python scripts-for-eval/compare_all_methods.py \
    --methods parsac superansac ransac kmeans gmm ours \
    --run_eval --known_count
```

## 📊 Output Result Explanation

### 1. CSV File

Format of `results/parsac/parsac_results.csv`:

| Column Name | Description |
|------|------|
| `sample_id` | Sample ID (0-19) |
| `total_cost` | Sum of distances from all points to the nearest hyperplane (TC) |
| `cost_ratio` | total_cost / gt_total_cost |
| `average_distance` | total_cost / number of points |
| `total_hbar_distance` | Sum of H-bar distances (HE) |
| `model_count` | Number of detected models (HN) |
| `gt_model_count` | Ground truth number of models |
| `model_count_error` | \|detected count - ground truth count\| |
| `runtime` | Runtime (seconds) |

### 2. Summary Statistics File

`results/parsac/parsac_summary.txt` contains the mean and standard deviation of each metric.

### 3. Comparison Charts

| File | Description |
|------|------|
| `multi_panel_comparison.png` | Multi-panel comparison chart showing all metrics |
| `boxplot_comparison.png` | Box plot showing distribution differences |
| `bar_*.png` | Bar charts for each metric |
| `comparison_table.txt` | Text-format comparison table (including LaTeX format) |

## 📐 Evaluation Metrics Explanation

This evaluation framework uses metric definitions identical to those in the paper:

### 1. HN (Hyperplane Number) - Model Count

The number of detected hyperplanes.

### 2. TC (Total Cost) - Total Cost

$$\text{TC} = \sum_{i=1}^{n} \min_j |n_j^T x_i - d_j|$$

Sum of distances from all points to their nearest fitted hyperplane.

### 3. HE (H-bar Error) - H-bar Error

$$\hbar_j = d_j \cdot n_j$$

$$\text{HE} = \sum_{j=1}^{m} \min_k \|\hbar_j^{\text{result}} - \hbar_k^{\text{GT}}\|$$

Sum of distances between fitted hyperplanes and ground truth hyperplanes in H-bar space.

### Additional Metrics

- **Cost Ratio**: TC / GT_TC, ideal value is 1.0
- **Average Distance**: TC / n, average fitting error per point
- **Model Count Error**: |HN - GT_HN|

---

## 🔬 Comparison with Other Methods

This section details the core ideas and implementation differences of various methods we compared.

### Overview of Comparison Methods

| Method | Type | Core Idea | Requires Known Model Count? |
|------|------|----------|-------------------|
| **Ours (Full Pipeline)** | Manifold Optimization | Riemannian gradient descent on spherical manifold | ❌ Auto-detectable |
| **PARSAC** | RANSAC Variant | Parallel hypothesis generation + soft inliers + greedy selection | ✓ Recommended |
| **SupeRANSAC** | Sequential RANSAC | PROSAC sampling + MSAC scoring + progressive stripping | ✓ Recommended |
| **RANSAC** | Random Sample Consensus | Random sampling + voting + inlier stripping | ✓ Required |
| **K-Means** | Clustering + Fitting | PCA/SVD fitting for each cluster after point clustering | ✓ Required |
| **GMM** | Probabilistic Clustering + Fitting | Weighted SVD fitting after soft clustering via Gaussian Mixture Model | ✓ Required |

### Detailed Explanation of Each Method

#### 1. Ours (Manifold Optimization Method)

**Core Idea**: Constrain hyperplane normal vectors on the unit sphere $\mathbb{S}^{d-1}$ and solve directly using Riemannian optimization.

**Algorithm Pipeline**:
1. **Initial Value Estimation**: Uniformly sample candidate directions on the sphere and filter via sliding window
2. **Soft Assignment**: Use weight matrix $W_{ij}$ to represent the probability that point $i$ belongs to hyperplane $j$
3. **Manifold Optimization**: Optimize normal vectors on the sphere using steepest descent method
4. **Hard Assignment Refinement**: Finally assign points to the nearest hyperplane and fine-tune

**Advantages**:
- No need to pre-know the number of hyperplanes
- Theoretically guaranteed convergence
- Normal vector constraints are automatically satisfied

**Code Location**: `algorithm/hyperplanes_fitting.py`

---

#### 2. PARSAC (Parallel RANSAC)

**Core Idea**: Generate a large number of hypotheses in parallel, use neural networks (replaced with random sampling in our implementation) to learn sampling weights, and greedily select the optimal combination of hypotheses.

**Original Algorithm Features**:
- Neural network predicts sampling weights and instance weights
- Differentiable soft inlier function
- Supports multiple geometric models such as vanishing points and fundamental matrices

**Our Adaptation**:
- Replace neural network with uniform random sampling
- Retain soft inlier counting: $s(r) = \frac{1}{1 + e^{\beta(r - \tau)}}$
- Retain greedy selection strategy

**Code Location**: 
- 2D: `scripts-for-eval/parsac/line_fitter.py`
- 3D: `scripts-for-eval/parsac/plane_fitter_3d.py`

---

#### 3. SupeRANSAC (Sequential RANSAC)

**Core Idea**: Borrow advanced sampling and scoring strategies from SupeRANSAC to fit multiple models in a sequential manner.

**Key Technologies**:
- **PROSAC Sampling**: Prioritize sampling higher-quality points (sorted by inlier probability)
- **MSAC Scoring**: Truncated M-estimator, $\rho(r) = \min(r^2, \tau^2)$
- **MAGSAC Scoring**: Adaptive scoring with marginalized threshold
- **Adaptive Iterations**: Dynamically adjust the number of iterations based on inlier ratio

**Sequential Strategy**:
1. Use RANSAC to find one model
2. Remove inliers of this model
3. Repeat steps 1-2 for remaining points

**Code Location**: 
- 2D: `scripts-for-eval/superansac/sequential_ransac.py`
- 3D: `scripts-for-eval/superansac/sequential_ransac_3d.py`

---

#### 4. RANSAC (Random Sample Consensus)

**Core Idea**: Randomly sample minimal point sets, fit model parameters, count inliers, and iterate to find the best model.

**Algorithm Pipeline**:
```
for iteration in range(max_iters):
    1. Randomly sample k points (2D line: k=2, 3D plane: k=3)
    2. Fit model parameters
    3. Calculate distances from all points to the model
    4. Count inliers (distance < threshold)
    5. Update the best model if the number of inliers is the largest
```

**Multi-Model Strategy**: Sequential (progressive stripping of inliers)

**Advantages**: Simple, robust
**Disadvantages**: Requires careful threshold tuning, low efficiency for multi-model scenarios

**Code Location**: 
- 2D: `compared_alg/others/RANSAC.py`
- 3D: `scripts-for-eval/compared_alg_3d/RANSAC_3D.py`

---

#### 5. K-Means + PCA/SVD

**Core Idea**: First cluster points using K-Means, then fit hyperplanes for each cluster using PCA/SVD.

**Algorithm Pipeline**:
```
1. K-Means Clustering:
   - Initialize K cluster centers
   - Iteratively assign points to the nearest center
   - Update centers to cluster means
   
2. Fit hyperplane for each cluster:
   - Calculate cluster center (centroid)
   - Center the data
   - SVD decomposition: the vector corresponding to the smallest singular value is the normal vector
   - d = n · centroid
```

**Advantages**: Simple and efficient
**Disadvantages**: 
- K-Means is sensitive to initialization
- Assumes clusters are spherically distributed, not suitable for elongated distributions

**Code Location**: 
- 2D: `compared_alg/others/K_Means.py`
- 3D: `scripts-for-eval/compared_alg_3d/K_Means_3D.py`

---

#### 6. GMM (Gaussian Mixture Model)

**Core Idea**: Perform soft clustering using Gaussian Mixture Model (each point belongs to each Gaussian component with a certain probability), then fit hyperplanes for each component using weighted SVD.

**Algorithm Pipeline**:
```
1. Initialize GMM parameters (mean, covariance, mixing coefficients)

2. EM algorithm iteration:
   E step: Calculate posterior probability γ_ik = P(z_i = k | x_i)
   M step: Update parameters
         μ_k = Σ γ_ik x_i / Σ γ_ik
         Σ_k = Σ γ_ik (x_i - μ_k)(x_i - μ_k)^T / Σ γ_ik
         π_k = Σ γ_ik / N

3. Fit hyperplane for each component:
   - Weighted SVD: use γ_ik as weights
   - The eigenvector corresponding to the smallest eigenvalue is the normal vector
```

**Advantages**: 
- Soft clustering, more smooth
- Can handle overlapping regions

**Disadvantages**: 
- High computational cost
- Sensitive to initialization
- Assumes Gaussian distribution

**Code Location**: 
- 2D: `compared_alg/others/GMM.py`
- 3D: `scripts-for-eval/compared_alg_3d/GMM_3D.py`

---

### Method Performance Comparison (Example Results)

#### 2D Line Fitting

| Method | Total Cost | Cost Ratio | H-bar Dist | Runtime |
|------|------------|------------|------------|---------|
| **Ours** | 5.58 ± 0.30 | 0.94 ± 0.03 | 0.07 ± 0.02 | 0.24s |
| PARSAC | 5.57 ± 0.30 | 0.93 ± 0.03 | 0.07 ± 0.03 | 0.02s |
| SupeRANSAC | 5.75 ± 0.37 | 0.96 ± 0.04 | 0.09 ± 0.04 | 0.02s |

#### 3D Plane Fitting

| Method | Total Cost | Cost Ratio | H-bar Dist | Runtime |
|------|------------|------------|------------|---------|
| **Ours** | 5.51 ± 0.30 | 0.92 ± 0.03 | 0.09 ± 0.02 | 2.24s |
| PARSAC | 5.50 ± 0.30 | 0.92 ± 0.03 | 0.09 ± 0.02 | 0.02s |
| SupeRANSAC | 5.70 ± 0.36 | 0.95 ± 0.04 | 0.12 ± 0.03 | 0.04s |
| RANSAC | 6.20 ± 0.73 | 1.03 ± 0.10 | 0.16 ± 0.05 | 0.09s |
| K-Means | 68.4 ± 23.9 | 11.5 ± 4.2 | 7.0 ± 2.7 | 0.001s |
| GMM | 26.8 ± 23.0 | 4.5 ± 3.9 | 2.1 ± 2.2 | 0.02s |

**Note**: The above results are mean ± standard deviation of 20 samples, using known model counts.

---

## 🔍 Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'data'**
   ```bash
   # Ensure running in the project root directory
   cd /path/to/hyperplanes_fitting-main
   ```

2. **pandas not installed**
   ```bash
   pip install pandas scipy matplotlib
   ```

3. **Result files do not exist**
   ```bash
   Use --run_eval parameter to automatically run evaluation
   python scripts-for-eval/compare_all_methods.py --run_eval
   ```

## 📝 Citation

If this evaluation framework is helpful for your research, please cite:

```bibtex
@article{hyperplanes_fitting,
  title={Fitting Unknown Number of Hyperplanes with Manifold Optimization},
  author={},
  journal={},
  year={2026}
}
```

---


## 🎯 Fairness Discussion

### Why This Comparison is Meaningful?

1. **Preserved Algorithmic Ideas**: Our implementations retain the core algorithmic ideas of PARSAC and SupeRANSAC
   - PARSAC: Parallel hypothesis generation + soft inliers + greedy selection
   - SupeRANSAC: PROSAC sampling + MAGSAC scoring + LSQ refinement

2. **Problem Adaptation**: The original algorithms do not support direct multi-line fitting on 2D point clouds, so our adaptation is necessary

3. **Same Starting Point**: All methods (including RANSAC, K-Means, GMM) start from the same input data and evaluation metrics

### Limitations

1. **No Deep Learning Components**: The neural network-based sampling weights of PARSAC are replaced with uniform random sampling
2. **Performance Differences**: The C++ implementation of SupeRANSAC is replaced with Python, which may lead to performance differences
---

## 📄 License

MIT License
