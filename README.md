# Hyperplanes Fitting with Manifold Optimization

[Chinese Version](README_zh.md)
Implementation of the hyperplane fitting algorithm based on manifold optimization, corresponding to the paper "Fitting Unknown Number of Hyperplanes with Manifold Optimization".

## 📖 Project Introduction

This project implements a novel hyperplane fitting algorithm that can automatically identify and fit an unknown number of hyperplanes from noisy point cloud data. Based on manifold optimization technology, the method performs Riemannian optimization on the spherical manifold and supports 2D and 3D data processing.

### Core Features

- 🔍 **Automatic Detection of Hyperplane Count**: No need to pre-specify the number of hyperplanes; the algorithm can automatically discover the number of hyperplanes contained in the data.
- 🎯 **High-Precision Fitting**: Uses manifold optimization to optimize normal vectors on the unit sphere, ensuring compliance with normal vector constraints.
- ⚡ **Flexible Optimization Methods**: Supports three optimization modes (Method A, B, A+B)
- 📊 **Comprehensive Evaluation System**: Provides multiple evaluation metrics to measure fitting quality.
- 🔄 **Implementation of Comparative Algorithms**: Includes traditional algorithms such as RANSAC, K-Means, GMM, and DBSCAN for comparison.

## 🧮 Core Idea of Algorithm

### Problem Definition

Given a set of points $\{x_i\}_{i=1}^n \subset \mathbb{R}^d$ distributed near $m$ unknown hyperplanes. Each hyperplane can be expressed as:

$$\mathcal{H}_j = \{x \in \mathbb{R}^d : n_j^T x = d_j\}$$

where $n_j \in \mathbb{S}^{d-1}$  is the unit normal vector，and $d_j \in \mathbb{R}$ is the signed distance to the origin.

### Objective Function

Minimize the sum of distances from all points to their nearest hyperplane:

$$\min_{n_j, d_j} \sum_{i=1}^n \min_j |n_j^T x_i - d_j|$$

### Procedure of Algorithm

1. **Initial Value Estimation (Algorithm 3)**: 
   - Uniformly sample candidate normal vectors on the unit sphere.
   - For each candidate direction, use a sliding window algorithm to find the optimal hyperplane.
   - Select the initial hyperplane set based on a scoring function.

2. **Soft Optimization (Algorithm 1 - Method A)**: 
   - Perform soft assignment of points to hyperplanes using a weight matrix $W$.
   - Optimize normal vectors using the steepest descent method on the spherical manifold $\mathbb{S}^{d-1}$.
   - Alternately update weights and hyperplane parameters until convergence.

3. **Hard Optimization (Algorithm 1 - Method B)**: 
   - Further optimize each hyperplane based on hard assignment.

### Spherical Manifold Optimization

The normal vector constraint $\|n\|_2 = 1$ defines a spherical manifold. The optimization process includes:

- **Riemannian Gradient**: $\text{grad} f(x) = P_x(\nabla f(x))$，where $P_x$ is the projection onto the tangent space.
- **Retraction**: $R_x(\xi) = \frac{x + \xi}{\|x + \xi\|}$
- **Line Search**: Use Backtracking line search to determine the step size.


## 📁 Project Structure

```
hyperplanes_fitting/
├── algorithm/                    # Core algorithm module
│   ├── hyperplanes_fitting.py    # Main algorithm class HyperplanesFitting
│   ├── initial_value.py          # Initial value estimation algorithm
│   └── manifold_optimization.py  # Manifold optimization implementation
├── manifolds/                    # Manifold geometry module
│   ├── manifold.py               # Riemannian manifold base class
│   └── sphere.py                 # Spherical manifold implementation
├── solver/                       # Optimization solver
│   ├── baseSolver.py             # Solver base class
│   ├── lineSearch.py             # Line search algorithm
│   └── steepest_descent.py       # Steepest descent method
├── data/                         # Data generation and reading
│   ├── random_data.py            # Random data generator
│   ├── csv_data_generator.py     # CSV dataset generator
│   └── read_data.py              # Data reading tools
├── view/                         # Visualization module
│   └── plot.py                   # Result plotting tool
├── example/                      # Example code
│   ├── single_data_test.py       # Single random data test
│   ├── single_csv_test.py        # Single CSV file test
│   └── dataset_test.py           # Complete dataset test
├── compared_alg/                 # Comparative algorithms
│   └── others/                   # Traditional methods (RANSAC, K-Means, GMM, etc.)
├── csv_dataset/                  # Test datasets
├── csv_groundtruth/              # Ground truth data
├── evaluate.py                   # Evaluation metric calculation
├── optimization_test/            # Optimization method tests in the appendix
└── figures/                      # Output images directory
```

## 🚀 Quick Start

### Environment Requirements

- Python 3.7+
- NumPy
- Matplotlib
- joblib            (for parallel computing)
- scikit-learn      (required for comparative algorithms)

```bash
pip install numpy matplotlib joblib scikit-learn
```

### Examples

#### 1. Single Random Data Test

Test with randomly generated data to intuitively demonstrate the algorithm's effect:

```bash
cd example
python single_data_test.py
```

Parameters can be modified in the this file:
```python
DIM = 2                    # Data dimension (2 or 3)
HYPERPLANES_NUM = 4        # True number of hyperplanes
MAX_POINT_DISTANCE = 0.1   # Maximum noise distance from points to hyperplanes
MIN_POINT_NUM = 30         # Minimum number of points per hyperplane
MAX_POINT_NUM = 30         # Maximum number of points per hyperplane
METHOD = "3"               # Optimization method: "1"=A, "2"=B, "3"=A+B
INITIAL = True             # Whether to use initial value estimation
```

#### 2. Single CSV File Test

```bash
cd example
python single_csv_test.py
```

#### 3. Dataset Test

First generate the test dataset (if re-generation is needed):

```bash
# Note: Clear old files in csv_dataset/ and csv_groundtruth/ directories before execution
cd data
python csv_data_generator.py
```

Then run the dataset test:

```bash
cd example
python dataset_test.py
```

## 📊 Evaluation Metrics

`evaluate.py` provides the following evaluation metrics:

| Metric | Description |
|------|------|
| `total_cost` | Sum of distances from all points to their nearest fitted hyperplane |
| `average_distance` | Average fitting error per point |
| `total_hbar_distance` | Sum of $\hbar$ distances between fitted hyperplanes and ground truth hyperplanes |
| `ground_truth_average_distance` | Average distance of ground truth (for comparison) |

## ⚙️ Core API

### HyperplanesFitting Class

```python
from algorithm.hyperplanes_fitting import HyperplanesFitting

# Initialization
alg = HyperplanesFitting(
    dim=2,                      # Data dimension
    data=data,                  # Point cloud data (n x d numpy array)
    parallel=False,             # Whether to use parallel computing
    method="3",                 # "1": Method A, "2": Method B, "3": A+B
    whether_initial_value=True  # Whether to use automatic initial value estimation
)
# Solve
hyperplanes = alg.solve(true_num=None)  # true_num is only required when whether_initial_value=False
```

### Hyperplane and Polyhedron Classes

```python
from algorithm.initial_value import Hyperplane, Polyhedron

# Create a hyperplane:n^T x = d
hp = Hyperplane(normal=[0.6, 0.8], distance=2.5)

# Create a polyhedron (set of hyperplanes)
poly = Polyhedron(dim=2, hps=[hp1, hp2, hp3])
```

### Visualization

```python
from view.plot import ResultViewer

viewer = ResultViewer(
    dim=2, 
    data=data,
    ground_truth=ground_truth_poly,      # Ground truth (black dashed lines)
    initial_hyperplanes=result_poly,     # Fitting results (blue dashed lines)
    X_limit=[-5., 5.], 
    Y_limit=[-5., 5.]
)
viewer.draw_result()
viewer.show(save=False, show=True)
```

## 🔬 Comparative Algorithms

`compared_alg/` contains implementations of various comparative algorithms:

### Traditional Clustering Methods (`others/`)
- **RANSAC**: Random Sample Consensus
- **K-Means**: K-Means clustering + PCA fitting
- **GMM**: Gaussian Mixture Model + EM
- **DBSCAN**: Density-Based Spatial Clustering of Applications with Noise
- **Agglomerative Clustering**: Hierarchical clustering
- **OPTICS**: Ordering Points To Identify the Clustering Structure


Run comparative experiments:

```
cd compared_alg
python dataset_test.py
```


## 📚 Related Documents

| Document | Description |
|------|------|
| [README-Method-Comparison.md](README-Method-Comparison.md) | Comprehensive method comparison and evaluation guide, including evaluation results of all methods |
| [README-PARSAC-SupeRANSAC.md](./scripts-for-eval/README-PARSAC-SupeRANSAC.md) | Detailed implementation instructions for PARSAC and SupeRANSAC |

## 📝 Citation

If this project is helpful for your research, please cite the following paper:

```bibtex
@article{hyperplanes_fitting,
  author={},
  title={Fitting Unknown Number of Hyperplanes with Manifold Optimization},
  journal={},
  year={2026},
  volume={},
  pages={}
}
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🤝 Contribution

Contributions are welcome! Please submit Issues and Pull Requests.

---

**Notes**:
1. 3D data visualization requires an environment that supports interactive graphics.
2. Gurobi requires a valid Gurobi license.
3. Parallel computing is recommended for large-scale datasets (`parallel=True`)