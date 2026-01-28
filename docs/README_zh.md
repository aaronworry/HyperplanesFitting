# Hyperplanes Fitting with Manifold Optimization

[英文](../README.md) | 中文

基于流形优化的超平面拟合算法实现，对应论文《Fitting Unknown Number of Hyperplanes with Manifold Optimization》。

## 📖 项目简介

本项目实现了一种新颖的超平面拟合算法，能够自动从带噪声的点云数据中识别和拟合**未知数量**的超平面。该方法基于流形优化技术，在球面流形上进行黎曼优化，支持2D和3D数据处理。

### 核心特性

- 🔍 **自动检测超平面数量**：无需预先指定超平面数量，算法能自动发现数据中包含的超平面数目
- 🎯 **高精度拟合**：使用流形优化方法在单位球面上优化法向量，确保法向量约束满足
- ⚡ **灵活的优化方法**：支持三种优化模式 (Method A, B, A+B)
- 📊 **完整的评估体系**：提供多种评估指标来衡量拟合质量
- 🔄 **对比算法实现**：包含 RANSAC、K-Means、GMM、DBSCAN 等传统算法用于对比

## 🧮 算法原理

### 问题定义

给定一组点 $\{x_i\}_{i=1}^n \subset \mathbb{R}^d$，这些点分布在 $m$ 个未知超平面附近。每个超平面可表示为：

$$\mathcal{H}_j = \{x \in \mathbb{R}^d : n_j^T x = d_j\}$$

其中 $n_j \in \mathbb{S}^{d-1}$ 是单位法向量，$d_j \in \mathbb{R}$ 是到原点的有符号距离。

### 目标函数

最小化所有点到其最近超平面的距离之和：

$$\min_{n_j, d_j} \sum_{i=1}^n \min_j |n_j^T x_i - d_j|$$

### 算法流程

1. **初始值估计 (Algorithm 4)**：
   - 在单位球面上均匀采样候选法向量
   - 对每个候选方向，使用滑动窗口算法寻找最佳超平面
   - 基于评分函数选择初始超平面集合

2. **流形优化 (Algorithm 1 - Method A)**：
   - 使用权重矩阵 $W$ 进行点到超平面的软分配
   - 在球面流形 $\mathbb{S}^{d-1}$ 上使用最速下降法优化法向量
   - 交替更新权重和超平面参数直至收敛

3. **细化优化 (Method B)**：
   - 基于硬分配进一步优化每个超平面

### 球面流形优化

法向量约束 $\|n\|_2 = 1$ 定义了一个球面流形。优化过程包括：

- **黎曼梯度**：$\text{grad} f(x) = P_x(\nabla f(x))$，其中 $P_x$ 是到切空间的投影
- **回缩映射**：$R_x(\xi) = \frac{x + \xi}{\|x + \xi\|}$
- **线搜索**：使用 Backtracking 线搜索确定步长

## 📁 项目结构

```
hyperplanes_fitting/
├── algorithm/                    # 核心算法模块
│   ├── hyperplanes_fitting.py    # 主算法类 HyperplanesFitting
│   ├── initial_value.py          # 初始值估计算法
│   └── manifold_optimization.py  # 流形优化实现
├── manifolds/                    # 流形几何模块
│   ├── manifold.py               # 黎曼流形基类
│   └── sphere.py                 # 球面流形实现
├── solver/                       # 优化求解器
│   ├── baseSolver.py             # 求解器基类
│   ├── lineSearch.py             # 线搜索算法
│   └── steepest_descent.py       # 最速下降法
├── data/                         # 数据生成与读取
│   ├── random_data.py            # 随机数据生成器
│   ├── csv_data_generator.py     # CSV数据集生成器
│   └── read_data.py              # 数据读取工具
├── view/                         # 可视化模块
│   └── plot.py                   # 结果绘图工具
├── example/                      # 示例代码
│   ├── single_data_test.py       # 单次随机数据测试
│   ├── single_csv_test.py        # 单个CSV文件测试
│   └── dataset_test.py           # 完整数据集测试
├── compared_alg/                 # 对比算法
│   └── others/                   # 传统方法 (RANSAC, K-Means, GMM等)
├── csv_dataset/                  # 测试数据集
├── csv_groundtruth/              # 真值数据
├── evaluate.py                   # 评估指标计算
├── optimization_test/            # 附录中的优化方法测试
└── figures/                      # 输出图像目录
```

## 🚀 快速开始

### 环境要求

- Python 3.7+
- NumPy
- Matplotlib
- joblib (用于并行计算)
- scikit-learn (对比算法需要)

### 安装依赖

```bash
pip install numpy matplotlib joblib scikit-learn
```

### 运行示例

#### 1. 单次随机数据测试

使用随机生成的数据进行测试，直观展示算法效果：

```bash
cd example
python single_data_test.py
```

可在代码中修改参数：
```python
DIM = 2                    # 数据维度 (2 或 3)
HYPERPLANES_NUM = 4        # 真实超平面数量
MAX_POINT_DISTANCE = 0.1   # 点到超平面的最大噪声距离
MIN_POINT_NUM = 30         # 每个超平面上的最少点数
MAX_POINT_NUM = 30         # 每个超平面上的最多点数
METHOD = "3"               # 优化方法: "1"=A, "2"=B, "3"=A+B
INITIAL = True             # 是否使用初始值估计
```

#### 2. 单个CSV文件测试

```bash
cd example
python single_csv_test.py
```

#### 3. 完整数据集测试

首先生成测试数据集（如需重新生成）：

```bash
# 注意：执行前请清空 csv_dataset/ 和 csv_groundtruth/ 目录中的旧文件
cd data
python csv_data_generator.py
```

然后运行数据集测试：

```bash
cd example
python dataset_test.py
```

## 📊 评估指标

`evaluate.py` 提供以下评估指标：

| 指标 | 说明 |
|------|------|
| `total_cost` | 所有点到其最近拟合超平面的距离总和 |
| `average_distance` | 平均每点的拟合误差 |
| `total_hbar_distance` | 拟合超平面与真值超平面的 $\hbar$ 距离总和 |
| `ground_truth_average_distance` | 真值的平均距离（用于对比） |

## ⚙️ 核心API

### HyperplanesFitting 类

```python
from algorithm.hyperplanes_fitting import HyperplanesFitting

# 初始化
alg = HyperplanesFitting(
    dim=2,                      # 数据维度
    data=data,                  # 点云数据 (n x d numpy数组)
    parallel=False,             # 是否并行计算
    method="3",                 # "1": Method A, "2": Method B, "3": A+B
    whether_initial_value=True  # 是否使用自动初始值估计
)

# 求解
hyperplanes = alg.solve(true_num=None)  # true_num仅在whether_initial_value=False时需要
```

### 超平面和多面体类

```python
from algorithm.initial_value import Hyperplane, Polyhedron

# 创建超平面：n^T x = d
hp = Hyperplane(normal=[0.6, 0.8], distance=2.5)

# 创建多面体（超平面集合）
poly = Polyhedron(dim=2, hps=[hp1, hp2, hp3])
```

### 可视化

```python
from view.plot import ResultViewer

viewer = ResultViewer(
    dim=2, 
    data=data,
    ground_truth=ground_truth_poly,      # 真值（黑色虚线）
    initial_hyperplanes=result_poly,     # 拟合结果（蓝色虚线）
    X_limit=[-5., 5.], 
    Y_limit=[-5., 5.]
)
viewer.draw_result()
viewer.show(save=False, show=True)
```

## 🔬 对比算法

`compared_alg/` 目录包含多种对比算法实现：

### 传统聚类方法 (`others/`)
- **RANSAC**: 随机采样一致性
- **K-Means**: K均值聚类 + PCA拟合
- **GMM**: 高斯混合模型
- **DBSCAN**: 密度聚类
- **Agglomerative Clustering**: 层次聚类
- **OPTICS**: 基于密度的排序聚类


运行对比实验：

```
cd compared_alg
python dataset_test.py
```


## 📚 相关文档

| 文档 | 说明 |
|------|------|
| [README-Method-Comparison_zh.md](../docs/README-Method-Comparison_zh.md) | 完整的方法对比评估指南，包含所有方法的评估结果 |
| [README-PARSAC-SupeRANSAC_zh.md](../scripts-for-eval/README-PARSAC-SupeRANSAC_zh.md) | PARSAC 和 SupeRANSAC 的详细实现说明 |

## 📝 论文引用

如果本项目对您的研究有帮助，请引用以下论文：

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

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**注意事项**：
1. 3D数据可视化需要支持交互式图形的环境
2. Gurobi 需要有效的 Gurobi 许可证
3. 大规模数据集建议开启并行计算 (`parallel=True`)