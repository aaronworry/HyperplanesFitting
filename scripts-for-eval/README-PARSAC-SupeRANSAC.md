# PARSAC 和 SupeRANSAC 评估指南

本文档说明如何在我们的 **2D 直线拟合** 和 **3D 平面拟合** 数据集上评估 PARSAC 和 SupeRANSAC 方法，以便与论文中的其他对比方法进行公平比较。

## 📋 目录

- [⚠️ 重要声明：实现差异说明](#重要声明实现差异说明)
- [🔬 原始算法与我们实现的对比分析](#原始算法与我们实现的对比分析)
- [环境准备](#环境准备)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [🌐 3D 平面拟合评估](#3d-平面拟合评估)
- [详细参数说明](#详细参数说明)
- [输出结果说明](#输出结果说明)
- [与其他方法对比](#与其他方法对比)
- [评估指标说明](#评估指标说明)
- [🚀 如何扩展到高维超平面拟合](#如何扩展到高维超平面拟合)

---

## ⚠️ 重要声明：实现差异说明

**本项目中的 "PARSAC" 和 "SupeRANSAC" 实现是针对 2D 直线拟合任务的简化适配版本，而非原始算法的完整复现。**

### 为什么不能直接使用原始算法？

| 原始算法 | 设计目的 | 我们的任务 | 差异 |
|----------|----------|------------|------|
| **PARSAC** | 消失点检测、基础矩阵/单应矩阵估计 | 2D点云多直线拟合 | 完全不同的问题定义 |
| **SupeRANSAC** | 图像特征匹配的鲁棒估计 | 2D点云多直线拟合 | 不支持通用直线拟合 |

### 实现策略

我们选择了以下策略来在公平条件下进行对比：

1. **PARSAC**: 借鉴其核心思想（并行假设生成、软内点加权、贪心模型选择），实现了一个几何版本
2. **SupeRANSAC**: 借鉴其 RANSAC 变体（PROSAC采样、MSAC/MAGSAC评分），实现了 Sequential RANSAC

**置信度评估: 60-70%** - 这些实现保留了原始方法的核心算法思想，但移除了依赖于深度学习的组件。

---

## 🔬 原始算法与我们实现的对比分析

### PARSAC 对比分析

#### 原始 PARSAC 算法流程

```
输入: 特征点/线段 + 图像特征
    ↓
┌─────────────────────────────────────────┐
│ 1. 神经网络 (CNNet)                      │
│    - 输入: 点特征 (N × input_dim)        │
│    - 输出: log_inlier_weights (采样权重) │
│           log_sample_weights (实例权重)  │
│    - 5层 ResNet 风格 1D 卷积              │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 2. 并行采样 (sampling.py)                │
│    - 使用学习的权重进行重要性采样          │
│    - 生成 M×S×K 个最小集                  │
│    - minimal_solver 计算假设              │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 3. 残差计算 + 软内点计数                  │
│    - soft_inlier: sigmoid(β×(τ-d)/τ)    │
│    - 加权内点比例作为假设得分              │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 4. 假设选择 + 聚类 (postprocessing.py)   │
│    - 贪心选择：每次选增益最大的假设        │
│    - 点分配到最近假设                     │
└─────────────────────────────────────────┘
    ↓
输出: 多个模型实例 + 点标签
```

#### 我们的 SimplePARSACLineFitter 实现

```
输入: 2D 点云 (N × 2)
    ↓
┌─────────────────────────────────────────┐
│ 1. 随机采样 (替代神经网络)                │
│    - 均匀随机采样点对                     │
│    - 无学习权重                          │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 2. 假设生成                              │
│    - 从点对计算直线: n = (p2-p1)⊥        │
│    - 生成 H 个候选直线                   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 3. 软内点计数 ✓ (与原始相同)             │
│    - soft_inlier: 1/(1+exp(β×(d-τ)))    │
│    - 计算每个假设的软内点总分             │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 4. 贪心选择 ✓ (与原始类似)               │
│    - 选择得分最高的假设                   │
│    - 惩罚与已选假设相似的候选             │
│    - 标记已分配点                        │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 5. 迭代精化                              │
│    - 聚类分配 + SVD 重新拟合              │
└─────────────────────────────────────────┘
    ↓
输出: 多条直线 + 点标签
```

#### 关键差异总结

| 组件 | 原始 PARSAC | 我们的实现 | 保真度 |
|------|------------|-----------|-------|
| **采样策略** | 神经网络学习权重 | 均匀随机 | ❌ 50% |
| **假设生成** | minimal_solver | 点对拟合 | ✓ 90% |
| **软内点函数** | sigmoid(β(τ-d)/τ) | 1/(1+exp(β(d-τ))) | ✓ 95% |
| **假设选择** | 贪心+去重 | 贪心+相似惩罚 | ✓ 85% |
| **精化** | SVD (仅VP) | SVD 迭代 | ✓ 90% |
| **支持问题** | VP/F/H矩阵 | 2D直线 | ⚡ 适配 |

#### 🔍 未知超平面数量模式的实现说明

**原始 PARSAC 的"未知模型数量"特性：**

原始 PARSAC 论文确实支持"未知模型数量"的场景。其核心机制是：
1. 神经网络预测每个点的 **instance assignment probabilities** (实例分配概率)
2. 通过 softmax 归一化，每个点可以属于 K 个可能的模型实例
3. 后处理阶段，通过阈值过滤低置信度的模型

**我们的"未知模式"实现 (`auto_detect=True`)：**

由于我们移除了神经网络组件，我们实现了一个基于 **贪心得分衰减** 的自动检测机制：

```python
def _auto_detect_num_models(self, points, hypotheses, inlier_scores,
                            max_models=10, min_score_ratio=0.1, 
                            min_inliers_ratio=0.05):
    """
    自动检测策略：贪心选择假设，直到满足终止条件
    """
    N = len(points)
    selected_indices = []
    assigned_points = np.zeros(N, dtype=bool)
    first_score = None
    
    for _ in range(max_models):
        # 计算剩余未分配点的得分
        remaining_scores = np.sum(inlier_scores[~assigned_points, :], axis=0)
        remaining_scores[selected_indices] = -inf
        
        best_idx = np.argmax(remaining_scores)
        best_score = remaining_scores[best_idx]
        
        # 记录第一个模型得分作为参考
        if first_score is None:
            first_score = best_score
        
        # 终止条件 1: 新模型得分相对于第一个太低
        if best_score < first_score * min_score_ratio:
            break
        
        # 终止条件 2: 新模型的内点数太少
        hard_inliers = inlier_scores[:, best_idx] > 0.5
        num_new_inliers = np.sum(hard_inliers & ~assigned_points)
        if num_new_inliers < N * min_inliers_ratio:
            break
        
        selected_indices.append(best_idx)
        assigned_points |= hard_inliers
    
    return len(selected_indices)
```

**与原始算法的对比：**

| 特性 | 原始 PARSAC | 我们的实现 |
|------|------------|-----------|
| **模型数量检测** | 神经网络输出 + softmax | 贪心得分衰减 |
| **终止条件** | 置信度阈值 | 得分比例 + 内点比例 |
| **理论依据** | 学习的先验分布 | 启发式规则 |
| **适用场景** | 已训练的特定任务 | 通用几何拟合 |
| **参数敏感性** | 低 (神经网络学习) | 中等 (需要调参) |

**参数说明：**
- `max_models=10`: 最多检测的模型数量上限
- `min_score_ratio=0.1`: 新模型得分至少是第一个模型的 10%
- `min_inliers_ratio=0.05`: 新模型至少覆盖 5% 的剩余点

**注意：** 这种启发式方法在简单场景下效果良好，但在复杂场景（如模型大小差异大、噪声高）下可能不如原始的神经网络方法稳健。

---

### SupeRANSAC 对比分析

#### 原始 SupeRANSAC 算法流程

```
输入: 点对应关系 + 图像尺寸
    ↓
┌─────────────────────────────────────────┐
│ RANSAC 主循环                            │
│ while (iter < max_iter && iter < adaptive_iter):
│   │
│   ├─ 采样器 (SamplerType)               │
│   │  - Uniform / PROSAC / NAPSAC        │
│   │  - ImportanceSampler / ARSampler    │
│   │
│   ├─ 模型估计 (estimator)               │
│   │  - Homography / Fundamental / Essential
│   │  - RigidTransform / AbsolutePose    │
│   │
│   ├─ 评分 (ScoringType)                 │
│   │  - RANSAC / MSAC / MAGSAC / ACRANSAC│
│   │
│   ├─ 局部优化 (LocalOptimizationType)   │
│   │  - LSQ / IteratedLSQ                │
│   │  - NestedRANSAC / GCRANSAC          │
│   │
│   └─ 终止准则 (自适应迭代数)             │
└─────────────────────────────────────────┘
    ↓
输出: 单个最佳模型 + 内点掩码
```

**关键特点**:
- C++ 核心实现，Python 绑定
- 专为**单模型**估计设计
- 支持多种采样、评分、优化策略组合
- 内置空间分区加速

#### 我们的 SequentialRANSAC2DLine 实现

```
输入: 2D 点云 (N × 2)
    ↓
┌─────────────────────────────────────────┐
│ Sequential 多模型循环                    │
│ for model_idx in range(max_models):     │
│   │
│   ├─ RANSAC 单模型拟合                   │
│   │  ├─ PROSAC 采样 ✓                   │
│   │  ├─ 点对拟合直线                    │
│   │  ├─ MSAC/MAGSAC 评分 ✓              │
│   │  └─ 自适应迭代数 ✓                  │
│   │
│   ├─ LSQ 精化 ✓                         │
│   │  └─ SVD 使用所有内点重拟合          │
│   │
│   └─ 移除内点，继续下一轮               │
└─────────────────────────────────────────┘
    ↓
输出: 多条直线 + 点标签
```

#### 关键差异总结

| 组件 | 原始 SupeRANSAC | 我们的实现 | 保真度 |
|------|----------------|-----------|-------|
| **设计目标** | 单模型估计 | 多模型拟合 | ⚡ 适配 |
| **采样器** | 多种可选 | PROSAC | ✓ 80% |
| **评分** | RANSAC/MSAC/MAGSAC | MSAC/MAGSAC | ✓ 90% |
| **局部优化** | LSQ/GCRANSAC/Nested | LSQ (SVD) | ✓ 70% |
| **支持模型** | H/F/E/Rigid/Pose | 2D直线 | ⚡ 适配 |
| **实现语言** | C++ + pybind11 | 纯 Python | ⚠️ 性能差异 |
| **多模型策略** | 无 (单模型) | Sequential | ⚡ 扩展 |

---

### 🌐 3D 平面拟合算法实现

我们将 2D 算法扩展到 3D 平面拟合，主要修改了假设生成和残差计算部分。

#### SimplePARSACPlaneFitter (3D)

```
输入: 3D 点云 (N × 3)
    ↓
┌─────────────────────────────────────────┐
│ 1. 随机三点采样                          │
│    - 均匀随机采样 3 个点                  │
│    - 与 2D 的 2 点采样对应                │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 2. 平面假设生成                          │
│    - 计算两向量叉积得法向量:              │
│      v1 = p2 - p1, v2 = p3 - p1          │
│      n = v1 × v2 / ||v1 × v2||           │
│    - 计算距离: d = n · p1                │
│    - 生成 H 个候选平面                   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 3. 软内点计数 (与 2D 相同)               │
│    - 点到平面距离: |n·x - d|             │
│    - soft_inlier: 1/(1+exp(β×(r-τ)))     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 4. 贪心选择 + SVD 精化                   │
│    - 选择得分最高的平面                   │
│    - 使用 SVD 重新拟合内点                │
└─────────────────────────────────────────┘
    ↓
输出: 多个平面 (n1, n2, n3, d) + 点标签
```

**关键代码片段**:
```python
# 3D 平面假设生成
p1, p2, p3 = points[idx]  # 采样 3 个点
v1 = p2 - p1
v2 = p3 - p1
normal = np.cross(v1, v2)  # 叉积得到法向量
normal = normal / np.linalg.norm(normal)
d = np.dot(normal, p1)
```

#### SequentialRANSAC3DPlane

```
输入: 3D 点云 (N × 3)
    ↓
┌─────────────────────────────────────────┐
│ Sequential 多模型循环                    │
│ for plane_idx in range(max_planes):     │
│   │
│   ├─ RANSAC 单平面拟合                   │
│   │  ├─ 3 点采样 (vs 2D 的 2 点)         │
│   │  ├─ 叉积计算法向量                   │
│   │  ├─ MSAC/MAGSAC 评分                 │
│   │  └─ 自适应迭代数                     │
│   │
│   ├─ SVD 精化                            │
│   │  └─ 使用所有内点重新拟合平面         │
│   │
│   └─ 移除内点，继续下一轮               │
└─────────────────────────────────────────┘
    ↓
输出: 多个平面 + 点标签
```

#### 2D vs 3D 实现差异总结

| 组件 | 2D 直线 | 3D 平面 | 修改 |
|------|---------|---------|------|
| **最小采样** | 2 点 | 3 点 | 增加 1 点 |
| **假设生成** | 垂直向量 | 叉积 | 算法不同 |
| **模型参数** | (a, b, c) 或 (n1, n2, d) | (n1, n2, n3, d) | 增加 1 维 |
| **残差计算** | \|ax + by + c\| / √(a²+b²) | \|n·x - d\| | 公式类似 |
| **SVD 精化** | 2D 协方差 | 3D 协方差 | 维度扩展 |

---

## 🔧 环境准备

### 1. 官方仓库参考

官方仓库代码位于 `potential-repositories/` 目录下，仅供参考，不需要修改：

- **PARSAC**: `potential-repositories/parsac/`
- **SupeRANSAC**: `potential-repositories/superansac/`

### 2. 评估环境配置

我们的评估脚本已简化了对原始仓库的依赖，只需基本的 Python 环境即可运行：

```bash
# 创建或激活环境
conda create -n parsac python=3.10
conda activate parsac

# 安装依赖
pip install numpy pandas scipy matplotlib scikit-learn
```

## 📁 目录结构

本项目新增的评估相关文件集中在以下三个目录：

```
hyperplanes_fitting-main/
├── scripts-for-eval/               # 评估脚本目录
│   ├── evaluate_utils.py           # 统一评估工具（与论文一致的指标）
│   ├── compare_all_methods.py      # 2D 多方法对比脚本
│   ├── compare_all_methods_3d.py   # 3D 多方法对比脚本
│   ├── data_utils.py               # 数据读取工具
│   ├── metrics.py                  # 评估指标模块
│   │
│   ├── ours/                       # Ours (流形优化) 评估模块
│   │   ├── __init__.py
│   │   └── evaluate_ours.py        # 2D/3D: Ours 评估脚本
│   │
│   ├── parsac/                     # PARSAC 评估模块
│   │   ├── __init__.py
│   │   ├── line_fitter.py          # 2D: SimplePARSACLineFitter 实现
│   │   ├── evaluate_parsac.py      # 2D: PARSAC 评估脚本
│   │   ├── plane_fitter_3d.py      # 3D: SimplePARSACPlaneFitter 实现
│   │   └── evaluate_parsac_3d.py   # 3D: PARSAC 评估脚本
│   │
│   ├── superansac/                 # SupeRANSAC 评估模块
│   │   ├── __init__.py
│   │   ├── sequential_ransac.py    # 2D: Sequential RANSAC 实现
│   │   ├── evaluate_superansac.py  # 2D: SupeRANSAC 评估脚本
│   │   ├── sequential_ransac_3d.py # 3D: Sequential RANSAC 3D 实现
│   │   └── evaluate_superansac_3d.py # 3D: SupeRANSAC 评估脚本
│   │
│   └── compared_alg_3d/            # 3D 对比算法模块
│       ├── __init__.py
│       ├── RANSAC_3D.py            # 3D RANSAC 平面拟合
│       ├── K_Means_3D.py           # 3D K-Means + SVD 拟合
│       └── GMM_3D.py               # 3D GMM + SVD 拟合
│
├── results/                        # 结果输出目录
│   ├── 2d/                         # 2D 直线拟合结果
│   │   ├── ours/                   # Ours 2D 结果
│   │   ├── parsac/                 # PARSAC 2D 结果
│   │   ├── superansac/             # SupeRANSAC 2D 结果
│   │   └── figures/                # 2D 对比图表
│   │
│   └── 3d/                         # 3D 平面拟合结果
│       ├── ours/                   # Ours 3D 结果
│       ├── parsac/                 # PARSAC 3D 结果
│       ├── superansac/             # SupeRANSAC 3D 结果
│       ├── ransac/                 # RANSAC 3D 结果
│       ├── kmeans/                 # K-Means 3D 结果
│       ├── gmm/                    # GMM 3D 结果
│       └── figures/                # 3D 对比图表
│
├── visualization/                  # 可视化脚本目录
│   ├── plot_results.py             # 2D 结果可视化工具
│   └── visualize_3d.py             # 3D 可视化工具 (plotly)
│
├── csv_dataset/                    # 2D 数据集
├── csv_groundtruth/                # 2D 真值
├── csv_dataset_3d/                 # 3D 数据集
├── csv_groundtruth_3d/             # 3D 真值
│
└── README-PARSAC-SupeRANSAC.md     # 本文档
```

**注意**: 原有项目结构（`algorithm/`, `compared_alg/`, `data/`, `csv_dataset/` 等）保持不变。

## 🚀 快速开始

### 1. 运行 PARSAC 评估

```bash
cd /path/to/hyperplanes_fitting-main

# 使用已知模型数量（推荐，与论文设置一致）
python scripts-for-eval/parsac/evaluate_parsac.py --known_count

# 自动检测模型数量
python scripts-for-eval/parsac/evaluate_parsac.py
```

### 2. 运行 SupeRANSAC 评估

```bash
# 使用已知模型数量
python scripts-for-eval/superansac/evaluate_superansac.py --known_count

# 自动检测模型数量
python scripts-for-eval/superansac/evaluate_superansac.py
```

### 3. 运行多方法对比

```bash
# 对比所有方法并生成图表
python scripts-for-eval/compare_all_methods.py \
    --methods parsac superansac ransac kmeans gmm \
    --run_eval --known_count
```

---

## 🌐 3D 平面拟合评估

### 概述

除了 2D 直线拟合，我们还提供了 **3D 平面拟合** 的评估脚本，用于在 $\mathbb{R}^3$ 空间中拟合多个平面。

### 1. 生成 3D 测试数据

```bash
cd /path/to/hyperplanes_fitting-main

# 生成 3D 数据 (4 个平面, 每平面 100 点, 噪声 0.1)
python data/csv_data_generator.py \
    --dim 3 \
    --data_dir csv_dataset_3d \
    --gt_dir csv_groundtruth_3d \
    --num_samples 20 \
    --num_hyperplanes 4 \
    --points_per_hyperplane 100 \
    --noise 0.1
```

**3D 数据格式**:
- 数据文件 (`csv_dataset_3d/X.csv`): `x,y,z` 三列坐标
- 真值文件 (`csv_groundtruth_3d/X.csv`): `n1,n2,n3,d,totaldistance` 五列

### 2. 运行 PARSAC 3D 评估

```bash
# 使用已知模型数量（推荐）
python scripts-for-eval/parsac/evaluate_parsac_3d.py --known_count

# 自动检测模型数量
python scripts-for-eval/parsac/evaluate_parsac_3d.py

# 自定义参数
python scripts-for-eval/parsac/evaluate_parsac_3d.py \
    --data_dir csv_dataset_3d \
    --gt_dir csv_groundtruth_3d \
    --output_dir results/3d/parsac \
    --num_samples 20 \
    --num_hypotheses 500 \
    --inlier_threshold 0.2 \
    --known_count
```

### 3. 运行 SupeRANSAC 3D 评估

```bash
# 使用已知模型数量
python scripts-for-eval/superansac/evaluate_superansac_3d.py --known_count

# 自动检测模型数量
python scripts-for-eval/superansac/evaluate_superansac_3d.py

# 自定义参数
python scripts-for-eval/superansac/evaluate_superansac_3d.py \
    --data_dir csv_dataset_3d \
    --gt_dir csv_groundtruth_3d \
    --output_dir results/3d/superansac \
    --num_samples 20 \
    --max_iterations 1000 \
    --inlier_threshold 0.3 \
    --known_count
```

### 4. 3D 可视化

```bash
# 可视化单个样本的拟合结果
python visualization/visualize_3d.py \
    --data_file csv_dataset_3d/0.csv \
    --gt_file csv_groundtruth_3d/0.csv \
    --output_dir results/3d/figures \
    --method parsac  # 或 superansac

# 交互式 HTML 可视化（推荐）
python visualization/visualize_3d.py \
    --data_file csv_dataset_3d/0.csv \
    --gt_file csv_groundtruth_3d/0.csv \
    --format html
```

### 5. 3D 评估参数说明

#### PARSAC 3D (`evaluate_parsac_3d.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `csv_dataset_3d/` | 3D 数据目录 |
| `--gt_dir` | `csv_groundtruth_3d/` | 3D 真值目录 |
| `--output_dir` | `results/3d/parsac/` | 输出目录 |
| `--num_hypotheses` | 500 | 生成的候选平面数量 |
| `--inlier_threshold` | 0.2 | 内点判定阈值 |
| `--known_count` | False | 使用已知的平面数量 |

#### SupeRANSAC 3D (`evaluate_superansac_3d.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `csv_dataset_3d/` | 3D 数据目录 |
| `--gt_dir` | `csv_groundtruth_3d/` | 3D 真值目录 |
| `--output_dir` | `results/3d/superansac/` | 输出目录 |
| `--max_iterations` | 1000 | 单次 RANSAC 最大迭代数 |
| `--inlier_threshold` | 0.3 | 内点判定阈值 |
| `--min_inliers` | 15 | 一个平面需要的最小内点数 |
| `--known_count` | False | 使用已知的平面数量 |

### 6. 3D 算法实现说明

#### SimplePARSACPlaneFitter (3D)

**与 2D 版本的主要区别**:
- **假设生成**: 从 3 个点计算平面法向量（叉积），而非 2 个点计算直线
- **维度**: 假设格式 `(n1, n2, n3, d)` 而非 `(a, b, c)`

```python
# 3D 平面假设生成
p1, p2, p3 = points[idx]  # 采样 3 个点
v1, v2 = p2 - p1, p3 - p1
normal = np.cross(v1, v2)  # 叉积得到法向量
normal = normal / np.linalg.norm(normal)
d = np.dot(normal, p1)
```

#### SequentialRANSAC3DPlane

**与 2D 版本的主要区别**:
- **最小采样**: 3 个点拟合平面，而非 2 个点拟合直线
- **距离计算**: 点到平面的垂直距离

```python
# 3D 点到平面距离
distances = np.abs(np.dot(points, normal) - d)
```

---

## 📖 详细参数说明

### PARSAC 评估脚本 (`evaluate_parsac.py`)

```bash
python scripts-for-eval/parsac/evaluate_parsac.py [OPTIONS]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_dir` | str | `csv_dataset/` | 数据目录路径 |
| `--gt_dir` | str | `csv_groundtruth/` | 真值目录路径 |
| `--output_dir` | str | `results/parsac/` | 输出目录路径 |
| `--num_samples` | int | 20 | 评估的样本数量 |
| `--known_count` | flag | False | 使用已知的模型数量（Ground Truth 中的直线数） |
| `--num_hypotheses` | int | 500 | 生成的候选假设数量，越大越准确但更慢 |
| `--inlier_threshold` | float | 0.15 | 内点判定阈值，点到直线距离小于此值视为内点 |
| `--num_iterations` | int | 3 | 多次迭代取最优结果的迭代次数 |

**示例**:
```bash
# 高精度模式（更多假设）
python scripts-for-eval/parsac/evaluate_parsac.py \
    --known_count \
    --num_hypotheses 1000 \
    --inlier_threshold 0.1

# 快速模式
python scripts-for-eval/parsac/evaluate_parsac.py \
    --num_hypotheses 200 \
    --num_iterations 1
```

### SupeRANSAC 评估脚本 (`evaluate_superansac.py`)

```bash
python scripts-for-eval/superansac/evaluate_superansac.py [OPTIONS]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_dir` | str | `csv_dataset/` | 数据目录路径 |
| `--gt_dir` | str | `csv_groundtruth/` | 真值目录路径 |
| `--output_dir` | str | `results/superansac/` | 输出目录路径 |
| `--num_samples` | int | 20 | 评估的样本数量 |
| `--known_count` | flag | False | 使用已知的模型数量 |
| `--max_iterations` | int | 1000 | 单次 RANSAC 的最大迭代次数 |
| `--inlier_threshold` | float | 0.3 | 内点判定阈值 |
| `--min_inliers` | int | 10 | 一条直线需要的最小内点数 |

**示例**:
```bash
# 严格内点阈值
python scripts-for-eval/superansac/evaluate_superansac.py \
    --known_count \
    --inlier_threshold 0.2 \
    --min_inliers 15

# 宽松设置（适合噪声大的数据）
python scripts-for-eval/superansac/evaluate_superansac.py \
    --inlier_threshold 0.5 \
    --min_inliers 8
```

### 多方法对比脚本 (`compare_all_methods.py`)

```bash
python scripts-for-eval/compare_all_methods.py [OPTIONS]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--methods` | list | `['parsac', 'superansac', 'ransac', 'kmeans', 'gmm']` | 要对比的方法列表 |
| `--results_dir` | str | `results/` | 已有结果的读取目录 |
| `--output_dir` | str | `results/figures/` | 图表输出目录 |
| `--run_eval` | flag | False | 如果结果不存在，自动运行评估 |
| `--known_count` | flag | False | 评估时使用已知模型数量 |

**支持的方法**:
- `parsac`: PARSAC 算法
- `superansac`: Sequential RANSAC (模拟 SupeRANSAC)
- `ransac`: sklearn RANSAC
- `kmeans`: K-Means 聚类
- `gmm`: 高斯混合模型
- `ours`: 我们的流形优化方法

**示例**:
```bash
# 只对比 PARSAC 和 SupeRANSAC
python scripts-for-eval/compare_all_methods.py \
    --methods parsac superansac \
    --run_eval

# 对比所有方法
python scripts-for-eval/compare_all_methods.py \
    --methods parsac superansac ransac kmeans gmm ours \
    --run_eval --known_count
```

## 📊 输出结果说明

### 1. CSV 结果文件

`results/parsac/parsac_results.csv` 格式：

| 列名 | 说明 |
|------|------|
| `sample_id` | 样本编号 (0-19) |
| `total_cost` | 所有点到最近超平面的距离总和 (TC) |
| `cost_ratio` | total_cost / gt_total_cost |
| `average_distance` | total_cost / 点数 |
| `total_hbar_distance` | H-bar 距离总和 (HE) |
| `model_count` | 检测到的模型数量 (HN) |
| `gt_model_count` | 真值模型数量 |
| `model_count_error` | \|检测数 - 真值数\| |
| `runtime` | 运行时间 (秒) |

### 2. 汇总统计文件

`results/parsac/parsac_summary.txt` 包含各指标的均值和标准差。

### 3. 对比图表

| 文件 | 说明 |
|------|------|
| `multi_panel_comparison.png` | 多联对比图，展示所有指标 |
| `boxplot_comparison.png` | 箱线图，展示分布差异 |
| `bar_*.png` | 各指标的柱状图 |
| `comparison_table.txt` | 文本格式对比表格（含 LaTeX 格式） |

## 📐 评估指标说明

本评估框架使用与论文完全一致的指标定义：

### 1. HN (Hyperplane Number) - 模型数量

检测到的超平面数量。

### 2. TC (Total Cost) - 总代价

$$\text{TC} = \sum_{i=1}^{n} \min_j |n_j^T x_i - d_j|$$

所有点到其最近拟合超平面的距离之和。

### 3. HE (H-bar Error) - H-bar 误差

$$\bar{h}_j = d_j \cdot n_j$$

$$\text{HE} = \sum_{j=1}^{m} \min_k \|\bar{h}_j^{\text{result}} - \bar{h}_k^{\text{GT}}\|$$

拟合超平面与真值超平面在 H-bar 空间的距离总和。

### 附加指标

- **Cost Ratio**: TC / GT_TC，理想值为 1.0
- **Average Distance**: TC / n，平均每点的拟合误差
- **Model Count Error**: |HN - GT_HN|

---

## 🔬 与其他方法对比

本节详细说明我们对比的各种方法的核心思路和实现差异。

### 对比方法概览

| 方法 | 类型 | 核心思想 | 是否需要已知模型数 |
|------|------|----------|-------------------|
| **Ours** | 流形优化 | 球面流形上黎曼梯度下降 | ❌ 可自动检测 |
| **PARSAC** | RANSAC变体 | 并行假设生成+软内点+贪心选择 | ✓ 推荐使用 |
| **SupeRANSAC** | Sequential RANSAC | PROSAC采样+MSAC评分+逐步剥离 | ✓ 推荐使用 |
| **RANSAC** | 随机一致采样 | 随机采样+投票+内点剥离 | ✓ 必需 |
| **K-Means** | 聚类+拟合 | 点聚类后对每簇PCA/SVD拟合 | ✓ 必需 |
| **GMM** | 概率聚类+拟合 | 软聚类后加权SVD拟合 | ✓ 必需 |

### 各方法详细说明

#### 1. Ours (流形优化方法)

**核心思想**: 将超平面法向量约束在单位球面 $\mathbb{S}^{d-1}$ 上，使用黎曼优化直接求解。

**算法流程**:
1. **初始值估计**: 在球面上均匀采样候选方向，滑动窗口筛选
2. **软分配**: 使用权重矩阵 $W_{ij}$ 表示点 $i$ 属于超平面 $j$ 的概率
3. **流形优化**: 在球面上使用最速下降法优化法向量
4. **硬分配精化**: 最终分配点到最近超平面并微调

**优势**:
- 无需预知超平面数量
- 理论保证收敛
- 法向量约束自动满足

**代码位置**: `algorithm/hyperplanes_fitting.py`

---

#### 2. PARSAC (Parallel RANSAC)

**核心思想**: 并行生成大量假设，使用神经网络（我们用随机采样替代）学习采样权重，贪心选择最佳假设组合。

**原始算法特点**:
- 神经网络预测采样权重和实例权重
- 可微分的软内点函数
- 支持消失点、基础矩阵等多种几何模型

**我们的适配**:
- 用均匀随机采样替代神经网络
- 保留软内点计数: $s(r) = \frac{1}{1 + e^{\beta(r - \tau)}}$
- 保留贪心选择策略

**代码位置**: 
- 2D: `scripts-for-eval/parsac/line_fitter.py`
- 3D: `scripts-for-eval/parsac/plane_fitter_3d.py`

---

#### 3. SupeRANSAC (Sequential RANSAC)

**核心思想**: 借鉴 SupeRANSAC 的先进采样和评分策略，以 Sequential 方式拟合多个模型。

**关键技术**:
- **PROSAC 采样**: 优先采样质量更高的点（按内点概率排序）
- **MSAC 评分**: 截断的 M-estimator，$\rho(r) = \min(r^2, \tau^2)$
- **MAGSAC 评分**: 边际化阈值的自适应评分
- **自适应迭代**: 根据内点比例动态调整迭代次数

**Sequential 策略**:
1. 用 RANSAC 找到一个模型
2. 移除该模型的内点
3. 对剩余点重复步骤 1-2

**代码位置**: 
- 2D: `scripts-for-eval/superansac/sequential_ransac.py`
- 3D: `scripts-for-eval/superansac/sequential_ransac_3d.py`

---

#### 4. RANSAC (Random Sample Consensus)

**核心思想**: 随机采样最小点集，拟合模型，统计内点，迭代寻找最佳模型。

**算法流程**:
```
for iteration in range(max_iters):
    1. 随机采样 k 个点 (2D直线: k=2, 3D平面: k=3)
    2. 拟合模型参数
    3. 计算所有点到模型距离
    4. 统计内点数量 (距离 < 阈值)
    5. 如果内点数最多，更新最佳模型
```

**多模型策略**: Sequential (逐步剥离内点)

**优势**: 简单、鲁棒
**劣势**: 需要精心调节阈值，多模型时效率低

**代码位置**: 
- 2D: `compared_alg/others/RANSAC.py`
- 3D: `scripts-for-eval/compared_alg_3d/RANSAC_3D.py`

---

#### 5. K-Means + PCA/SVD

**核心思想**: 先用 K-Means 将点聚类，再对每个簇使用 PCA/SVD 拟合超平面。

**算法流程**:
```
1. K-Means 聚类:
   - 初始化 K 个聚类中心
   - 迭代分配点到最近中心
   - 更新中心为簇均值
   
2. 对每个簇拟合超平面:
   - 计算簇中心 (质心)
   - 中心化数据
   - SVD 分解，最小奇异值对应的向量即为法向量
   - d = n · 质心
```

**优势**: 简单高效
**劣势**: 
- K-Means 对初始化敏感
- 假设簇是球形分布，不适合长条形分布

**代码位置**: 
- 2D: `compared_alg/others/K_Means.py`
- 3D: `scripts-for-eval/compared_alg_3d/K_Means_3D.py`

---

#### 6. GMM (Gaussian Mixture Model)

**核心思想**: 用高斯混合模型进行软聚类，每个点以一定概率属于各个高斯分量，再对每个分量使用加权 SVD 拟合超平面。

**算法流程**:
```
1. 初始化 GMM 参数 (均值, 协方差, 混合系数)

2. EM 算法迭代:
   E 步: 计算后验概率 γ_ik = P(z_i = k | x_i)
   M 步: 更新参数
         μ_k = Σ γ_ik x_i / Σ γ_ik
         Σ_k = Σ γ_ik (x_i - μ_k)(x_i - μ_k)^T / Σ γ_ik
         π_k = Σ γ_ik / N

3. 对每个分量拟合超平面:
   - 加权 SVD: 用 γ_ik 作为权重
   - 最小特征值对应的特征向量即为法向量
```

**优势**: 
- 软聚类，更平滑
- 可以处理重叠区域

**劣势**: 
- 计算量大
- 对初始化敏感
- 假设高斯分布

**代码位置**: 
- 2D: `compared_alg/others/GMM.py`
- 3D: `scripts-for-eval/compared_alg_3d/GMM_3D.py`

---

### 方法性能对比 (示例结果)

#### 2D 直线拟合

| 方法 | Total Cost | Cost Ratio | H-bar Dist | Runtime |
|------|------------|------------|------------|---------|
| **Ours** | 5.58 ± 0.30 | 0.94 ± 0.03 | 0.07 ± 0.02 | 0.24s |
| PARSAC | 5.57 ± 0.30 | 0.93 ± 0.03 | 0.07 ± 0.03 | 0.02s |
| SupeRANSAC | 5.75 ± 0.37 | 0.96 ± 0.04 | 0.09 ± 0.04 | 0.02s |

#### 3D 平面拟合

| 方法 | Total Cost | Cost Ratio | H-bar Dist | Runtime |
|------|------------|------------|------------|---------|
| **Ours** | 5.51 ± 0.30 | 0.92 ± 0.03 | 0.09 ± 0.02 | 2.24s |
| PARSAC | 5.50 ± 0.30 | 0.92 ± 0.03 | 0.09 ± 0.02 | 0.02s |
| SupeRANSAC | 5.70 ± 0.36 | 0.95 ± 0.04 | 0.12 ± 0.03 | 0.04s |
| RANSAC | 6.20 ± 0.73 | 1.03 ± 0.10 | 0.16 ± 0.05 | 0.09s |
| K-Means | 68.4 ± 23.9 | 11.5 ± 4.2 | 7.0 ± 2.7 | 0.001s |
| GMM | 26.8 ± 23.0 | 4.5 ± 3.9 | 2.1 ± 2.2 | 0.02s |

**注**: 以上结果为 20 个样本的均值 ± 标准差，使用已知模型数量。

---

## 🔍 故障排除

### 常见问题

1. **ModuleNotFoundError: No module named 'data'**
   ```bash
   # 确保在项目根目录运行
   cd /path/to/hyperplanes_fitting-main
   ```

2. **pandas 未安装**
   ```bash
   pip install pandas scipy matplotlib
   ```

3. **结果文件不存在**
   ```bash
   # 使用 --run_eval 参数自动运行评估
   python scripts-for-eval/compare_all_methods.py --run_eval
   ```

## 📝 引用

如果本评估框架对您的研究有帮助，请引用：

```bibtex
@article{hyperplanes_fitting,
  title={Fitting Unknown Number of Hyperplanes with Manifold Optimization},
  author={},
  journal={},
  year={2026}
}
```

---

## 🚀 如何扩展到高维超平面拟合

### 当前实现的维度

本评估框架目前针对 **2D 直线拟合** 设计，即在 $\mathbb{R}^2$ 空间中拟合多条 1 维超平面（直线）。

### 扩展到 n 维超平面

若要将实现扩展到 $\mathbb{R}^n$ 空间的超平面拟合，需要修改以下关键组件：

#### 1. 假设生成 (Hypothesis Generation)

**2D 实现** (从 2 个点生成直线):
```python
# 2 points → 1 line in R^2
direction = p2 - p1
normal = [-direction[1], direction[0]]  # 垂直向量
d = dot(normal, p1)
```

**n 维扩展** (从 n 个点生成超平面):
```python
# n points → 1 hyperplane in R^n
points = sample_n_points(data, n)  # 采样 n 个点
centered = points - mean(points)
U, S, Vt = svd(centered)
normal = Vt[-1]  # 最小奇异值对应的右奇异向量
d = dot(normal, mean(points))
```

#### 2. 残差计算

**公式保持不变**:
$$\text{residual} = |n^T x - d|$$

代码修改：
```python
# 2D
residual = abs(n[0]*x + n[1]*y - d)

# nD (向量化)
residual = abs(np.dot(points, normal) - d)
```

#### 3. 内点函数

**无需修改**，软内点函数与维度无关：
$$s(r) = \frac{1}{1 + e^{\beta(r - \tau)}}$$

#### 4. 模型数量估计

高维情况下模型数量估计更具挑战性：
- 可使用信息准则 (BIC/AIC)
- 或保持使用已知数量 (`--known_count`)

### 示例：扩展 SimplePARSACLineFitter 到 3D

```python
class SimplePARSACPlaneFitter:
    def _generate_hypotheses(self, points):
        """在 R^3 中生成平面假设"""
        N, dim = points.shape
        assert dim == 3
        hypotheses = []
        
        for _ in range(self.num_hypotheses):
            # 采样 3 个点
            idx = np.random.choice(N, 3, replace=False)
            p1, p2, p3 = points[idx]
            
            # 计算法向量
            v1, v2 = p2 - p1, p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal = normal / norm
            d = np.dot(normal, p1)
            
            hypotheses.append([*normal, d])  # (n1, n2, n3, d)
        
        return np.array(hypotheses)
```

---

## 🎯 公平性讨论

### 为什么这种对比是有意义的？

1. **算法思想保留**: 我们的实现保留了 PARSAC 和 SupeRANSAC 的核心算法思想
   - PARSAC: 并行假设生成 + 软内点 + 贪心选择
   - SupeRANSAC: PROSAC 采样 + MAGSAC 评分 + LSQ 精化

2. **问题适配**: 原始算法不支持直接的 2D 点云多直线拟合，我们的适配是必要的

3. **相同起点**: 所有方法（包括 RANSAC、K-Means、GMM）都从相同的输入数据和评估指标出发

### 局限性

1. **无深度学习组件**: PARSAC 的神经网络采样权重被替换为均匀随机采样
2. **性能差异**: SupeRANSAC 的 C++ 实现被替换为 Python，可能有性能差异
3. **单任务评估**: 我们只在 2D 直线拟合任务上评估，不代表原始算法的全部能力

### 建议在论文中的表述

```
We implement simplified versions of PARSAC and SupeRANSAC adapted for 
2D line fitting, preserving their core algorithmic principles (parallel 
hypothesis generation, soft inlier weighting, greedy selection) while 
replacing domain-specific components (neural network sampling, image 
feature processing) with geometric alternatives suitable for our task.
```

---

## 📄 许可证

MIT License
