# 超平面拟合方法对比评估

[英文版](README-Method-Comparison.md)
本文档说明如何使用我们提供的评估框架对比不同的超平面拟合方法，包括 **2D 直线拟合** 和 **3D 平面拟合** 任务。

## 📋 目录

- [额外项目结构](#额外项目结构)
- [方法列表](#方法列表)
- [快速评估](#快速评估)
- [评估结果](#评估结果)
- [评估指标说明](#评估指标说明)
- [运行单个方法](#运行单个方法)

---
## 📁 额外项目结构

```
hyperplanes_fitting/
├── visualization/                # 可视化工具
├── results/                      # 补充测试结果
└── scripts-for-eval/             # 额外baseline及测试方法
```


## 方法列表

本评估框架支持以下 8 种方法的对比：

| 方法 | 类型 | 2D | 3D | 模型数量模式 |
|------|------|----|----|-------------|
| **Ours** | 流形优化 | ✓ | ✓ | 自动检测 |
| **PARSAC (Known)** | 并行采样一致性 | ✓ | ✓ | 已知 |
| **PARSAC (Unknown)** | 并行采样一致性 | ✓ | ✓ | 自动检测 |
| **SupeRANSAC (Known)** | Sequential RANSAC | ✓ | ✓ | 已知 |
| **SupeRANSAC (Unknown)** | Sequential RANSAC | ✓ | ✓ | 自动检测 |
| **RANSAC** | 随机采样一致性 | ✓ | ✓ | 已知 |
| **K-Means** | 聚类 + SVD | ✓ | ✓ | 已知 |
| **GMM** | 高斯混合 + SVD | ✓ | ✓ | 已知 |

### 方法说明

- **已知模型数量模式**: 使用真实的超平面数量作为输入参数
- **自动检测模式**: 算法自动估计超平面数量（基于得分阈值或停止条件）

---

## 快速评估

### 运行完整评估

```bash
cd /path/to/hyperplanes_fitting-main
conda activate parsac

# 运行 2D 评估（所有方法）
python scripts-for-eval/run_all_evaluations.py --dim 2

# 运行 3D 评估（所有方法）
python scripts-for-eval/run_all_evaluations.py --dim 3

# 运行所有评估（2D + 3D）
python scripts-for-eval/run_all_evaluations.py --dim all
```

### 选择性评估

```bash
# 只评估特定方法
python scripts-for-eval/run_all_evaluations.py --dim 2 \
    --methods parsac_known parsac_unknown superansac_known

# 安静模式（不打印详细信息）
python scripts-for-eval/run_all_evaluations.py --dim 2 --quiet

# 指定样本数量
python scripts-for-eval/run_all_evaluations.py --dim 2 --num_samples 10
```

---

## 评估结果

### 2D 直线拟合结果

| 方法 | Total Cost | Cost Ratio | 模型数 | 运行时间 |
|------|------------|------------|--------|---------|
| PARSAC (Known) | 5.57±0.30 | 0.93±0.03 | 4.0±0.0 | 0.016s |
| PARSAC (Unknown) | 5.57±0.30 | 0.93±0.03 | 4.0±0.0 | 0.021s |
| SupeRANSAC (Known) | 5.61±0.30 | 0.94±0.03 | 4.0±0.0 | 0.016s |
| SupeRANSAC (Unknown) | 5.61±0.30 | 0.94±0.03 | 4.0±0.0 | 0.017s |
| RANSAC | 188.57±147.28 | 31.42±23.93 | 4.0±0.0 | 0.023s |
| K-Means | 40.95±22.34 | 6.89±3.75 | 4.0±0.0 | 0.013s |
| GMM | 44.25±14.31 | 7.42±2.44 | 4.0±0.0 | 0.350s |

### 3D 平面拟合结果

| 方法 | Total Cost | Cost Ratio | 模型数 | 运行时间 |
|------|------------|------------|--------|---------|
| PARSAC (Known) | 5.50±0.30 | 0.92±0.03 | 4.0±0.0 | 0.018s |
| PARSAC (Unknown) | 5.56±0.39 | 0.93±0.06 | 4.0±0.0 | 0.023s |
| SupeRANSAC (Known) | 5.55±0.30 | 0.93±0.03 | 4.0±0.0 | 0.038s |
| SupeRANSAC (Unknown) | 5.59±0.32 | 0.93±0.03 | 4.0±0.0 | 0.038s |
| RANSAC | 6.14±0.61 | 1.02±0.08 | 4.0±0.0 | 0.085s |
| K-Means | 64.81±21.75 | 10.83±3.59 | 4.0±0.0 | 0.001s |
| GMM | 30.45±25.96 | 5.12±4.42 | 4.0±0.0 | 0.015s |

### 结果分析

1. **PARSAC 和 SupeRANSAC** 在两种任务中表现最佳
2. **已知模型数量 vs 未知**：自动检测模式的性能略有下降
3. **RANSAC** 在 2D 任务中表现较差，但在 3D 任务中显著改善
4. **K-Means 和 GMM** 作为聚类方法，在超平面拟合任务中表现一般

---

## 评估指标说明

### Total Cost (TC)
点到最近超平面的总距离：
$$TC = \sum_{i=1}^{N} \min_{j} d(x_i, H_j)$$

### Cost Ratio (CR)
与真值的代价比值：
$$CR = \frac{TC_{result}}{TC_{ground\_truth}}$$

- CR = 1.0 表示与真值完全一致
- CR < 1.0 表示优于真值
- CR > 1.0 表示劣于真值

### H-bar Distance
超平面参数空间中的距离：
$$d_{H-bar} = \sum_{i} \min_{j} ||\hbar_i - g_j||$$

其中 $\hbar = d \cdot n$ 是 H-bar 表示。

### Model Count Error
模型数量误差：
$$MCE = |K_{result} - K_{ground\_truth}|$$

---

## 运行单个方法

### PARSAC

```bash
# 2D 已知模型数量
python scripts-for-eval/parsac/evaluate_parsac.py --known_count

# 2D 自动检测
python scripts-for-eval/parsac/evaluate_parsac.py

# 3D 已知模型数量
python scripts-for-eval/parsac/evaluate_parsac_3d.py --known_count

# 3D 自动检测
python scripts-for-eval/parsac/evaluate_parsac_3d.py
```

### SupeRANSAC

```bash
# 2D 已知模型数量
python scripts-for-eval/superansac/evaluate_superansac.py --known_count

# 2D 自动检测
python scripts-for-eval/superansac/evaluate_superansac.py

# 3D 已知模型数量
python scripts-for-eval/superansac/evaluate_superansac_3d.py --known_count

# 3D 自动检测
python scripts-for-eval/superansac/evaluate_superansac_3d.py
```

### 传统方法 (RANSAC, K-Means, GMM)

这些方法通过 `compared_alg/` 目录中的原始实现进行评估。

```bash
# 使用原有评估框架
python compared_alg/evaluate.py

# 或使用统一评估脚本
python scripts-for-eval/run_all_evaluations.py --methods ransac kmeans gmm
```

---

## 结果文件位置

评估结果保存在以下目录：

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

每个方法目录包含：
- `*_results.csv`: 每个样本的详细评估结果
- `*_summary.txt`: 统计汇总

---

## 参考文档

- [PARSAC/SupeRANSAC 实现细节](README-PARSAC-SupeRANSAC_zh.md): 算法实现的详细说明
