# PARSAC 和 SupeRANSAC 评估指南

本文档说明如何在我们的 2D 直线拟合数据集上评估 PARSAC 和 SupeRANSAC 方法，以便与论文中的其他对比方法进行公平比较。

## 📋 目录

- [环境准备](#环境准备)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [详细参数说明](#详细参数说明)
- [输出结果说明](#输出结果说明)
- [与其他方法对比](#与其他方法对比)
- [评估指标说明](#评估指标说明)

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
│   ├── compare_all_methods.py      # 多方法对比脚本
│   ├── data_utils.py               # 数据读取工具
│   ├── metrics.py                  # 评估指标模块
│   │
│   ├── parsac/                     # PARSAC 评估模块
│   │   ├── __init__.py
│   │   ├── line_fitter.py          # SimplePARSACLineFitter 实现
│   │   └── evaluate_parsac.py      # PARSAC 评估脚本
│   │
│   └── superansac/                 # SupeRANSAC 评估模块
│       ├── __init__.py
│       ├── sequential_ransac.py    # Sequential RANSAC 实现
│       └── evaluate_superansac.py  # SupeRANSAC 评估脚本
│
├── results/                        # 结果输出目录
│   ├── parsac/                     # PARSAC 结果
│   │   ├── parsac_results.csv      # 逐样本详细结果
│   │   └── parsac_summary.txt      # 汇总统计
│   ├── superansac/                 # SupeRANSAC 结果
│   │   ├── superansac_results.csv
│   │   └── superansac_summary.txt
│   └── figures/                    # 对比图表
│       ├── multi_panel_comparison.png
│       ├── boxplot_comparison.png
│       ├── bar_*.png
│       ├── comparison_table.txt
│       └── detailed_results.csv
│
├── visualization/                  # 可视化脚本目录
│   └── plot_results.py             # 结果可视化工具
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

## 📄 许可证

MIT License
