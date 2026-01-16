#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3D 平面拟合结果可视化工具

使用 plotly 进行交互式 3D 可视化，支持：
- 点云显示（按聚类着色）
- 平面网格显示
- 真值与拟合结果对比
- 保存为 HTML 交互式图像或静态图像
"""

import numpy as np
import os
import sys

# 尝试导入可视化库
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("警告: plotly 未安装，部分功能不可用")
    print("安装命令: pip install plotly")

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def create_plane_mesh(normal, d, xlim=(-5, 5), ylim=(-5, 5), zlim=(-5, 5), n_points=20):
    """
    创建平面网格用于可视化
    
    Args:
        normal: 法向量 (3,)
        d: 距离参数
        xlim, ylim, zlim: 坐标范围
        n_points: 网格点数
    
    Returns:
        xx, yy, zz: 网格坐标
    """
    n = np.array(normal)
    n = n / np.linalg.norm(n)
    
    # 根据法向量主方向选择参数化方式
    abs_n = np.abs(n)
    max_idx = np.argmax(abs_n)
    
    if max_idx == 2:  # z 分量最大，用 x, y 参数化
        xx, yy = np.meshgrid(
            np.linspace(xlim[0], xlim[1], n_points),
            np.linspace(ylim[0], ylim[1], n_points)
        )
        if abs(n[2]) > 1e-10:
            zz = (d - n[0] * xx - n[1] * yy) / n[2]
        else:
            return None, None, None
    elif max_idx == 1:  # y 分量最大，用 x, z 参数化
        xx, zz = np.meshgrid(
            np.linspace(xlim[0], xlim[1], n_points),
            np.linspace(zlim[0], zlim[1], n_points)
        )
        if abs(n[1]) > 1e-10:
            yy = (d - n[0] * xx - n[2] * zz) / n[1]
        else:
            return None, None, None
    else:  # x 分量最大，用 y, z 参数化
        yy, zz = np.meshgrid(
            np.linspace(ylim[0], ylim[1], n_points),
            np.linspace(zlim[0], zlim[1], n_points)
        )
        if abs(n[0]) > 1e-10:
            xx = (d - n[1] * yy - n[2] * zz) / n[0]
        else:
            return None, None, None
    
    # 裁剪到边界
    mask = (
        (xx >= xlim[0]) & (xx <= xlim[1]) &
        (yy >= ylim[0]) & (yy <= ylim[1]) &
        (zz >= zlim[0]) & (zz <= zlim[1])
    )
    
    return xx, yy, zz


def visualize_3d_plotly(points, labels=None, planes_result=None, planes_gt=None,
                        title="3D Plane Fitting Result", save_path=None):
    """
    使用 Plotly 进行交互式 3D 可视化
    
    Args:
        points: (N, 3) 点云
        labels: (N,) 聚类标签 (可选)
        planes_result: 拟合结果平面列表 [(n, d), ...] (可选)
        planes_gt: 真值平面列表 [(n, d), ...] (可选)
        title: 图标题
        save_path: 保存路径 (可选)
    """
    if not HAS_PLOTLY:
        print("错误: 需要安装 plotly 库")
        return
    
    fig = go.Figure()
    
    # 计算坐标范围
    xlim = (points[:, 0].min() - 1, points[:, 0].max() + 1)
    ylim = (points[:, 1].min() - 1, points[:, 1].max() + 1)
    zlim = (points[:, 2].min() - 1, points[:, 2].max() + 1)
    
    # 颜色列表
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # 绘制点云
    if labels is not None:
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if label == -1:
                color = 'gray'
                name = 'Outliers'
            else:
                color = colors[int(label) % len(colors)]
                name = f'Cluster {label}'
            
            fig.add_trace(go.Scatter3d(
                x=points[mask, 0],
                y=points[mask, 1],
                z=points[mask, 2],
                mode='markers',
                marker=dict(size=3, color=color, opacity=0.7),
                name=name
            ))
    else:
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(size=3, color='blue', opacity=0.7),
            name='Points'
        ))
    
    # 绘制拟合平面
    if planes_result is not None:
        for i, (normal, d) in enumerate(planes_result):
            xx, yy, zz = create_plane_mesh(normal, d, xlim, ylim, zlim)
            if xx is not None:
                fig.add_trace(go.Surface(
                    x=xx, y=yy, z=zz,
                    colorscale=[[0, colors[i % len(colors)]], [1, colors[i % len(colors)]]],
                    opacity=0.3,
                    showscale=False,
                    name=f'Fitted Plane {i}'
                ))
    
    # 绘制真值平面
    if planes_gt is not None:
        for i, (normal, d) in enumerate(planes_gt):
            xx, yy, zz = create_plane_mesh(normal, d, xlim, ylim, zlim)
            if xx is not None:
                fig.add_trace(go.Surface(
                    x=xx, y=yy, z=zz,
                    colorscale=[[0, 'black'], [1, 'black']],
                    opacity=0.15,
                    showscale=False,
                    name=f'GT Plane {i}'
                ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        legend=dict(x=1.02, y=0.98),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    if save_path:
        if save_path.endswith('.html'):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path)
        print(f"图像已保存到: {save_path}")
    
    return fig


def visualize_3d_matplotlib(points, labels=None, planes_result=None, planes_gt=None,
                            title="3D Plane Fitting Result", save_path=None, show=True):
    """
    使用 Matplotlib 进行 3D 可视化（静态）
    
    Args:
        points: (N, 3) 点云
        labels: (N,) 聚类标签 (可选)
        planes_result: 拟合结果平面列表 [(n, d), ...] (可选)
        planes_gt: 真值平面列表 [(n, d), ...] (可选)
        title: 图标题
        save_path: 保存路径 (可选)
        show: 是否显示图像
    """
    if not HAS_MATPLOTLIB:
        print("错误: 需要安装 matplotlib 库")
        return
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 颜色列表
    colors = ['blue', 'orange', 'green', 'red', 'purple',
              'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # 计算坐标范围
    xlim = (points[:, 0].min() - 1, points[:, 0].max() + 1)
    ylim = (points[:, 1].min() - 1, points[:, 1].max() + 1)
    zlim = (points[:, 2].min() - 1, points[:, 2].max() + 1)
    
    # 绘制点云
    if labels is not None:
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if label == -1:
                color = 'gray'
                label_name = 'Outliers'
            else:
                color = colors[int(label) % len(colors)]
                label_name = f'Cluster {label}'
            
            ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                      c=color, s=10, alpha=0.7, label=label_name)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c='blue', s=10, alpha=0.7, label='Points')
    
    # 绘制拟合平面
    if planes_result is not None:
        for i, (normal, d) in enumerate(planes_result):
            xx, yy, zz = create_plane_mesh(normal, d, xlim, ylim, zlim, n_points=10)
            if xx is not None:
                ax.plot_surface(xx, yy, zz, alpha=0.3,
                              color=colors[i % len(colors)],
                              label=f'Fitted Plane {i}')
    
    # 绘制真值平面
    if planes_gt is not None:
        for i, (normal, d) in enumerate(planes_gt):
            xx, yy, zz = create_plane_mesh(normal, d, xlim, ylim, zlim, n_points=10)
            if xx is not None:
                ax.plot_surface(xx, yy, zz, alpha=0.15, color='black')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def compare_methods_3d(points, results_dict, gt_planes=None, save_dir=None):
    """
    对比多种方法的 3D 拟合结果
    
    Args:
        points: (N, 3) 点云
        results_dict: {method_name: (planes, labels), ...}
        gt_planes: 真值平面列表 [(n, d), ...] (可选)
        save_dir: 保存目录 (可选)
    """
    if not HAS_PLOTLY:
        print("错误: 需要安装 plotly 库")
        return
    
    n_methods = len(results_dict)
    
    # 创建子图
    fig = make_subplots(
        rows=1, cols=n_methods,
        specs=[[{'type': 'scatter3d'}] * n_methods],
        subplot_titles=list(results_dict.keys())
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for col, (method_name, (planes, labels)) in enumerate(results_dict.items(), 1):
        # 添加点云
        if labels is not None:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                if label == -1:
                    color = 'gray'
                else:
                    color = colors[int(label) % len(colors)]
                
                fig.add_trace(
                    go.Scatter3d(
                        x=points[mask, 0],
                        y=points[mask, 1],
                        z=points[mask, 2],
                        mode='markers',
                        marker=dict(size=2, color=color, opacity=0.7),
                        showlegend=False
                    ),
                    row=1, col=col
                )
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode='markers',
                    marker=dict(size=2, color='blue', opacity=0.7),
                    showlegend=False
                ),
                row=1, col=col
            )
    
    fig.update_layout(
        title="3D Plane Fitting: Method Comparison",
        height=600,
        width=400 * n_methods
    )
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "method_comparison_3d.html")
        fig.write_html(save_path)
        print(f"对比图已保存到: {save_path}")
    
    return fig


if __name__ == "__main__":
    # 测试可视化
    np.random.seed(42)
    
    # 生成测试数据
    all_points = []
    
    # 平面 1: z = 0
    p1 = np.column_stack([
        np.random.uniform(-3, 3, 40),
        np.random.uniform(-3, 3, 40),
        np.random.randn(40) * 0.05
    ])
    all_points.append(p1)
    
    # 平面 2: y = 2
    p2 = np.column_stack([
        np.random.uniform(-3, 3, 40),
        2 + np.random.randn(40) * 0.05,
        np.random.uniform(-3, 3, 40)
    ])
    all_points.append(p2)
    
    # 平面 3: x + y + z = 3
    t1 = np.random.uniform(-2, 2, 40)
    t2 = np.random.uniform(-2, 2, 40)
    p3 = np.column_stack([
        t1, t2, 3 - t1 - t2 + np.random.randn(40) * 0.05
    ])
    all_points.append(p3)
    
    points = np.vstack(all_points)
    labels = np.concatenate([np.full(40, i) for i in range(3)])
    
    # 定义平面
    planes = [
        (np.array([0, 0, 1]), 0),      # z = 0
        (np.array([0, 1, 0]), 2),      # y = 2
        (np.array([1, 1, 1]) / np.sqrt(3), 3 / np.sqrt(3))  # x + y + z = 3
    ]
    
    print("测试 3D 可视化...")
    
    if HAS_PLOTLY:
        fig = visualize_3d_plotly(points, labels, planes, title="Test 3D Visualization")
        fig.show()
    elif HAS_MATPLOTLIB:
        visualize_3d_matplotlib(points, labels, planes, title="Test 3D Visualization")
    else:
        print("没有可用的可视化库")
