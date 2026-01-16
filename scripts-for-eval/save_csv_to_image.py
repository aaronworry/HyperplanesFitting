import numpy as np
import cv2


def pixel_line_to_physical_line(a, b, c,
                                x_min=-10, x_max=10,
                                y_min=-10, y_max=10,
                                W=640, H=480):
    """
    将像素坐标系(u,v)下的直线方程转换为物理坐标系(x,y)下的直线方程

    参数说明：
    ----------
    a, b, c : float
        像素直线的一般式参数：a*u + b*v + c = 0
    x_min, x_max : float (默认-10,10)
        物理坐标系x轴范围（单位：m）
    y_min, y_max : float (默认-10,10)
        物理坐标系y轴范围（单位：m）
    W : int (默认640)
        图像宽度（像素）
    H : int (默认480)
        图像高度（像素）

    返回值：
    ----------
    A, B, C : float
        物理直线的一般式参数：A*x + B*y + C = 0
    physical_line_eq : str
        物理直线方程的字符串（便于查看）
    """
    # 步骤1：计算坐标映射的缩放系数
    scale_x = (x_max - x_min) / W    # x轴缩放系数（m/像素）
    scale_y = (y_max - y_min) / H    # y轴缩放系数（m/像素）

    # 步骤2：推导物理直线的参数A, B, C
    # 核心公式推导：
    # u = (x - x_min)/scale_x
    # v = H - (y - y_min)/scale_y
    # 代入像素直线方程 a*u + b*v + c = 0 并整理
    A = a / scale_x
    B = -b / scale_y
    C = -a * x_min / scale_x + b * (H + y_min / scale_y) + c
    return A, B, C




# ---------------------- 1. 定义参数 ----------------------
# 图像尺寸
W = 640
H = 480
# 物理坐标范围（m）
x_min, x_max = -10, 10
y_min, y_max = -10, 10
# 点的颜色：黑色 (BGR格式，OpenCV默认BGR)
point_color = (0, 0, 0)
# 点的大小（像素半径）
point_radius = 1
# 点集示例（替换为你的 N×2 点集矩阵）
# 格式：points = np.array([[x1,y1], [x2,y2], ..., [xn,yn]], dtype=np.float32)
points = np.array([
    [1, 2], [-3, 5], [5, -4], [-8, -8], [10, 10], [-10, -10]
], dtype=np.float32)

# ---------------------- 2. 创建白色背景图像 ----------------------
# 白色背景：BGR(255,255,255)
img = np.ones((H, W, 3), dtype=np.uint8) * 255

# ---------------------- 3. 坐标映射 + 绘制点 ----------------------
# x朝右, y朝上
# u朝右, v朝下
for (x, y) in points:
    # 计算像素坐标u（x轴映射）
    u = int((x - x_min) / (x_max - x_min) * W)
    # 计算像素坐标v（y轴映射，注意取反）
    v = int(H - (y - y_min) / (y_max - y_min) * H)

    # 边界检查：确保点在图像内
    if 0 <= u < W and 0 <= v < H:
        # 绘制圆形点（也可用cv2.line画点）
        cv2.circle(img, (u, v), point_radius, point_color, -1)  # -1表示填充圆

# ---------------------- 4. 保存图像 ----------------------
save_path = "point_set_image.png"
cv2.imwrite(save_path, img)
print(f"图像已保存到: {save_path}")
