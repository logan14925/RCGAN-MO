import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import utils
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap, Normalize
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
file_folder_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

sys.path.append(file_folder_path)
from train_config import *

plt.rcParams.update({
    'font.weight': 'bold',            # 加粗字体
    'axes.labelweight': 'bold',       # 加粗坐标轴标签
    'axes.titlesize': 26,            # 增加标题的字体大小
    'axes.labelsize': 18,            # 增加坐标轴标签字体大小
    'xtick.labelsize': 14,           # 增加X轴刻度字体大小
    'ytick.labelsize': 14,           # 增加Y轴刻度字体大小
    'axes.titleweight': 'bold',      # 标题加粗
    'legend.fontsize': 20            # 设置图例字体大小
})

output_file = 'ML/models/gan/multi-objective_optimizer/loss_results_error_halfweights.csv'
df = pd.read_csv(output_file)

# 提取数据
d_weights_for_plot = df['d_weight'].values
l3_weights_for_plot = df['l3_weight'].values
final_d = df['final_r'].values
final_d1 = df['final_d1'].values
final_l3 = df['final_l'].values
errors = df['error'].values

# 转换为网格形状（用于绘制平面）
dweights = np.unique(d_weights_for_plot)
lweights = np.unique(l3_weights_for_plot)
d_grid, l3_grid = np.meshgrid(dweights, lweights)

final_d_grid = final_d.reshape(d_grid.shape)
final_l3_grid = final_l3.reshape(d_grid.shape)
final_d1_grid = final_d1.reshape(d_grid.shape)
errors_grid = errors.reshape(d_grid.shape)

print("数据已加载完毕，准备绘图。")



color_blue = [
    (0.0, "#9EAAD1"),  # Very Light Blue
    (1.0, "#3B549D"),  # Deep Blue
]

color_red = [
    (0.0, "#FFCDD2"),  # Very Light Red
    (1.0, "#B71C1C"),  # Deep Red
]

color_green = [
    (0.0, "#DCEDC8"),  # Very Light Green
    (1.0, "#2E7D32"),  # Deep Green
]

# 创建自定义的颜色映射
cmap_b = LinearSegmentedColormap.from_list("blue_gradient", color_blue)
cmap_red = LinearSegmentedColormap.from_list("red_gradient", color_red)
cmap_green = LinearSegmentedColormap.from_list("green_gradient", color_green)

# 创建归一化对象（根据数据的最大最小值自动调整）
norm_d = Normalize(vmin=min(final_d), vmax=max(final_d))  # final_d的归一化
norm_l3 = Normalize(vmin=min(final_l3), vmax=max(final_l3))  # final_l3的归一化

fig = plt.figure(figsize=(11, 11))
ax = fig.add_subplot(111, projection='3d')

scatter_size = 50
# 高亮点的坐标
highlight_points_x = [0, 0, 1, 1, 0.5]  # d_weights_for_plot 的坐标
highlight_points_y = [0, 1, 0, 1, 0.5]  # l3_weights_for_plot 的坐标

# Z 方向的范围（从数据的最小值到最大值）
z_min = min(min(final_d), min(final_l3), np.min(final_d1_grid))
z_max = max(max(final_d), max(final_l3), np.max(final_d1_grid))
# for x, y in zip(highlight_points_x, highlight_points_y):
#     ax.bar3d(
#         x, y, z_min,    # 起点坐标 (x, y, z_min)
#         0.03, 0.03,     # 柱子的宽度和深度
#         z_max - z_min,  # 柱子的高度
#         color='gray', # 柱体颜色
#         alpha=0.3       # 不透明度
#     )
#     ax.text(x, y, z_min, f'({x:.2f}, {y:.2f})', color='black', fontsize=14, ha='center', va='top')

scatter = ax.scatter(d_weights_for_plot, l3_weights_for_plot, final_d, c=final_d, 
                     cmap=cmap_b, marker='o', s=scatter_size, label='d', norm=norm_d)
scatter_l3 = ax.scatter(d_weights_for_plot, l3_weights_for_plot, final_l3, c=final_l3, 
                        cmap=cmap_red, marker='^', s=scatter_size, label='l', norm=norm_l3)



# 绘制拟合面
ax.plot_trisurf(d_weights_for_plot, l3_weights_for_plot, final_d, cmap='Blues', alpha=0.8)
ax.plot_trisurf(d_weights_for_plot, l3_weights_for_plot, final_l3, cmap='Reds', alpha=0.8)
ax.plot_trisurf(d_weights_for_plot, l3_weights_for_plot, final_d1, cmap='Greens', alpha=0.8)

ax.set_xlabel(r'$W_d$', labelpad=7)
ax.set_ylabel(r'$W_l$', labelpad=7)
ax.set_zlabel('Values', labelpad=5)
# ax.legend(['r', 'l', 'd1'], loc='upper left',fontsize=20)

plt.tight_layout()

ax.view_init(elev=15, azim=-135)
def on_mouse_motion(event):
    elev = ax.elev
    azim = ax.azim
    print(f"Camera Elevation: {elev}, Azimuth: {azim}")

fig.canvas.mpl_connect('motion_notify_event', on_mouse_motion)

# 显示图形
plt.show()

