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
from mpl_toolkits.mplot3d import Axes3D  # 需要导入用于三维图的模块
import seaborn as sns
from matplotlib.colors import Normalize  # 用于归一化颜色映射
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
file_folder_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(file_folder_path)
from train_config import *


file_folder_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(file_folder_path)
from train_config import *

# Read configuration and initialize parameters
noisy_dim = config["rcgan_cellular"]["noise_size"]
epochs = config["rcgan_cellular"]["epochs"]
z_dim = config["rcgan_cellular"]["latent_size"]
sample_num = config["rcgan_cellular"]["sample_num"]
label_propertys = list(np.linspace(0, 0.95, int(0.95 / 0.05) + 1))
label_d = config["rcgan_cellular"]["label_d"]
xlsx_base_save_path = config["rcgan_cellular"]["generated_data_path"]

label_d = torch.full((sample_num, 1), label_d).to(device)

# Generator and prediction models
generator = Generator(noisy_dim, cellular_gan_output_dim, cellular_gan_input_dim).to(device)
generator.load_state_dict(torch.load(best_g_model_path_cellular))
generator.eval()

forward_net = forward_regression(cellular_input_dim, cellular_output_dim).to(device)
forward_net.load_state_dict(torch.load(cellular_pth_path))
forward_net.eval()



recons, preds, conds = [], [], []

# Generate data and predictions
for label_property in label_propertys:
    cond = torch.full((sample_num, cellular_gan_output_dim), label_property).to(device)
    z = torch.randn(sample_num, noisy_dim).to(device)

    with torch.no_grad():
        generated_data = generator(z, cond)
        prediction = forward_net(generated_data)

    pred = prediction.cpu().numpy()
    recons.append(generated_data)
    preds.append(pred)
    conds.append(label_property)

# Now we plot the 3D scatter plot

# Convert lists to numpy arrays for easier manipulation
recons = np.vstack([r.cpu().numpy() for r in recons])  # Reconstructed data
preds = np.vstack(preds)  # Predicted values

l1 = recons[:, 5]  
d0 = recons[:, 0]  
d1 = recons[:, 1]  

l1_from_dataset = normalized_cellular_input[:, 5]
d0_from_dataset = normalized_cellular_input[:, 0]
d1_from_dataset = normalized_cellular_input[:, 1]
preds_from_dataset = normalized_cellular_output

# Normalize for color mapping of the bright dataset points
dataset_norm = Normalize(vmin=preds.min(), vmax=preds.max())

# # Create a 3D scatter plot
# fig = plt.figure(figsize=(12, 10))
# ax = fig.add_subplot(111, projection='3d')

# # 绘制浅灰色点 (Reconstructed Points)
# scatter_gray = ax.scatter(l1, d0, d1, color='lightgray', alpha=0.5, s=10, label='Reconstructed Points')

# # 绘制明亮颜色的点 (Dataset Points)
# scatter_bright = ax.scatter(
#     l1_from_dataset.cpu().numpy(),
#     d0_from_dataset.cpu().numpy(),
#     d1_from_dataset.cpu().numpy(),
#     c='orange',  # 明亮颜色
#     edgecolors='black',  # 边框颜色
#     linewidth=0.5,
#     s=30,  # 点的大小
#     label='Dataset Points'
# )

# # 添加颜色条并嵌入到3D图中
# mappable = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=dataset_norm)
# mappable.set_array([])
# cbar = fig.colorbar(mappable, ax=ax, shrink=0.8, aspect=20, location='right', pad=0.1)  # 内嵌颜色条
# cbar.set_label('Elastic Modulus', fontsize=14, weight='bold')  # 调整为更小字体

# # 设置坐标轴标签
# ax.set_xlabel('l1', fontsize=18, weight='bold', labelpad=20)
# ax.set_ylabel('d0', fontsize=18, weight='bold', labelpad=20)
# ax.set_zlabel('d1', fontsize=18, weight='bold', labelpad=20)

# # 设置标题
# plt.title("Dataset and Reconstructed Points in 3D", fontsize=26, weight='bold', pad=40)

# # 初始化摄像机角度
# ax.view_init(elev=25, azim=-119)

# # 定义更新角度的函数
# def on_mouse_motion(event):
#     elev = ax.elev
#     azim = ax.azim
#     print(f"Camera Elevation: {elev}, Azimuth: {azim}")

# # 绑定鼠标事件
# fig.canvas.mpl_connect('motion_notify_event', on_mouse_motion)

# # 显示图形
# plt.tight_layout()
# plt.legend()
# plt.show()

fig, ax = plt.subplots(figsize=(10, 8))

# Create a custom colormap with the colors you've defined
colors = [
    (0.0, "#2878B5"),  # Deep Blue
    (0.3, "#9AC9DB"),  # Light Blue
    (0.5, "#FFBE7A"),  # Orange
    (0.7, "#F8AC8C"),  # Dark Red
    (1.0, "#D8383A")   # Deep Red
]
cmap = plt.cm.colors.LinearSegmentedColormap.from_list("custom_cmap", colors)

# Normalize based on the preds range
dataset_norm = Normalize(vmin=np.min(preds), vmax=np.max(preds))

# Identify points where l1 > 0 and d0 > 0
mask_positive = (l1 > 0) & (d0 > 0)

# Identify points where l1 <= 0 or d0 <= 0
mask_negative = ~mask_positive

# Scatter plot for points with l1 > 0 and d0 > 0 (color based on preds)
scatter_bright = ax.scatter(
    l1[mask_positive],
    d0[mask_positive],
    c=preds[mask_positive, 0],  # Color by preds (or any other column)
    cmap=cmap,  # Use the custom colormap
    norm=dataset_norm,  # Apply normalization
    edgecolors='black',  # Black edges for the points
    linewidth=0.5,
    s=30,  # Size of the points
    label='Generated data'  # Label for the legend
)

# Scatter plot for points with l1 <= 0 or d0 <= 0 (retain their colors, change marker to '×')
ax.scatter(
    l1[mask_negative],
    d0[mask_negative],
    c=preds[mask_negative, 0],  # Retain their original color mapping
    cmap=cmap,  # Use the same colormap
    norm=dataset_norm,  # Apply normalization
    edgecolors='black',  # Black edges for the points
    linewidth=0.5,
    marker='D',  # 'x' marker for these points
    s=50,  # Size of the 'x' marker
    label='Created data'  # Label for the legend
)

ax.scatter(
    l1_from_dataset.cpu().numpy(),
    d0_from_dataset.cpu().numpy(),
    c=preds_from_dataset.cpu().numpy(),  # Color by preds (or any other column)
    cmap=cmap,  # Use the custom colormap
    norm=dataset_norm,  # Apply normalization
    marker='^',  # Hollow triangle marker
    alpha=0.9,
    s=60,  # Size of the marker
    label='Dataset'
)

# Add color bar to show color mapping based on preds
mappable = ScalarMappable(cmap=cmap, norm=dataset_norm)
mappable.set_array([])  # Empty array for colorbar
# cbar = fig.colorbar(mappable, ax=ax, shrink=0.8, aspect=20, location='right', pad=0.1)
# cbar.set_label('Predictions (or any other metric)', fontsize=14, weight='bold')

# Set axis labels
ax.set_xlabel('l', weight='bold', labelpad=1)
ax.set_ylabel('r', weight='bold', labelpad=1)

# Add title (optional)
# plt.title("Dataset and Reconstructed Points in 2D (l1 vs d0)", fontsize=22, weight='bold', pad=20)

# Add legend
plt.legend(loc = 'upper right')

# Tight layout for better spacing
plt.tight_layout()

# Show plot
plt.show()

