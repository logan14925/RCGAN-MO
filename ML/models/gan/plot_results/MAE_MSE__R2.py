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
file_folder_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(file_folder_path)
from train_config import *
import seaborn as sns
import time  # 导入 time 模块
from sklearn.metrics import r2_score
from scipy.ndimage import gaussian_filter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_mse_mae(csv_data):
    """
    此函数用于计算 csv_data 中倒数第二列和倒数第一列数据的均方误差（MSE）和平均绝对误差（MAE）

    :param csv_data: 输入的 DataFrame
    :return: 无，直接打印 MSE 和 MAE 的值
    """
    exp_targets = csv_data.iloc[:, -2].values  # 第一列为目标数据
    exp_preds = csv_data.iloc[:, -1].values    # 第二列为预测数据

    # 计算均方误差 (MSE)
    mse_value = np.mean((exp_targets - exp_preds) ** 2)

    # 计算平均绝对误差 (MAE)
    mae_value = np.mean(np.abs(exp_targets - exp_preds))

    print(f"均方误差 (MSE): {mse_value}")
    print(f"平均绝对误差 (MAE): {mae_value}")
    
# Read configuration and initialize parameters
noisy_dim = config["rcgan_cellular"]["noise_size"]
epochs = config["rcgan_cellular"]["epochs"]
z_dim = config["rcgan_cellular"]["latent_size"]
sample_num = config["rcgan_cellular"]["sample_num"]
label_propertys = list(np.linspace(0, 0.95, int(0.95 / 0.005)))
# label_propertys = [0.1, 0.3, 0.5, 0.8]
label_d = config["rcgan_cellular"]["label_d"]
xlsx_base_save_path = config["rcgan_cellular"]["generated_data_path"]
error_threshold = config["rcgan_cellular"]["error_threshold"]
visualization_path = config["rcgan_cellular"]["visualization_path"]
label_d = torch.full((sample_num, 1), label_d).to(device)  # Initialize a tensor with shape (sample_num, 1)

generator = Generator(noisy_dim, cellular_gan_output_dim, cellular_gan_input_dim).to(device)
generator.load_state_dict(torch.load(best_g_model_path_cellular))
generator.eval()

forward_net = forward_regression(cellular_input_dim, cellular_output_dim).to(device)
forward_net.load_state_dict(torch.load(cellular_pth_path))
forward_net.eval()


recons, preds, conds = [], [], []
img_path = 'ML/models/gan/gen_img.jpg'
start_time = time.time()

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
    
end_time = time.time()

total_time = end_time - start_time
print(f"Total time taken to generate all label corresponding data: {total_time:.4f} seconds")

filtered_recons, filtered_preds, filtered_conds = [], [], []

for idx, (label_property, recon, pred, cond) in enumerate(zip(label_propertys, recons, preds, conds)):
    condition_diff = abs(label_property - pred)
    valid_mask = (condition_diff < error_threshold).squeeze()

    filtered_recon = recon[valid_mask]
    filtered_pred = pred[valid_mask]

    filtered_recons.append(filtered_recon)
    filtered_preds.append(filtered_pred)
    filtered_conds.append(cond)


# Denormalize and plot real data
real_recons = [d_minmax_normal(torch.tensor(recon, device=device), min_cellular, max_cellular) for recon in filtered_recons]
real_preds = [d_minmax_normal(torch.tensor(pred, device=device), min_cellular_output, max_cellular_output) for pred in filtered_preds]
real_conds = [d_minmax_normal(torch.tensor(cond, device=device), min_cellular_output, max_cellular_output) for cond in filtered_conds]

# 读取output_moduli.csv文件并跳过第一行
csv_path = 'Exp_results/output_moduli.csv'
csv_data = pd.read_csv(csv_path, header=0)  # 使用header=0来跳过第一行

# 从CSV文件中获取目标和预测值
exp_targets = csv_data.iloc[:, -2].values  # 第一列为目标数据
exp_preds = csv_data.iloc[:, -1].values    # 第二列为预测数据
r2_exp = r2_score(exp_targets, exp_preds)
calculate_mse_mae(csv_data)
# 打印 R² 值
print(f"Exp. R² Value: {r2_exp:.4f}")

# 读取output_moduli.csv文件并跳过第一行
fem_path = 'ML/models/gan/plot_results/exp_generated_v2.csv'
fem_data = pd.read_csv(fem_path, header=0)  # 使用header=0来跳过第一行

# 从CSV文件中获取目标和预测值
fem_targets = fem_data.iloc[:, -3].values  # 第一列为目标数据
fem_preds = fem_data.iloc[:, -2].values    # 第二列为预测数据
r2_fem = r2_score(fem_targets, fem_preds)

# 打印 R² 值
print(f"FEM R² Value: {r2_fem:.4f}")

extended_targets = []
extended_preds = []

# Ensure both targets and preds are the same length
for target, pred in zip(real_conds, real_preds):
    # Move target to CPU if it's a tensor, then convert to a scalar value
    if isinstance(target, torch.Tensor):
        target = target.cpu().item()  # Move to CPU and extract value
    # Extend target value to match the length of the corresponding pred values
    extended_targets.extend([target] * len(pred)) 

    # Move pred to CPU if it's a tensor, then convert to NumPy array
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()  # Move to CPU and convert to numpy

    extended_preds.extend(pred)  # Add pred values directly

# Convert to numpy arrays for plotting (already on CPU)
extended_targets = np.array(extended_targets)
extended_preds = np.array(extended_preds)

# 创建一个图形对象
# 创建一个图形对象
plt.figure(figsize=(10, 8))

# 绘制 y=x 的虚线
min_val = min(np.min(extended_targets), np.min(exp_targets))  # 取目标和条件数据中的最小值
max_val = max(np.max(extended_targets), np.max(exp_targets))  # 取目标和条件数据中的最大值
plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)  # 'k--'表示黑色虚线

# 绘制已有数据点
plt.scatter(extended_targets, extended_preds, color=(44/255, 145/255, 224/255), label='Generated results', alpha=0.6)
plt.scatter(fem_targets, fem_preds, color=(58/255, 191/255, 153/255), label='FEM Process', alpha=0.9, marker='s', s=80)
plt.scatter(exp_targets, exp_preds, color='#C5272D', label='Experiments', alpha=0.9, marker='^', s=100)

# 样例点的实验数据
new_x = [0.0294, 0.0406, 0.0517, 0.0685]
new_y = [0.020297969405987007, 0.04184899718436937, 0.05367208608889057, 0.06165081331241493]
plt.scatter(new_x, new_y, color='#C5272D', alpha=0.9, marker='^', s=150)

# 添加轴标签
plt.xlabel('Targets', fontweight='bold', labelpad=1)
plt.ylabel('Predictions', fontweight='bold', labelpad=1)

# 添加图例
plt.legend()

# 显示图形
plt.tight_layout()
plt.show()

print(111)
