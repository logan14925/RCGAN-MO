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

# 你的配置部分
file_folder_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(file_folder_path)
from train_config import *


class LossCalculation:
    def __init__(self, target, recon):
        self.target = target
        self.recon = recon
        with torch.no_grad():
            self.pred = forward_net(recon)

        self.loss_l_history = []
        self.loss_r_history = []
        self.diameters = self.recon[:, 0:5]
    def LossError(self):
        error = torch.log(abs(self.pred - self.target) + 1e-10)
        return error

    def LossR(self):
        sum_d = torch.sum(self.diameters,dim=1)
        loss_d = -torch.relu(sum_d + 1e-10)
        return loss_d

    def lossL(self):
        l = self.recon[:, 5]
        loss_l = torch.relu(l + 1)
        return loss_l

    def normalize_loss(self, loss):
        """Normalize each loss term with Min-Max scaling"""
        min_loss = torch.min(loss)
        max_loss = torch.max(loss)
        normalized_loss = (loss - min_loss) / (max_loss - min_loss + 1e-10)
        return normalized_loss

    def LossTotal(self, weights):
        weights = torch.tensor(weights, dtype=torch.float32)
        weights_trans = weights
        weights_trans[0] =  torch.pow(10, -7 + 2 *0.8 * weights[0])
        weights_trans[1] =  torch.pow(10, -7 + 2 *0.55 * weights[1])

        loss_terms = [
            self.LossR(),
            self.lossL(),
        ]
        
        # 记录每个损失项的历史
        self.loss_r_history.append(torch.mean(loss_terms[0]).item())
        self.loss_l_history.append(torch.mean(loss_terms[1]).item())
        normalized_losses = [self.normalize_loss(loss) for loss in loss_terms]
        normalized_loss_means = [torch.mean(loss) for loss in normalized_losses]
        loss_total = sum(w * loss for w, loss in zip(weights_trans, normalized_loss_means))
        return loss_total

# 配置项初始化
noisy_dim = config["rcgan_cellular"]["noise_size"]
epochs = config["rcgan_cellular"]["epochs"]
sample_num = config["rcgan_cellular"]["sample_num"]
label_propertys = [0.3]  # 假设为一个示例值
label_d = config["rcgan_cellular"]["label_d"]

gen_batch_size = config["rcgan_cellular"]["gradient_optimizer"]["genrated_size"]
weights = config["rcgan_cellular"]["gradient_optimizer"]["weights"]
gen_lr = config["rcgan_cellular"]["gradient_optimizer"]["lr"]
gen_epochs = config["rcgan_cellular"]["gradient_optimizer"]["epochs"]

weights = np.array(weights)
label_d = torch.full((gen_batch_size, 1), label_d).to(device)  # Initialize a tensor with shape (sample_num, 1)

# Generator and prediction models
generator = Generator(noisy_dim, cellular_gan_output_dim, cellular_gan_input_dim).to(device)
generator.load_state_dict(torch.load(best_g_model_path_cellular))
generator.eval()

forward_net = forward_regression(cellular_input_dim, cellular_output_dim).to(device)
forward_net.load_state_dict(torch.load(cellular_pth_path))
forward_net.eval()


losses = []
errors = []
l1_means = []
r_means = []
d1_means = []
d2_means = []
d3_means = []
d4_means = []

total_losses = []  # 用于记录 Total Loss 的变化

for label_property in label_propertys:
    cond = torch.full((gen_batch_size, cellular_gan_output_dim), label_property).to(device)
    z = torch.randn(gen_batch_size, noisy_dim).to(device)

    with torch.no_grad():
        generated_data = generator(z, cond)
        prediction = forward_net(generated_data)
        error = prediction-cond

    valid_mask = (error.abs() < 0.05).all(dim=1)  # 判断每个样本的 error 是否小于 0.05
    generated_data = generated_data[valid_mask]  # 筛选出符合条件的数据
    prediction = prediction[valid_mask]
    
    generated_data.requires_grad = True 
    optimizer = torch.optim.Adam([generated_data], lr=gen_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=3, verbose=True, min_lr=1e-20)
    epochs = gen_epochs

    for epoch in range(epochs):
        loss_cal = LossCalculation(label_property, generated_data)
        loss = loss_cal.LossTotal(weights)
        losses.append(loss.item())
        total_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _error = torch.mean((forward_net(generated_data) - label_property)**2)  # 重新计算预测误差
        
        # 计算每个生成数据的均值
        l1_data = torch.mean(generated_data[:, 5]).item()  # .item() 用于将 Tensor 转为标量
        r_data = torch.mean(generated_data[:, 0]).item()
        d1_data = torch.mean(generated_data[:, 1]).item()
        d2_data = torch.mean(generated_data[:, 2]).item()
        d3_data = torch.mean(generated_data[:, 3]).item()
        d4_data = torch.mean(generated_data[:, 4]).item()
        
        errors.append(_error.item())
        l1_means.append(l1_data)
        r_means.append(r_data)
        d1_means.append(d1_data)
        d2_means.append(d2_data)
        d3_means.append(d3_data)
        d4_means.append(d4_data)

        # Print d, l1, l2, l3 every 20 epochs
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], l: {l1_data:.4f}, r: {r_data:.4f}, d1: {d1_data:.4f}, d2: {d2_data:.4f}")
        
        scheduler.step(loss)
        if _error.item() > 0.1:
            print(f"Epoch {epoch+1}: Error is greater than 0.1, stopping training.")
            break  # 退出循环，停止训练
    
    l1 = l1_means[-1]
    r = r_means[-1]
    d1 = d1_means[-1]
    d2 = d2_means[-1]
    d3 = d3_means[-1]
    d4 = d4_means[-1]
    input_array = torch.tensor([r, d1, d2, d3, d4, l1]).to(device)
    print('Normalized input data is ', input_array)
    real_array = d_minmax_normal(input_array, min_cellular, max_cellular)

    print('While weights is {}, Real input data is {}'.format(weights, real_array.cpu().numpy()))

    epochs = range(len(errors))

    def min_max_scale(data):
        """
        对数据进行最小-最大缩放，将数据缩放到 [0, 1] 范围
        :param data: 输入数据列表或数组
        :return: 缩放后的数据列表或数组
        """
        min_val = np.min(data)
        max_val = np.max(data)
        if min_val == max_val:
            # 如果最大最小值相同，创建一个与输入数据相同长度，元素全为 0.5 的数组
            return np.full_like(data, 0.5)
        else:
            return (data - min_val) / (max_val - min_val)

    epochs = range(len(errors))

    scaled_errors = min_max_scale(errors)
    scaled_l1_means = min_max_scale(l1_means)
    scaled_r_means = min_max_scale(r_means)
    scaled_d1_means = min_max_scale(d1_means)
    scaled_d2_means = min_max_scale(d2_means)
    scaled_total_losses = min_max_scale(total_losses)

    fig = plt.figure(figsize=(10, 8))

    plt.plot(epochs, scaled_errors, label='Mean Error', color='black', linewidth=4)

    plt.plot(epochs, scaled_l1_means, label='l', color='green', linewidth=4)

    plt.plot(epochs, scaled_r_means, label='r', color='red', linewidth=4)

    # 绘制 Total Loss 图
    plt.plot(epochs, scaled_total_losses, label='Total Loss', color='orange', linewidth=4)


    plt.xlabel('Epochs')
    plt.ylabel('Normalized values')

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    print('dddasd')