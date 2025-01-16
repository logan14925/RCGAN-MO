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


file_folder_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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
        

        self.loss_r_history.append(torch.mean(loss_terms[0]).item())
        self.loss_l_history.append(torch.mean(loss_terms[1]).item())
        normalized_losses = [self.normalize_loss(loss) for loss in loss_terms]
        normalized_loss_means = [torch.mean(loss) for loss in normalized_losses]
        loss_total = sum(w * loss for w, loss in zip(weights_trans, normalized_loss_means))
        return loss_total

class PathVisualizer:
    def __init__(self):
        self.start_data = None  
        self.end_datas = []  
        self.weights_labels = []  

    def set_start_data(self, data):
        """记录初始数据"""
        self.start_data = data.cpu().detach().numpy()

    def set_end_data(self, data, weight_label):
        """记录最终数据"""
        self.end_datas.append(data.cpu().detach().numpy())
        self.weights_labels.append(weight_label)

    def plot_start_end(self):
        """
        绘制初始和最终数据结果
        """
        plt.figure(figsize=(10, 8))


        if self.start_data is not None:
            plt.scatter(self.start_data[:, 5], self.start_data[:, 0],
                        color='#EF2C2B', label='Start', s = 40)


        colors = ['#6AAF2D', '#AA66EB', '#F5AB5E', '#324C63','#3981AF']
        for i, end_data in enumerate(self.end_datas):
            plt.scatter(end_data[:, 5], end_data[:, 0],
                        color = colors[i], label = f'End(w= {self.weights_labels[i]})', s = 40)


        if self.start_data is not None:
            for i in range(len(self.end_datas)):
                for j in range(min(self.start_data.shape[0], self.end_datas[i].shape[0])):
                    plt.plot([self.start_data[j, 5], self.end_datas[i][j, 5]],
                            [self.start_data[j, 0], self.end_datas[i][j, 0]],
                            color = colors[i], alpha = 0.01, linestyle = '--')

        plt.xlim(0.15, 0.8)
        plt.ylim(0.4, 1.0)
        # plt.title('Generated Data: Start vs End', fontsize = 22)
        plt.xlabel('l')
        plt.ylabel('r')


        plt.legend(loc="best")
        # plt.grid(True)
        plt.tight_layout()
        plt.show()
        

path_visualizer = PathVisualizer()


noisy_dim = config["rcgan_cellular"]["noise_size"]
label_propertys = [0.3]  
label_d = config["rcgan_cellular"]["label_d"]

gen_batch_size = 1000
gen_lr = config["rcgan_cellular"]["gradient_optimizer"]["lr"]
gen_epochs = config["rcgan_cellular"]["gradient_optimizer"]["epochs"]

label_d = torch.full((gen_batch_size, 1), label_d).to(device)  # Initialize a tensor with shape (sample_num, 1)

# Generator and prediction models
generator = Generator(noisy_dim, cellular_gan_output_dim, cellular_gan_input_dim).to(device)
generator.load_state_dict(torch.load(best_g_model_path_cellular))
generator.eval()

forward_net = forward_regression(cellular_input_dim, cellular_output_dim).to(device)
forward_net.load_state_dict(torch.load(cellular_pth_path))
forward_net.eval()



total_losses = []  

weights_list = [[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5]]
for weights in weights_list:
    weights = np.array(weights)
    for label_property in label_propertys:
        cond = torch.full((gen_batch_size, cellular_gan_output_dim), label_property).to(device)
        z = torch.randn(gen_batch_size, noisy_dim).to(device)

        with torch.no_grad():
            generated_data = generator(z, cond)
            prediction = forward_net(generated_data)
            error = prediction - cond


        valid_mask = (error.abs() < 0.05).all(dim=1)  
        generated_data = generated_data[valid_mask]  
        prediction = prediction[valid_mask]
        if path_visualizer.start_data is None:
            path_visualizer.set_start_data(generated_data)

        generated_data.requires_grad = True 
        optimizer = torch.optim.Adam([generated_data], lr=gen_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min', factor=0.2, patience=3, verbose=True, min_lr=1e-20)
        epochs = gen_epochs

        for epoch in range(epochs):
            loss_cal = LossCalculation(label_property, generated_data)
            loss = loss_cal.LossTotal(weights)
            total_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                _error = torch.mean((forward_net(generated_data) - label_property)**2) 
            
            scheduler.step(loss)
            if _error.item() > 0.1:
                print(f"Epoch {epoch+1}: Error is greater than 0.1, stopping training.")
                break  
        path_visualizer.set_end_data(generated_data, weights)

path_visualizer.plot_start_end()

