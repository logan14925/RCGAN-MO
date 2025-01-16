import torch
import numpy as np
from torch.utils.data import Dataset, random_split
from torch import nn
import torch.nn.functional as F
import os
import sys
file_folder_path = os.path.dirname(os.path.abspath(__file__))
print(file_folder_path)
sys.path.append(file_folder_path)
from torch.utils.data import DataLoader
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt
import json
import pandas as pd


torch.manual_seed(43)
class MyDataset(Dataset):
    """Coder在写这段代码的时候满脑都是小恐龙
    
    Args:
        Dataset (class): torch
    """
    def __init__(self, x, y):
        """_summary_

        Args:
            x (numpy): 输入层数据，未经归一化！！！
            y (numpy): 输出层数据，未经归一化！！！
        """
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        """X 进行了归一化， y没有进行归一化
        """
        return self.x[index], self.y[index]

    # def get_MINMAX_normal_data(self):
    #     """_summary_
    #     """
    #     self.normal_type = "min-max"
    #     self.get_min_max_data()
    #     self.normal_x = torch.div((self.x - self.min_x), (self.max_x - self.min_x))
    #     self.normal_y = torch.div((self.y - self.min_y), (self.max_y - self.min_y))

    # def z_score_normal(self):
    #     self.normal_type = "z-score"
    #     self.mean_x = torch.mean(self.x, dim=0)[0]
    #     self.std_x = torch.std(self.x, dim=0)
    #     self.mean_y = torch.mean(self.y, dim=0)
    #     self.std_y = torch.std(self.y, dim=0)

    #     self.normal_x = (self.x - self.mean_x) / self.std_x
    #     self.normal_y = (self.y - self.mean_y) / self.std_y


    # def get_min_max_data(self):
    #     """MIN-MAX归一化, 计算min_x, max_x, min_y, max_y
    #     """
    #     self.min_x = torch.min(self.x, dim=0)[0]
    #     self.max_x = torch.max(self.x, dim=0)[0]
    #     self.min_y = torch.min(self.y, dim=0)[0]
    #     self.max_y = torch.max(self.y, dim=0)[0]

    # def get_mean_std_data(self):
    #     """Z-Score归一化,计算mean_x, std_x, mean_y, std_y
    #     """
    #     self.mean_x = torch.mean(self.x, dim=0)[0]
    #     self.std_x = torch.std(self.x, dim=0)
    #     self.mean_y = torch.mean(self.y, dim=0)
    #     self.std_y = torch.std(self.y, dim=0)

def z_score_normal(x):
    mean_x = torch.mean(x, dim=0)[0]
    std_x = torch.std(x, dim=0)
    print('std is ', std_x)
    normal_x = (x - mean_x) / std_x
    return normal_x

def get_minmax_normal_data(x):
    min_x = torch.min(x, dim=0)[0]
    max_x = torch.max(x, dim=0)[0]
    normal_x = torch.div((x - min_x), (max_x - min_x))
    return normal_x, min_x, max_x

def normalize_mean_variance(data):
    mean = torch.mean(data, dim=0)
    std = torch.std(data, dim=0)
    normalized_data = (data - mean) / std
    return normalized_data, mean, std

def get_data_loader(inputs, targets):
    dataset = MyDataset(inputs, targets)

    total_size = len(dataset)
    train_ratio = 0.8
    val_ratio = 0.15
    test_ratio = 0.05

    # 计算每个数据集的大小
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader

def d_minmax_normal(x, min_val, max_val):
    # 判断输入类型，并选择适当的操作
    if isinstance(x, torch.Tensor):
        dnormal_x = torch.mul(x, (max_val - min_val)) + min_val
    elif isinstance(x, np.ndarray):
        if isinstance(min_val, torch.Tensor):
            min_val = min_val.cpu().numpy()
        elif isinstance(min_val, np.ndarray):
            min_val = min_val
        else:
            raise TypeError("min_val must be either a torch.Tensor or numpy.ndarray")

        if isinstance(max_val, torch.Tensor):
            max_val = max_val.cpu().numpy()
        elif isinstance(max_val, np.ndarray):
            max_val = max_val
        else:
            raise TypeError("max_val must be either a torch.Tensor or numpy.ndarray")

        dnormal_x = np.multiply(x, (max_val - min_val)) + min_val
    else:
        raise TypeError("Input must be either a torch.Tensor or numpy.ndarray")
    return dnormal_x


def d_zscore_normal(normalized_data, means, stds):
    denormalized_data = normalized_data * stds + means
    return denormalized_data

def one_hot_encoder(id_list):
    """
    将ID转化为按排序后的独热编码
    
    参数:
    id_list (list): 待编码的ID
    
    返回:
    numpy数组: 表示独热编码的数组
    """
    # 获取唯一的id并进行排序
    unique_ids = np.unique(id_list)
    sorted_ids = np.sort(unique_ids)  # 获取排序后的id列表
    
    # 构建原始id到新排序号的映射
    id_mapping = {original_id: new_id for new_id, original_id in enumerate(sorted_ids)}
    
    # 初始化独热编码数组
    data_num = len(id_list)
    num_unique_ids = len(sorted_ids)
    encoded = np.zeros((data_num, num_unique_ids))
    
    # 进行独热编码
    for i, original_id in enumerate(id_list):
        sorted_id = id_mapping[original_id]
        encoded[i][sorted_id] = 1  # 在对应的位置置1
    
    return encoded

def one_hot_decoder(id_list, onehot_code):
    """
    将onehot code转化为原始数据
    
    参数:
    id_list (list): 原始数据集ID
    onehot_code (numpy array): 需要解码的独热编码
    
    返回:
    numpy数组: 解码后的原始ID列表
    """
    # 获取唯一的id并进行排序
    unique_ids = np.unique(id_list)
    sorted_ids = np.sort(unique_ids)  # 获取排序后的id列表
    
    # 初始化一个空的decode_list用于存储解码后的ID
    decode_list = []
    
    # 对每一行的独热编码进行解码
    for onehot in onehot_code:
        # 获取值为1的位置
        index = np.argmax(onehot)  # 找到独热编码中1的位置
        decode_list.append(sorted_ids[index])  # 根据索引找到对应的原始id
    
    return np.array(decode_list)
 
class forward_regression(nn.Module):
    def __init__(self, forward_input_dims, forward_output_dims):
        """_summary_

        Args:
            forward_input_dims (_type_): _description_
        """
        super(forward_regression, self).__init__()
        self.hidden1 = nn.Linear(forward_input_dims, 5000)
        self.hidden2 = nn.Linear(5000, 5000)
        self.hidden3 = nn.Linear(5000, 576)
        self.hidden4 = nn.Linear(576, 252)
        self.predict = nn.Linear(252, forward_output_dims)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float)
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden3(x)
        x = F.relu(x)
        x = self.hidden4(x)
        x = F.relu(x)
        predict = self.predict(x)
        return predict
    
 # Generator model
class Generator(nn.Module):
    def __init__(self, z_dim, y_dim, x_dim):
        super(Generator, self).__init__()
        self.fc = nn.Linear(z_dim + y_dim, 4096)        
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.predict = nn.Linear(256, x_dim)

    def forward(self, z, y):
        x = torch.cat([z, y], dim=1)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        x = torch.tanh(self.predict(x))
        return x

 # Discriminator model
class Discriminator(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(x_dim + y_dim, 4096)        
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.predict = nn.Linear(256, 1)
        
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        x = torch.sigmoid(self.predict(x))
        return x
 
# Regressor model
class Regressor(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Regressor, self).__init__()
        self.fc = nn.Linear(x_dim, 512)        
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.predict_x = nn.Linear(64, x_dim)
        self.predict_y = nn.Linear(64, y_dim)

    def forward(self, x):
        # x = torch.cat((x, d), dim=1)  # dim=1 表示在列维度上拼接
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        latent_x = torch.tanh(self.predict_x(x))
        y = torch.tanh(self.predict_y(x))
        return latent_x, y

def read_config(json_path):
    with open(json_path) as json_file:
        config = json.load(json_file)
    return config
 
 
json_path = 'ML/configs/nn.json'
config = read_config(json_path)
dataset_path = config["dataset_path"]
pth_save_addr = config["pth_address"]



appendix_path = config["rcgan_cellular"]["apppendix_path"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################
# 晶格结构代码
######################

cellular_path = os.path.join(dataset_path, 'data_for_ml.csv')
cellular_data = np.genfromtxt(cellular_path, delimiter=',', skip_header=1)

cellular_input = cellular_data[:, 0:6]
cellular_output = cellular_data[:,11:12]

cellular_input = torch.tensor(cellular_input).to(device)
cellular_output = torch.tensor(cellular_output).to(device)


normalized_cellular_input, min_cellular, max_cellular = get_minmax_normal_data(cellular_input)
normalized_cellular_output, min_cellular_output, max_cellular_output = get_minmax_normal_data(cellular_output)

cellular_input_dim = normalized_cellular_input.shape[1]
cellular_output_dim = normalized_cellular_output.shape[1]

cellular_train, cellular_val, cellular_test = get_data_loader(normalized_cellular_input, normalized_cellular_output)
cellular_pth_path= os.path.join(pth_save_addr, 'forward_prediction.pth')


cellular_gan_input = cellular_input
cellular_gan_output = torch.tensor(cellular_output).to(device)
cellular_gan_input = torch.tensor(cellular_gan_input).to(device)

normalized_cellular_gan_input, min_gan_cellular, max_gan_cellular = get_minmax_normal_data(cellular_gan_input)
normalized_cellular_gan_output, min_cellular_gan_output, max_cellular_gan_output = get_minmax_normal_data(cellular_gan_output)

cellular_gan_input_dim = normalized_cellular_gan_input.shape[1]
cellular_gan_output_dim = normalized_cellular_gan_output.shape[1]

cellular_gan_train, cellular_gan_val, cellular_gan_test = get_data_loader(normalized_cellular_gan_input, normalized_cellular_gan_output)
best_g_model_path_cellular = os.path.join(pth_save_addr, 'cellular_best_generator.pth')
best_d_model_path_cellular = os.path.join(pth_save_addr, 'cellular_best_discriminator.pth')
best_r_model_path_cellular = os.path.join(pth_save_addr, 'cellular_best_regressor.pth')

plt.rcParams.update({
    'font.weight': 'bold',          
    'axes.labelweight': 'bold',      
    'axes.titlesize': 26,         
    'axes.labelsize': 18,           
    'xtick.labelsize': 12,          
    'ytick.labelsize': 12,           
    'legend.fontsize': 18,          
    'axes.titleweight': 'bold'        
})