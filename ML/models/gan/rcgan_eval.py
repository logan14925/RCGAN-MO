import torch
import os
import sys
import matplotlib.pyplot as plt

file_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_folder_path)
from train_config import *
import seaborn as sns
import time  # 导入 time 模块
def plot_recon_and_pred(labels, recons, preds, save_path=None):
    num_dimensions = recons[0].shape[1]  # Get the number of features for recon
    fig, axes = plt.subplots(len(labels), num_dimensions + 1, figsize=(30, 0.6 * len(labels) * (num_dimensions + 1)))

    for idx, (label, recon, pred) in enumerate(zip(labels, recons, preds)):
        # Plot recon feature distributions
        # Plot recon feature distributions
        for i in range(num_dimensions):
            axes[idx, i].hist(recon[:, i].cpu().numpy(), bins=25, color='blue', alpha=0.7, density=True)
            sns.kdeplot(recon[:, i].cpu().numpy(), color='blue', ax=axes[idx, i], linewidth=2, label=f'Feature {i}')
            # 去掉 X 和 Y 标签
            axes[idx, i].set(xlabel=None, ylabel=None)

        # # Plot pred_mass distributions
        # axes[idx, -2].hist(pred_mass, bins=30, color='purple', alpha=0.7, density=True)
        # sns.kdeplot(pred_mass, color='purple', ax=axes[idx, -2], linewidth=2, label='Prediction Mass', alpha=0.6)
        # # 去掉 X 和 Y 标签
        # axes[idx, -2].set(xlabel=None, ylabel=None)
        # axes[idx, -2].legend(loc='upper right')

        # Plot preds distributions
        axes[idx, -1].hist(pred, bins=30, color='green', alpha=0.7, density=True)
        sns.kdeplot(pred, color='green', ax=axes[idx, -1], linewidth=2, label='Prediction', alpha=0.6)
        # Add vertical line for label_property value
        axes[idx, -1].axvline(x=label, color='red', linestyle='--', label=f'Label Property (label={label:.2f})')
        # 去掉 X 和 Y 标签
        axes[idx, -1].set(xlabel=None, ylabel=None)
        axes[idx, -1].legend(loc='upper right')
        for ax_row in axes:
            for ax in ax_row:
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))


    
    # Adjust layout
    plt.tight_layout()
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    

# Read configuration and initialize parameters
noisy_dim = config["rcgan_cellular"]["noise_size"]
epochs = config["rcgan_cellular"]["epochs"]
z_dim = config["rcgan_cellular"]["latent_size"]
sample_num = config["rcgan_cellular"]["sample_num"]
# label_propertys = list(np.linspace(0, 0.95, int(0.95 / 0.005)))
label_propertys = [0.1, 0.3, 0.5, 0.8]
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
img_path = None
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

plot_recon_and_pred(label_propertys, recons, preds, img_path)


