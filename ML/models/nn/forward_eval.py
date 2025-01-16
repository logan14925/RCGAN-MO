import torch
import os
import sys
import time  # 导入 time 模块
file_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_folder_path)
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt
from train_config import *
import random
from sklearn.metrics import r2_score

seed = 51
torch.manual_seed(seed)

all_predictions = []
all_targets = []

# Initialize variable to track total time
total_time = 0

# Extract results from 30 batches
for _ in range(30):
    random_batch = random.choice(list(iter(cellular_train)))
    x, targets = random_batch
    forward_net = forward_regression(cellular_input_dim, cellular_output_dim).to(device)
    forward_net.load_state_dict(torch.load(cellular_pth_path))
    forward_net.eval()
    x = x.to(device)
    targets = targets.to(device)
    
    # Record start time
    start_time = time.time()
    
    # Use the model for prediction
    with torch.no_grad():
        prediction = forward_net(x)
    
    # Record end time and calculate the time taken for this batch
    end_time = time.time()
    total_time += (end_time - start_time)  # Accumulate the total time
    print('total time is ', total_time)
    
    # Append predictions and targets to the lists
    all_predictions.append(prediction.cpu().numpy())
    all_targets.append(targets.cpu().numpy())

# Concatenate predictions and targets from all batches
all_predictions = np.concatenate(all_predictions)
all_targets = np.concatenate(all_targets)

r2 = r2_score(all_targets, all_predictions)

print('R² Score:', r2)
real_preds = [d_minmax_normal(torch.tensor(pred, device=device), min_cellular_output, max_cellular_output) for pred in all_predictions]
real_tragets = [d_minmax_normal(torch.tensor(target, device=device), min_cellular_output, max_cellular_output) for target in all_targets]
real_preds_np = [pred.cpu().numpy() for pred in real_preds]
real_tragets_np = [target.cpu().numpy() for target in real_tragets]

plt.figure(figsize=(10, 8))

# Scatter plot for load prediction vs. target with solid blue circles (RGB = 20, 81, 124)
blue_color = (225/255, 0/255, 0/255)  # RGB normalized to [0, 1] range
plt.scatter(real_tragets_np, real_preds_np, color='black', label='Forward surrogate model', marker='o', alpha=0.7)

# plt.title('Forward Predictions vs. Targets', pad=10)
plt.xlabel('Targets', fontweight='bold', labelpad=1)
plt.ylabel('Predictions', fontweight='bold', labelpad=1)


# Plot y=x line with black color and dashed line
plt.plot([0.025, 0.075],
         [0.025, 0.075],
         color='black', linestyle='--', linewidth=2)  # Black dashed line with thicker linewidth
plt.legend()
plt.tight_layout()
plt.show()

# Print total time taken for forward passes
print(f"Total time taken for forward passes: {total_time:.4f} seconds")
