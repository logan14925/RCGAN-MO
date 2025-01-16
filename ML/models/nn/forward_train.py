import torch
import os
import sys
file_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_folder_path)
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt
from train_config import *


forward_regression_net = forward_regression(cellular_input_dim, cellular_output_dim).to(device)


lr = config["cellular_forward"]["lr"]
num_epochs = config["cellular_forward"]["epochs"]


optimizer = torch.optim.SGD(forward_regression_net.parameters(), lr=lr, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=3, verbose=True, min_lr=1e-20)


loss_func = torch.nn.MSELoss()
total_train_step = 0
train_losses = []
now_loss = []
test_loss_list = []
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
plt.ion()  # Turn on interactive mode
plt.show()


start_time = time.time()


for i in range(num_epochs):
    # Print the start of the current training epoch
    print("-------The {}th round of training starts-------".format(i+1))


    forward_regression_net.train() 
    # Change the dataset here
    for data in cellular_train:
        inputs, targets = data
        inputs = inputs.float().to(device)
        targets = targets.float().to(device)


        optimizer.zero_grad()
        prediction = forward_regression_net(inputs)
        loss = loss_func(prediction, targets)
        # print('prediction is {}\ntarget is {}'.format(prediction, targets))
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            # Print the training step and the current loss
            print("Training step: {}, Loss:{}".format(total_train_step, loss.item()))
            # print('prediction is {}\ntarget is {}'.format(prediction, targets))
    train_losses.append(loss.item())
    print(train_losses[-1])
    if len(train_losses) >= 10:
        now_loss = train_losses[-5:]
        now_average_loss = sum(now_loss) / len(now_loss)
        scheduler.step(now_average_loss)


    val_losses = []
    forward_regression_net.eval()


    with torch.no_grad():
        for data in cellular_val:
            inputs, targets = data
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)


            prediction = forward_regression_net(inputs)
            # if i == num_epochs -1:
            #     print('pre = {}'.format(prediction), 'target = {}'.format(targets), '\n')
            loss = loss_func(prediction, targets)
            val_losses.append(loss.item())


    average_test_loss = sum(val_losses) / len(val_losses)
    test_loss_list.append(average_test_loss)
    # Print the loss on the test set
    print("Test set loss: {}".format(average_test_loss))


end_time = time.time()
times = end_time - start_time
print(times)


plt.figure(figsize=(10, 8))
plt.plot(train_losses, label="Training Loss")
plt.plot(test_loss_list, label="Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.title('Training and Test Loss Curve')
plt.legend()
plt.tight_layout()
plt.show()


train_losses = torch.tensor(train_losses)
train_losses = torch.unsqueeze(train_losses, dim=1)


best_loss, index = torch.min(train_losses, dim=0)


best_loss_epoch = index + 1


best_model = forward_regression(cellular_input_dim, cellular_output_dim).to(device)
best_model.load_state_dict(forward_regression_net.state_dict())
torch.save(best_model.state_dict(), cellular_pth_path)


# Print the information of the saved model with the lowest loss
print(f"The model with the lowest loss has been saved as 'MaterialNet_model_best.pth', the loss value is: {best_loss}, and the corresponding epoch is: {best_loss_epoch}")