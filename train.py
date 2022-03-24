# %% Imports
from dataset import MNIST_test, MNIST_train
from model import Model_MNIST
import torch
from aum import AUMCalculator
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import csv
import re
random_seed = 1
torch.manual_seed(random_seed)
# %% Training and Testing and Recording
record_aum = True
if record_aum:
    save_dir = './aum/'
    aum_calculator = AUMCalculator(save_dir, compressed=True)
writer = SummaryWriter('./log/')


def train_loop(epoch, dataloader, model, loss_fn, Optimizer, random_delete=False, batch_data=None):
    size = len(dataloader.dataset)
    index_list = torch.Tensor()
    mid_list = torch.Tensor()
    for batch, (index, X, y) in enumerate(dataloader):
        index_list = torch.cat([index_list, index])
        # Compute prediction and loss
        pred = model(X)
        # mid_list = torch.cat([mid_list, mid])
        records = aum_calculator.update(pred, y, [int(i) for i in index])
        loss = loss_fn(pred, y)
        # Backpropagation
        Optimizer.zero_grad()
        loss.backward()
        Optimizer.step()
        if random_delete:
            if batch >= (size * batch_data['%'] / batch_data['batch_size']):
                print(f"{batch_data['%'] * 100}% data. [{batch * len(X):>5d}/{size:>5d}]")
                break
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    # writer.add_scalar('TRAIN_LOSS', loss, epoch)
    return index_list, mid_list


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for index, X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


Model = Model_MNIST()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Model.parameters(), lr=0.001)
epochs = 4
data_train = MNIST_train()
data_test = MNIST_test()
len_train = len(data_train)
batch_size = 64
delete = False
batch_data = {
    "%": 0.5,
    "batch_size": batch_size
}
train_dataloader = torch.utils.data.DataLoader(dataset=data_train,
                                               batch_size=batch_size,
                                               shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=data_test,
                                              batch_size=batch_size,
                                              shuffle=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.3)
for t in range(epochs):
    print(f"Epoch {t + 1}, LR = {optimizer.state_dict()['param_groups'][0]['lr']}\n-------------------------------")
    index_list1, mid_list1 = train_loop(t, train_dataloader, Model, loss_fn, optimizer, delete, batch_data)
    test_loop(test_dataloader, Model, loss_fn)
    scheduler.step()
if record_aum:
    aum_calculator.finalize()
print("Done!")


# %% AUM analysis + splitting
aum_rank = []
left_threshold = 0.01
right_threshold = 0.5
with open('aum/aum_values.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    ii=0
    for i in reader:
        if ii==0:
            ii+=1
            continue
        aum_rank.append(int(re.findall(r'\d.*?\)',i[0])[0][:-1]))
def split(left,right,rank):
    return rank[int(left*len(rank)):int(right*len(rank))]
Model_f = Model_MNIST()
loss_fn_f = torch.nn.CrossEntropyLoss()
optimizer_f = torch.optim.Adam(Model.parameters(), lr=0.001)
data_train_f = MNIST_train(remain=split(left_threshold,right_threshold,aum_rank))
data_test_f = MNIST_test()
len_train_f = len(data_train_f)
delete_f = False
batch_data_f = {
    "%": 0.5,
    "batch_size": batch_size
}
train_dataloader_f = torch.utils.data.DataLoader(dataset=data_train_f,
                                               batch_size=batch_size,
                                               shuffle=True)
scheduler_f = torch.optim.lr_scheduler.StepLR(optimizer_f, step_size=1, gamma=0.3)
for t in range(epochs):
    print(f"Epoch {t + 1}, LR = {optimizer_f.state_dict()['param_groups'][0]['lr']}\n-------------------------------")
    index_list_f, mid_list1_f = train_loop(t, train_dataloader_f, Model_f, loss_fn_f, optimizer_f, delete_f, batch_data_f)
    test_loop(test_dataloader, Model_f, loss_fn_f)
    scheduler_f.step()
print("Done!")