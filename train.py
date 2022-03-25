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
import random

random_seed = 13
torch.manual_seed(random_seed)
LR = 0.001
# %% Training and Testing and Recording
record_aum = True
if record_aum:
    save_dir = './aum/'
    aum_calculator = AUMCalculator(save_dir, compressed=True)

writer = SummaryWriter()


def train_loop(dataloader, model, Loss_fn, Optimizer):
    size = len(dataloader.dataset)
    index_list = torch.Tensor()
    # mid_list = torch.Tensor()
    for batch, (index, X, y) in enumerate(dataloader):
        Optimizer.zero_grad()
        index_list = torch.cat([index_list, index])
        pred = model(X)
        # mid_list = torch.cat([mid_list, mid])
        records = aum_calculator.update(pred, y, [int(I) for I in index])
        loss = Loss_fn(pred, y)
        loss.backward()
        Optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return index_list


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
    return 100 * correct, test_loss


def split(left, right, rank):
    return rank[int(left * len(rank)):int(right * len(rank))]


# %% Training
Model = Model_MNIST()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Model.parameters(), lr=0.001)
epochs = 10
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
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)
for t in range(epochs):
    print(f"Epoch {t + 1}, LR = {optimizer.state_dict()['param_groups'][0]['lr']}\n-------------------------------")
    index_list1 = train_loop(train_dataloader, Model, loss_fn, optimizer)
    test_loop(test_dataloader, Model, loss_fn)
    scheduler.step()
if record_aum:
    aum_calculator.finalize()
print("Done!")

# %% AUM analysis + splitting
aum_rank = []
left_threshold = 0.01
right_threshold = 0.5
random_seed = 13
batch_size = 64
epochs = 10
torch.manual_seed(random_seed)
with open('aum/aum_values.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    ii = 0
    for i in reader:
        if ii == 0:
            ii += 1
            continue
        aum_rank.append(int(re.findall(r'\d.*?,', i[0])[0][:-1]))
loss_fn = torch.nn.CrossEntropyLoss()
data_test = MNIST_test()
# all
data_train_f = MNIST_train()
Model_f = Model_MNIST()
optimizer_f = torch.optim.Adam(Model_f.parameters(), lr=LR)

# aum
data_train_f_aum = MNIST_train(remain=split(0.01, 0.51, aum_rank))
Model_f_aum = Model_MNIST()
optimizer_f_aum = torch.optim.Adam(Model_f_aum.parameters(), lr=LR)

# aum with weight
data_train_f_aum_w = MNIST_train(
    remain=random.choices(split(0, 0.5, aum_rank), k=int(len(split(0, 0.5, aum_rank)) * 0.11)) + random.choices(
        split(0.5, 1, aum_rank), k=int(len(split(0.5, 1, aum_rank))*0.89)))
Model_f_aum_w = Model_MNIST()
optimizer_f_aum_w = torch.optim.Adam(Model_f_aum_w.parameters(), lr=LR)

# random
data_train_f_random = MNIST_train(remain=random.choices(aum_rank, k=int(len(aum_rank) / 2)))
Model_f_random = Model_MNIST()
optimizer_f_random = torch.optim.Adam(Model_f_random.parameters(), lr=LR)

data_test_f = MNIST_test()
train_dataloader_f = torch.utils.data.DataLoader(dataset=data_train_f,
                                                 batch_size=batch_size,
                                                 shuffle=True)
train_dataloader_f_aum = torch.utils.data.DataLoader(dataset=data_train_f_aum,
                                                     batch_size=batch_size,
                                                     shuffle=True)
train_dataloader_f_aum_w = torch.utils.data.DataLoader(dataset=data_train_f_aum_w,
                                                       batch_size=batch_size,
                                                       shuffle=True)
train_dataloader_f_random = torch.utils.data.DataLoader(dataset=data_train_f_random,
                                                        batch_size=batch_size,
                                                        shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=data_test,
                                              batch_size=batch_size,
                                              shuffle=True)
scheduler_f = torch.optim.lr_scheduler.StepLR(optimizer_f, step_size=1, gamma=0.75)
scheduler_f_aum = torch.optim.lr_scheduler.StepLR(optimizer_f_aum, step_size=1, gamma=0.75)
scheduler_f_aum_w = torch.optim.lr_scheduler.StepLR(optimizer_f_aum_w, step_size=1, gamma=0.75)
scheduler_f_random = torch.optim.lr_scheduler.StepLR(optimizer_f_random, step_size=1, gamma=0.75)
for t in range(epochs):
    print(f"Epoch {t + 1}, LR = {optimizer_f.state_dict()['param_groups'][0]['lr']}\n-------------------------------")
    train_loop(train_dataloader_f, Model_f, loss_fn, optimizer_f)
    train_loop(train_dataloader_f_aum, Model_f_aum, loss_fn, optimizer_f_aum)
    train_loop(train_dataloader_f_aum_w, Model_f_aum_w, loss_fn, optimizer_f_aum_w)
    train_loop(train_dataloader_f_random, Model_f_random, loss_fn, optimizer_f_random)

    acc_f, loss_f = test_loop(test_dataloader, Model_f, loss_fn)
    acc_f_aum, loss_f_aum = test_loop(test_dataloader, Model_f_aum, loss_fn)
    acc_f_aum_w, loss_f_aum_w = test_loop(test_dataloader, Model_f_aum_w, loss_fn)
    acc_f_random, loss_f_random = test_loop(test_dataloader, Model_f_random, loss_fn)
    writer.add_scalars('ACC', {'full_data': acc_f,
                               'aum': acc_f_aum,
                               'aum_with_random_weight': acc_f_aum_w,
                               'random': acc_f_random}, t)
    writer.add_scalars('LOSS', {'full_data': loss_f,
                                'aum': loss_f_aum,
                                'aum_with_random_weight': loss_f_aum_w,
                                'random': loss_f_random}, t)
    scheduler_f.step()
    scheduler_f_aum.step()
    scheduler_f_aum_w.step()
    scheduler_f_random.step()
print("Done!")
writer.close()
