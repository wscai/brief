from dataset import Cifar10_test, Cifar10_train
from model import Model_Cifar10
import torch
from aum import AUMCalculator
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import random
import time


def compare():
    max_acc_aum = []
    max_acc_full = []
    min_loss_aum = []
    min_loss_full = []
    LR = 0.001
    epochs = 15
    batch_size = 64
    loss_fn = torch.nn.CrossEntropyLoss()
    data_test = Cifar10_test()
    test_dataloader = torch.utils.data.DataLoader(dataset=data_test,
                                                  batch_size=batch_size,
                                                  shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = './aum/'
    aum_calculator = AUMCalculator(save_dir, compressed=True)
    gamma = 0.9
    PR = 1.34

    def train_loop(dataloader, model, Loss_fn, Optimizer, record_aum=False):
        size = len(dataloader.dataset)
        index_list = torch.Tensor()
        # mid_list = torch.Tensor()
        for batch, (index, X, y) in enumerate(dataloader):
            X.to(device)
            y.to(device)
            Optimizer.zero_grad()
            index_list = torch.cat([index_list, index])
            pred = model(X)
            if record_aum:
                records = aum_calculator.update(
                    (pred - pred.mean()) / pred.std(), y,
                    [int(I) for I in index])
            loss = Loss_fn(pred, y)
            loss.backward()
            Optimizer.step()
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
        return 100 * correct, test_loss

    data_train_full = Cifar10_train()
    Model_full = Model_Cifar10().to(device)
    optimizer_full = torch.optim.Adam(Model_full.parameters(), lr=LR)
    scheduler_full = torch.optim.lr_scheduler.StepLR(optimizer_full, step_size=1, gamma=gamma)
    train_dataloader_full = torch.utils.data.DataLoader(dataset=data_train_full,
                                                        batch_size=batch_size,
                                                        shuffle=True)
    data_train_aum = Cifar10_train()
    Model_aum = Model_Cifar10().to(device)
    optimizer_aum = torch.optim.Adam(Model_aum.parameters(), lr=LR)
    scheduler_aum = torch.optim.lr_scheduler.StepLR(optimizer_aum, step_size=1, gamma=gamma)
    train_dataloader_aum = torch.utils.data.DataLoader(dataset=data_train_aum,
                                                       batch_size=batch_size,
                                                       shuffle=True)

    full_time = 0
    aum_time = 0
    for t in range(epochs):
        start = time.time()
        aum_calculator = AUMCalculator(save_dir, compressed=True)
        train_loop(train_dataloader_aum, Model_aum, loss_fn, optimizer_aum, True)
        aum_time += time.time() - start
        sp = [j[1] for j in
              [(aum_calculator.sums[i], i) for i in aum_calculator.sums.keys() if aum_calculator.sums[i] <= PR]]
        sp2 = list(set(data_train_full.remain) - set(sp))
        sp += random.choices(sp2, k=int(len(sp2) * 0.1))
        data_train_aum = Cifar10_train(remain=sp)
        train_dataloader_aum = torch.utils.data.DataLoader(dataset=data_train_aum,
                                                           batch_size=batch_size,
                                                           shuffle=True)

        start = time.time()
        train_loop(train_dataloader_full, Model_full, loss_fn, optimizer_full)
        full_time += time.time() - start
        acc_f_aum, loss_f_aum = test_loop(test_dataloader, Model_aum, loss_fn)
        acc_f, loss_f = test_loop(test_dataloader, Model_full, loss_fn)
        max_acc_aum.append(acc_f_aum)
        max_acc_full.append(acc_f)
        min_loss_aum.append(loss_f_aum)
        min_loss_full.append(loss_f)
        scheduler_full.step()
        scheduler_aum.step()
    print('done!')
    return max(max_acc_full),max(max_acc_aum),min(min_loss_full),min(min_loss_aum),full_time,aum_time

#%% main_loop
max_acc_f = []
max_acc_a = []
min_loss_f = []
min_loss_a = []
time_f = []
time_a = []
for i in range(100):
    print(i)
    a,b,c,d,e,f = compare()
    print(f'full: {a}% {c} {e}; aum: {b}%, {d} {f}')
    max_acc_a.append(b)
    max_acc_f.append(a)
    min_loss_a.append(d)
    min_loss_f.append(c)
    time_f.append(e)
    time_a.append(f)
