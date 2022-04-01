# %% Imports
from dataset import IMDB_test, IMDB_train
from model import Model_Imdb
import torch
from aum import AUMCalculator
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import csv
import re
import random
import time

LR = 0.001
epochs = 15
batch_size = 64
loss_fn = torch.nn.CrossEntropyLoss()
data_test = IMDB_test()
test_dataloader = torch.utils.data.DataLoader(dataset=data_test,
                                              batch_size=batch_size,
                                              shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = './aum_imdb/'
gamma = 0.9
PR = 1.36
MAX_WORDS = 10000
EMB_SIZE = 128  # embedding size
HID_SIZE = 128  # lstm hidden size
DROPOUT = 0.2
aum_calculator = AUMCalculator(save_dir, compressed=True)
writer = SummaryWriter()


def train_loop(dataloader, model, Loss_fn, Optimizer, record_aum=False):
    size = len(dataloader.dataset)
    index_list = torch.Tensor()
    for batch, (index, X, y) in enumerate(dataloader):
        X.to(device)
        y.to(device)
        Optimizer.zero_grad()
        pred = model(X)
        if record_aum:
            records = aum_calculator.update((pred - pred.mean()) / pred.std(), y, [int(I) for I in index])
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

#%% Train
data_train_full = IMDB_train()
Model_full = Model_Imdb(max_words=MAX_WORDS, emb_size=EMB_SIZE, hid_size=HID_SIZE, dropout=DROPOUT).to(device)
optimizer_full = torch.optim.Adam(Model_full.parameters(), lr=LR)
scheduler_full = torch.optim.lr_scheduler.StepLR(optimizer_full, step_size=1, gamma=gamma)
train_dataloader_full = torch.utils.data.DataLoader(dataset=data_train_full,
                                                    batch_size=batch_size,
                                                    shuffle=True)

data_train_aum = IMDB_train()
Model_aum = Model_Imdb(max_words=MAX_WORDS, emb_size=EMB_SIZE, hid_size=HID_SIZE, dropout=DROPOUT).to(device)
optimizer_aum = torch.optim.Adam(Model_aum.parameters(), lr=LR)
scheduler_aum = torch.optim.lr_scheduler.StepLR(optimizer_aum, step_size=1, gamma=gamma)
train_dataloader_aum = torch.utils.data.DataLoader(dataset=data_train_aum,
                                                     batch_size=batch_size,
                                                     shuffle=True)

full_time = 0
aum_time = 0
for t in range(epochs):
    print(f"Epoch {t + 1}, LR = {optimizer_full.state_dict()['param_groups'][0]['lr']}\n-------------------------------")

    start = time.time()
    aum_calculator = AUMCalculator(save_dir, compressed=True)
    random_seed = 13
    torch.manual_seed(random_seed)
    train_loop(train_dataloader_aum, Model_aum, loss_fn, optimizer_aum,True)
    #sp = split(0,0.83,[j[1] for j in sorted([(aum_calculator.sums[i],i) for i in aum_calculator.sums.keys()])])
    sp = [j[1] for j in [(aum_calculator.sums[i],i) for i in aum_calculator.sums.keys() if aum_calculator.sums[i]<=PR]]
    sp2 = list(set(data_train_full.remain)-set(sp))
    sp+=random.choices(sp2,k=int(len(sp2)*0.1))
    # data_train_aum = Cifar10_train(remain=
    #                                sp[0]+random.choices(sp[1],k=int(len(sp[1])*0.1))
    #                                )
    data_train_aum = IMDB_train(remain=sp)
    train_dataloader_aum = torch.utils.data.DataLoader(dataset=data_train_aum,
                                                       batch_size=batch_size,
                                                       shuffle=True)
    aum_time += time.time()-start
    start = time.time()
    random_seed = 13
    torch.manual_seed(random_seed)
    train_loop(train_dataloader_full, Model_full, loss_fn, optimizer_full)
    full_time+= time.time()-start

    acc_f, loss_f = test_loop(test_dataloader, Model_full, loss_fn)
    acc_f_aum, loss_f_aum = test_loop(test_dataloader, Model_aum, loss_fn)
    writer.add_scalars('ACC', {'full_data': acc_f,
                               'aum': acc_f_aum}, t)
    writer.add_scalars('LOSS', {'full_data': loss_f,
                                'aum': loss_f_aum}, t)
    writer.add_scalars('Time',{'full_data':full_time,'aum':aum_time},t)
    scheduler_full.step()
    scheduler_aum.step()
print("Done!")
writer.close()
