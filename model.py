import torch
import torch.nn as nn
import torch.nn.functional as F


class Model_MNIST(torch.nn.Module):

    def __init__(self):
        super(Model_MNIST, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2),  # padding=2保证输入输出尺寸相同
            nn.ReLU(),  # input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),  # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(16 * 5 * 5, 120),
            torch.nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(120, 64),
            nn.ReLU()
        )
        self.fc3 = torch.nn.Sequential(torch.nn.Linear(64, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class Model_Cifar10(nn.Module):  # 我们定义网络时一般是继承的torch.nn.Module创建新的子类
    def __init__(self):
        super(Model_Cifar10, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)  # 卷积层,默认padding=0,stride=1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层
        self.conv2 = nn.Conv2d(6, 16, 5)  # 卷积层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 全连接层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):  # 前向传播，反向传播涉及到torch.autograd模块
        x = self.pool(F.relu(self.conv1(x)))  # F是torch.nn.functional的别名，这里调用了relu函数 F.relu()
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Model_Imdb(nn.Module):
    def __init__(self, max_words, emb_size, hid_size, dropout):
        super(Model_Imdb, self).__init__()
        self.max_words = max_words
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.dropout = dropout
        self.Embedding = nn.Embedding(self.max_words, self.emb_size)
        self.LSTM = nn.LSTM(self.emb_size, self.hid_size, num_layers=2,
                            batch_first=True, bidirectional=True)  # 2层双向LSTM
        self.dp = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(self.hid_size * 2, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, 2)

    def forward(self, x):
        """
        input : [bs, maxlen]
        output: [bs, 2]
        """
        x = self.Embedding(x)  # [bs, ml, emb_size]
        x = self.dp(x)
        x, _ = self.LSTM(x)  # [bs, ml, 2*hid_size]
        x = self.dp(x)
        x = F.relu(self.fc1(x))  # [bs, ml, hid_size]
        x = F.avg_pool2d(x, (x.shape[1], 1)).squeeze()  # [bs, 1, hid_size] => [bs, hid_size]
        out = self.fc2(x)  # [bs, 2]
        return out  # [bs, 2]
