import torch.nn as nn


class Net1(nn.Module):
    def __init__(self, dim_in=3, class_n=30):
        super(Net1, self).__init__()
        self.linear1 = nn.Conv1d(dim_in, 64, 1, bias=True)
        self.bn1 = nn.BatchNorm1d(64)
        self.linear2 = nn.Conv1d(64, 128, 1, bias=True)
        self.bn2 = nn.BatchNorm1d(128)
        self.linear3 = nn.Conv1d(128, 128, 1, bias=True)
        self.bn3 = nn.BatchNorm1d(128)

        self.gp = nn.AdaptiveAvgPool1d(1)

        self.linear4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.linear5 = nn.Linear(64, class_n)
        self.act = nn.ReLU()                       # 激活函数

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()  # 即将第1维和第2维交换位置
        x = self.act(self.bn1(self.linear1(x)))
        x = self.act(self.bn2(self.linear2(x)))
        x = self.act(self.bn3(self.linear3(x)))
        x = self.gp(x)  # 平均池化
        x = x.squeeze(-1)  # 压缩维度
        x = self.act(self.bn4(self.linear4(x)))
        x = self.linear5(x)
        return x


class Net2(nn.Module):
    def __init__(self, dim_in=6, class_n=30):
        super(Net2, self).__init__()
        self.linear1 = nn.Conv1d(dim_in, 64, 1, bias=True)
        self.bn1 = nn.BatchNorm1d(64)
        self.linear2 = nn.Conv1d(64, 128, 1, bias=True)
        self.bn2 = nn.BatchNorm1d(128)
        self.linear3 = nn.Conv1d(128, 128, 1, bias=True)
        self.bn3 = nn.BatchNorm1d(128)

        self.gp = nn.AdaptiveAvgPool1d(1)

        self.linear4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.linear5 = nn.Linear(64, class_n)
        self.act = nn.ReLU()                       # 激活函数

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()  # 即将第1维和第2维交换位置
        x = self.act(self.bn1(self.linear1(x)))
        x = self.act(self.bn2(self.linear2(x)))
        x = self.act(self.bn3(self.linear3(x)))

        x = self.gp(x)  # 平均池化
        x = x.squeeze(-1)  # 压缩维度

        x = self.act(self.bn4(self.linear4(x)))
        x = self.linear5(x)
        return x


class Net3(nn.Module):
    def __init__(self, dim_in=1, class_n=30):
        super(Net3, self).__init__()
        self.linear1 = nn.Conv1d(dim_in, 64, 1, bias=True)
        self.bn1 = nn.BatchNorm1d(64)
        self.linear2 = nn.Conv1d(64, 128, 1, bias=True)
        self.bn2 = nn.BatchNorm1d(128)
        self.linear3 = nn.Conv1d(128, 128, 1, bias=True)
        self.bn3 = nn.BatchNorm1d(128)

        self.gp = nn.AdaptiveAvgPool1d(1)

        self.linear4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.linear5 = nn.Linear(64, class_n)
        self.act = nn.ReLU()                       # 激活函数

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()  # 即将第1维和第2维交换位置
        x = self.act(self.bn1(self.linear1(x)))
        x = self.act(self.bn2(self.linear2(x)))
        x = self.act(self.bn3(self.linear3(x)))

        x = self.gp(x)  # 平均池化
        x = x.squeeze(-1)  # 压缩维度

        x = self.act(self.bn4(self.linear4(x)))
        x = self.linear5(x)
        return x






