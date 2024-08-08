import torch
import torch.nn as nn
import torch.nn.functional as F

class CircleDetector(nn.Module):
    def __init__(self):
        super(CircleDetector, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(in_features=64*16*16, out_features=128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(in_features=128, out_features=3)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.bn_fc1(self.fc1(x)))
        x = self.fc2(x)
        c_x, c_y, r = torch.sigmoid(x[:, 0]), torch.sigmoid(x[:, 1]), torch.sigmoid(x[:, 2])
        return c_x, c_y, r
class RectangleDetector(torch.nn.Module):
    def __init__(self):
        super(RectangleDetector, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(in_features=64*16*16, out_features=128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(in_features=128, out_features=4)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.bn_fc1(self.fc1(x)))
        x = self.fc2(x)
        t_x, t_y,b_x, b_y = torch.sigmoid(x[:, 0]), torch.sigmoid(x[:, 1]), torch.sigmoid(x[:, 2]), torch.sigmoid(x[:, 3])
        return t_x, t_y, b_x, b_y

class Inside(nn.Module):
    def __init__(self):
        super(Inside, self).__init__()

    def forward(self, c1, c2, r, xbl, ybl, xtr, ytr):
        smooth_lt_xbl = torch.sigmoid(10 * (c1 - (xbl + r)))
        smooth_gt_xtr = torch.sigmoid(10 * ((xtr - r) - c1))
        smooth_lt_ybl = torch.sigmoid(10 * (c2 - (ybl + r)))
        smooth_gt_ytr = torch.sigmoid(10 * ((ytr - r) - c2))
        return smooth_lt_xbl * smooth_gt_xtr * smooth_lt_ybl * smooth_gt_ytr

    def __call__(self, *args):
        return self.forward(*args)

class Outside(nn.Module):
    def __init__(self):
        super(Outside, self).__init__()

    def forward(self, c1, c2, r, xbl, ybl, xtr, ytr):
        smooth_gt_xbl = torch.sigmoid(10 * ((xbl - r) - c1))
        smooth_lt_xtr = torch.sigmoid(10 * (c1 - (xtr + r)))
        smooth_gt_ybl = torch.sigmoid(10 * ((ybl - r) - c2))
        smooth_lt_ytr = torch.sigmoid(10 * (c2 - (ytr + r)))
        return smooth_gt_xbl + smooth_lt_xtr + smooth_gt_ybl + smooth_lt_ytr - \
               smooth_gt_xbl * smooth_lt_xtr * smooth_gt_ybl * smooth_lt_ytr

    def __call__(self, *args):
        return self.forward(*args)

class Intersect(nn.Module):
    def __init__(self):
        super(Intersect, self).__init__()

    def forward(self, c1, c2, r, xbl, ybl, xtr, ytr):
        smooth_lt_xbl = torch.sigmoid(10 * (c1 + r - xbl))
        smooth_gt_xtr = torch.sigmoid(10 * (xtr - (c1 - r)))
        smooth_lt_ybl = torch.sigmoid(10 * (c2 + r - ybl))
        smooth_gt_ytr = torch.sigmoid(10 * (ytr - (c2 - r)))
        return smooth_lt_xbl * smooth_gt_xtr * smooth_lt_ybl * smooth_gt_ytr

    def __call__(self, *args):
        return self.forward(*args)