import torch
from torch import nn
import config

# class BehavioralModel(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
#         # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(in_features=27, out_features=128)
#         self.fc2 = nn.Linear(in_features=128, out_features=256)
#         self.fc3 = nn.Linear(in_features=256, out_features=128)
#         self.fc4 = nn.Linear(in_features=128, out_features=4)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         x = self.fc4(x)
#         # print('1:', x.shape)
#         x = nn.functional.softmax(x, dim=1)
#         # print('2:', x.shape)
        
#         return x
    

class BehavioralModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(in_features=27, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        # self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=4)
        self.relu = nn.ReLU()

        # 初始化权重（可选，但推荐）
        self._init_weights()

    def _init_weights(self):
        # 使用ReLU激活函数，推荐使用kaiming初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc4(x)  # 输出层不加激活函数
        return x