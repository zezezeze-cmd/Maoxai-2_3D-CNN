# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn

class MaoXia3DNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(MaoXia3DNet, self).__init__()
        self.conv3dv1 = nn.Conv3d(100, 200, (3, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.bn1 = nn.BatchNorm3d(200)
        self.relu1 = nn.PReLU()
        self.conv3dv2 = nn.Conv2d(200, 300, (3, 3), stride=(2, 2), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(300)
        self.relu2 = nn.PReLU()
        self.conv3dv3 = nn.Conv2d(300, 400, (3, 3), stride=(2, 2), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(400)
        self.relu3 = nn.PReLU()
        self.conv3dv4 = nn.Conv2d(400, 600, (3, 3), stride=(2, 2), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(600)
        self.relu4 = nn.PReLU()
        self.conv3dv5 = nn.Conv2d(600, 800, (3, 3), stride=(2, 2), padding=(1, 1))
        self.bn5 = nn.BatchNorm2d(800)
        self.relu5 = nn.PReLU()
        self.conv3dv6 = nn.Conv2d(800, 1000, (3, 3), stride=(2, 2), padding=(1, 1))
        self.bn6 = nn.BatchNorm2d(1000)
        self.relu6 = nn.PReLU()
        self.conv3dv7 = nn.Conv2d(1000, 1000, (3, 3), stride=(2, 2), padding=(1, 1))
        self.bn7 = nn.BatchNorm2d(1000)
        self.relu7 = nn.PReLU()
        # self.conv3dv8 = nn.Conv2d(1000, 1000, (3, 3), stride=(2, 2), padding=(1, 1))
        # self.bn8 = nn.BatchNorm2d(1000)
        # self.relu8 = nn.PReLU()
        # self.conv3dv9 = nn.Conv2d(1000, 1000, (1, 1), stride=(1, 1), padding=(0, 0))
        # # self.bn9 = nn.BatchNorm3d(5000)
        # self.relu9 = nn.PReLU()
        # self.relu1 = nn.ReLU()
        self.fc_1 = nn.Linear(1000*2*2, 1000)
        self.bn_fc_1 = nn.BatchNorm1d(1000)
        self.relu_fc_1 = nn.PReLU()
        self.fc = nn.Linear(1000, num_classes)
        self.bn_fc = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        out = self.bn1(self.relu1(self.conv3dv1(x)))

        out = out.view(out.size(0), out.size(1),out.size(3), out.size(4))

        out = self.bn2(self.relu2(self.conv3dv2(out)))
        out = self.bn3(self.relu3(self.conv3dv3(out)))
        out = self.bn4(self.relu4(self.conv3dv4(out)))
        out = self.bn5(self.relu5(self.conv3dv5(out)))
        out = self.bn6(self.relu6(self.conv3dv6(out)))
        out = self.bn7(self.relu7(self.conv3dv7(out)))

        # out = self.bn8(self.relu8(self.conv3dv8(out)))
        print(out.shape)
        out = out.view(out.size(0), -1)

        # out = self.relu9(self.conv3dv9(out))

        # out = self.relu1(out)
        out = self.bn_fc_1(self.relu_fc_1(self.fc_1(out)))
        print(out.shape)
        out = self.fc(out)
        out = self.bn_fc(out)
        print(out.shape)  # (batch_size,3,256,256)

        return out  # x[batch_size, feature_size]


if __name__ == '__main__':
    frames = torch.randn(2, 100, 3, 256, 256)
    myNet = MaoXia3DNet(5)
    myNet.forward(frames)