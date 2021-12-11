import torch
import torchvision.models as models
import torch.nn.functional as F

class Ball_Conv(torch.nn.Module):
    def __init__(self, cin, cout, k, s, activation):
        super(Ball_Conv, self).__init__()
        self.conv1_1 = torch.nn.Conv1d(in_channels=cin, out_channels=cout,kernel_size=k,stride=s,padding=4) 
        self.bn1 = torch.nn.BatchNorm1d(cout)
        self.relu_1 = torch.nn.ReLU()
    def forward(self, x):
        ##print("input shape: ", x.shape)
        out = self.relu_1(self.bn1(self.conv1_1(x)))
        ##print("output shape", out.shape)
        return out
class Depthwise_conv1d(torch.nn.Module):
    def __init__(self, cin, cout, k, s, g, activation):
        super(Depthwise_conv1d, self).__init__()
        self.convp_1 = torch.nn.Conv1d(in_channels=cin, out_channels=cout,kernel_size=1,stride=1)
        self.bn1 = torch.nn.BatchNorm1d(cout)
        self.relu_1 = torch.nn.ReLU()
        self.dwconv = torch.nn.Conv1d(in_channels=cout, out_channels=cout,kernel_size=k,stride=s,groups=g,padding=4) 
        self.bn2 = torch.nn.BatchNorm1d(cout)
        self.relu_2 = torch.nn.ReLU()
        self.convp_2 = torch.nn.Conv1d(in_channels=cout, out_channels=cout,kernel_size=1,stride=1)
        self.bn3 = torch.nn.BatchNorm1d(cout)
        self.relu_3 = torch.nn.ReLU()
    def forward(self, x):
        out = self.relu_1(self.bn1(self.convp_1(x)))
        #print("pw shape: ", out.shape)
        out = self.relu_2(self.bn2(self.dwconv(out)))
        #print("dw shape: ", out.shape)
        out = self.relu_3(self.bn3(self.convp_2(out)))
        #print("pw shape: ", out.shape)
        return out

class glob_Conv_49(torch.nn.Module):
    def __init__(self, cin, cout,activation):
        super(glob_Conv_49, self).__init__()
        self.conv1_1 = torch.nn.Conv1d(in_channels=cin, out_channels=cout,kernel_size=49,stride=1) 
        self.bn1 = torch.nn.BatchNorm1d(cout)
        self.relu_1 = torch.nn.ReLU()
    def forward(self, x):
        ##print("input shape: ", x.shape)
        out = self.relu_1(self.bn1(self.conv1_1(x)))
        ##print("output shape", out.shape)
        return out

class Ball_Depthwise_net(torch.nn.Module):
    def __init__(self,num_classes):
        super(Ball_Depthwise_net,self).__init__()
        self.num_classes = num_classes
        self.dpconv1 = Depthwise_conv1d(cin=9, cout=32, k=9, s=4, g=32, activation='relu')
        self.dpconv2 = Depthwise_conv1d(cin=32, cout=64, k=9, s=4, g=64, activation='relu')
        self.dpconv3 = Depthwise_conv1d(cin=64, cout=128, k=9, s=4, g=128, activation='relu')
        self.dpconv4 = Depthwise_conv1d(cin=128, cout=256, k=9, s=4, g=256, activation='relu')
        self.dpconv5 = Depthwise_conv1d(cin=256, cout=512, k=9, s=4, g=512, activation='relu')
        self.dpconv6 = Depthwise_conv1d(cin=512, cout=512, k=9, s=4, g=512, activation='relu')
        self.dpconv7 = Depthwise_conv1d(cin=512, cout=512, k=9, s=4, g=512, activation='relu')
        self.dpconv8 = Depthwise_conv1d(cin=512, cout=512, k=9, s=4, g=512, activation='relu')
        #self.dense1 = torch.nn.Linear(512, 128)
        #self.relu_2 = torch.nn.ReLU()
        self.last = torch.nn.Linear(512, num_classes)
        self.bn1 = torch.nn.BatchNorm1d(num_classes)
        #self.softmax = torch.nn.Softmax(dim=1)

    def freeze(self,model_ft):
        ct = 0
        for child in model_ft.children():
            for param in child.parameters():
                param.requires_grad = False
        return model_ft

    def forward(self,x):
        out = self.dpconv1(x)
        #print("out 1: ", out.shape)
        out = self.dpconv2(out)
        #print("out 2: ", out.shape)
        out = self.dpconv3(out)
        #print("out 3: ", out.shape)
        out = self.dpconv4(out)
        #print("out 4: ", out.shape)
        out = self.dpconv5(out)
        #print("out 5: ", out.shape)
        out = self.dpconv6(out)
        #print("out 6: ", out.shape)
        out = self.dpconv7(out)
        #print("out 7: ", out.shape)
        out = self.dpconv8(out)
        #print("out 8: ", out.shape)
        out = out.view(out.size(0), -1)
        out = self.bn1(self.last(out))
        #print("out 6: ", out.shape)
        return out

    def get_model(self):
        return self.last

    def show_model(self):
        print(self.last)




# test Res
if __name__ == "__main__":
    model = models.resnet101(pretrained=True)
    num_classes = 1000
    resnet = Resnet(model, num_classes)
    resnet.show_model()
