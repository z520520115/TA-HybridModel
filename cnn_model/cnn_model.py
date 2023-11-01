import torch.nn as nn
import torch.nn as tnn
import torch.nn.functional as F

# CNN Module
def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        tnn.BatchNorm2d(chann_out),
        tnn.ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    layers += [tnn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return tnn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Linear(size_in, size_out),
        tnn.BatchNorm1d(size_out),
        tnn.ReLU()
    )
    return layer

class VGG16(tnn.Module):
    def __init__(self, n_classes=1000):
        super(VGG16, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(7 * 7 * 512, 4096)
        self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        self.layer8 = tnn.Linear(4096, n_classes)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        vgg16_features = self.layer5(x)
        x = vgg16_features.view(x.size(0), -1)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        # print("Vgg第八层shape: " + str(x.shape))
        return x.to('cuda')

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # 卷积层1：输入是224*224*3 计算(224-5)/1+1=220 conv1输出的结果是220
        self.conv1 = nn.Conv2d(3, 6, 5)  # input:3 output6 kernel:5
        # 池化层1：输入是220*220*6 窗口2*2  计算(220-0)/2=110 通过max_pooling层输出的是110*110*6
        self.pool = nn.MaxPool2d(2, 2)
        # 卷积层2， 输入是220*220*6，计算（110 - 5）/ 1 + 1 = 106，通过conv2输出的结果是106*106*16
        self.conv2 = nn.Conv2d(6, 16, 5)  # input:6, output:16, kernel:5

        self.fc1 = nn.Linear(16 * 53 * 53, 1024)  # input:16*53*53, output:1024
        self.fc2 = nn.Linear(1024, 1000)  # input:1024, output:512

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        return x.to('cuda')
