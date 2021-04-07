# ---------------------------------------------------------------------------- #
# An implementation of https://arxiv.org/pdf/1512.03385.pdf                    #
# See section 4.2 for the model architecture on CIFAR-10                       #
# Some part of the code was referenced from below                              #
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py   #
# ---------------------------------------------------------------------------- #

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration  GPU设置（没有GPU就用cpu进行运算）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 1
learning_rate = 0.001

# Image preprocessing modules图像预处理模块，用于扩增数据（样本多样性）
transform = transforms.Compose([
    transforms.Pad(4),                    #填充
    transforms.RandomHorizontalFlip(),    #图像一半的概率翻转，一半的概率不翻转
                                          # 功能：依据概率p对PIL图片进行垂直翻转
                                          # 参数：p- 概率，默认值为0.5
    transforms.RandomCrop(32),            #随即裁剪32*32
    transforms.ToTensor()])               #将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       b                                             transform=transform,

# CIFAR-10 dataset
#  训练数据集
path='E:\神经网络\资料\数据集'
train_dataset = torchvision.datasets.CIFAR10(root=path,
                                             train=True,
                                             transform=transform,
                                             download=True)

# 测试数据集
test_dataset = torchvision.datasets.CIFAR10(root=path,
                                            train=False,
                                            transform=transforms.ToTensor())

# Data loader
# 训练数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,
                                           shuffle=True)

# 测试数据加载器
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False)

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)
    # 输入深度（通道）,输出深度,滤波器（过滤器）大小为3*3,步长，默认为1, 0填充一层,不设偏置

#init()方法是所谓的对象的“构造函数”，负责在对象初始化时进行一系列的构建操作
# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride) # 3*3卷积层
        self.bn1 = nn.BatchNorm2d(out_channels)  # 批标准化
        self.relu = nn.ReLU(inplace=True)        # 激活函数
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample # shortcut操作

    def forward(self, x):
        residual = x             # 获得上一层的输出
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)     # 连接一个激活函数
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:      # 当shortcut存在
            residual = self.downsample(x)
            # 将上一层的输出x输入进这个downsample所拥有一些操作（卷积等），将结果赋给residual
            # 简单说，这个目的就是为了应对上下层输出输入深度不一致问题
        out += residual             # 将bn2的输出和shortcut过来加在一起
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):        # 创建类中的函数，也叫方法
        # block:为上边的基础块ResidualBlock，它其实就是一个对象
        # layers:每个大layer中的block个数，设为blocks更好，但每一个block实际上也很是一些小layer
        # num_classes:表示最终分类的种类数
        super(ResNet, self).__init__()    #首先找到 ResNet 的父类（就是类 nn.Module），然后把类 ResNet 的对象转换为类 nn.Module 的对象
        self.in_channels = 64
        self.conv = conv3x3(3, 64)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, layers[0])   # 特征图大小不变
        self.layer2 = self.make_layer(block, 128, layers[1], 2)
        self.layer3 = self.make_layer(block, 256, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)   # 平均池化
        self.fc = nn.Linear(256, num_classes)# 全连接层
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        # out_channels表示的是这个块中3*3卷积层的输入输出深度
        downsample = None     # shortcut内部的跨层实现
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []               #创建一个空层列表
        layers.append(block(self.in_channels, out_channels, stride, downsample)) # block()生成上面定义的基础块的对象，并将dowsample传递给block
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))#一定要注意，out_channels一直都是3*3卷积层的深度
        return nn.Sequential(*layers) # 这里表示将layers中的所有block按顺序接在一起
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)  # view函数将张量x变形成一维向量形式，总特征数不变，为全连接层做准备...将每个批次中的每一个输入都拉成一个维度
                                         # 池化层输出维度为4的batch；即（batchsize，channels，x，y）；out.size(0)是指batchsize的值，保留维度
                                         # self.view(out.size(0), -1)指转换有几行，-1指在不告诉函数有多少列的情况下，根据原tensor数据和batchsize自动分配列数。
        out = self.fc(out)               # 全连接层
        return out

# 实例化一个残差网络模型
model = ResNet(ResidualBlock, [5, 5, 5]).to(device)
# block对象为 基础块ResidualBlock
# layers列表为 [5,5,5]，这表示网络中每个大layer阶段都是由5个ResidualBlock组成

def restore_params():
    model.load_state_dict(torch.load('resnet1.ckpt'))

print(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# Test the model
# 设置为评估模式
model.eval()#pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值。
# 不然的话，一旦test的batch_size过小，很容易就会被BN层导致生成图片颜色失真极大；在模型测试阶段使用
#训练完train_datasets之后，model要来测试样本了。在model(test_datasets)之前，需要加上model.eval().
# 否则的话，有输入数据，即使不训练，它也会改变权值。这是model中含有batch normalization层所带来的的性质。

with torch.no_grad():    # 屏蔽梯度计算,数据不需要计算梯度，也不会进行反向传播
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)# 得到模型预测该样本属于哪个类别的信息，这里采用torch.max。
                                                 # torch.max()的第一个输入是tensor格式，所以用outputs.data而不是outputs作为输入；
                                                 # 第二个参数1是代表dim的意思，也就是取每一行的最大值
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'resnet2.ckpt')
#model_dict=model.load_state_dict(torch.load('resnet1.ckpt'))




