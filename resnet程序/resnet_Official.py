# ---------------------------------------------------------------------------- #
# An implementation of https://arxiv.org/pdf/1512.03385.pdf                    #
# See section 4.2 for the model architecture on CIFAR-10                       #
# Some part of the code was referenced from below                              #
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py   #
# 参考 https://blog.csdn.net/m0_37673307/article/details/81268222              #
#
# ---------------------------------------------------------------------------- #

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

# Device configuration
device = torch.device('cuda')

# Hyper-parameters
num_epochs = 1000
learning_rate = 0.001

# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 dataset
# 训练集，将相对目录./data下的cifar-10-batches-py文件夹中的全部数据（50000张图片作为训练数据）加载到内存中，若download为True时，会自动从网上下载数据并解压
path='E:\神经网络\GAN\cifar-10-batches-py'
train_dataset = torchvision.datasets.CIFAR10(root=path,
                                             train=True, 
                                             transform=transform,
                                             download=False)
# 测试集，将相对目录./data下的cifar-10-batches-py文件夹中的全部数据（10000张图片作为测试数据）加载到内存中，若download为True时，会自动从网上下载数据并解压
test_dataset = torchvision.datasets.CIFAR10(root=path,
                                            train=False, 
                                            transform=transforms.ToTensor())

# Data loader
# 将训练集的50000张图片划分成12500份，每份4张图，用于mini-batch输入。shffule=True在表示不同批次的数据遍历时，打乱顺序。num_workers=2表示使用两个子进程来加载数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=4,
                                           shuffle=True)
# 将测试集的10000张图片划分成2500份，每份4张图，用于mini-batch输入。
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=4,
                                          shuffle=False)

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1 , bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)     #每一层的输入数据分布一直发生变化，因为在训练的时候，前面层训练参数的更新将导致后面层输入数据分布的变化.保证每一层输入具有相同的分布
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual= x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:      #如果上一个ResidualBlock的输出维度和当前的ResidualBlock的维度不一样，那就对这个x进行downSample操作，如果维度一样，直接加
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64                                       # 定义残差块的输入
        self.conv = conv3x3(3, 64)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)                           #对从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存，不用多存储其他变量

        # 下面的每一个layer都是一个大layer,大layer里面有若干个ResidualBlock
        # 第二个参数是残差块中3*3卷积层的输出深度out_channels,最后是stride   layer=[5,5,5]
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], 2)
        self.layer3 = self.make_layer(block, 256, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)                             # 平均池化，滤波器为8*8，步长为1，特征图大小变为1*1
                                                                    # invariance(不变性)，这种不变性包括translation(平移)，rotation(旋转)，scale(尺度)
                                                                    # 保留主要的特征同时减少参数(降维，效果类似PCA)和计算量，防止过拟合，提高模型泛化能力
        self.fc = nn.Linear(256, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        # downsample 主要用来处理H(x)=F(x)+x中F(x)和x channel维度不匹配问题
        downsample = None                                          # shortcut内部的跨层实现
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        # block()生成上面定义的基础块的对象，并将dowsample传递给block
        # 将每个blocks(也就是大layer)的第一个residual结构保存在layers列表中

        self.in_channels = out_channels
        # 这使得该阶段下面blocks-1个block，即下面循环内构造的block与下一阶段的第一个block的在输入深度上是相同的。

        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))  #一定要注意，out_channels一直都是3*3卷积层的深度
            #将每个blocks的剩下residual结构保存在layers列表中，这样就完成了一个blocks的构造
        return nn.Sequential(*layers)   # 这里表示将layers中的所有block按顺序接在一起

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)   #目的是将多维的的数据如(none,36,2,2)平铺为一维如(none,144)
        out = self.fc(out)
        return out

# 实例化一个残差网络模型
model = ResNet(ResidualBlock, [6, 5, 5]).to(device)
# block对象为 基础块ResidualBlock
# layers列表为 [5,5,5]，这表示网络中每个大layer阶段都是由5个ResidualBlock组成

print(model)
model = torch.nn.DataParallel(model,device_ids=[0,1,2,3])

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  #叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate  动态修改学习率
# optimizer通过param_group来管理参数组.param_group中保存了参数组及其对应的学习率,动量等等.
# 所以我们可以通过更改param_group[‘lr’]的值来更改对应参数组的学习率。
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Record the Time
class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()    #返回当前时间的时间戳

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0
        return ret

    def reset(self):
        self.acc = 0

time1=timer()

# Train the model
fp=open('Res5_4.txt','w')
total_step = len(train_loader)
curr_lr = learning_rate

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader): # 这里我们遇到了第一步中出现的trailoader，代码传入数据
                                                        # enumerate是python的内置函数，既获得索引也获得数据，枚举数据并从下标0开始，i是序号
        time1.tic()
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)              # 把图像输进model
        loss = criterion(outputs, labels)    # 用前面定义的criterion（交叉熵损失函数）计算损失值
        
        # Backward and optimize
        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} Time:{:.3f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item(),time1.release()))

            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)  # torch.max()的第一个输入是tensor格式，所以用outputs.data而不是outputs作为输入；
                                                               # 第二个参数1是代表dim的意思，也就是取每一行的最大值，其实就是我们常见的取概率最大的那个index
                    total += labels.size(0)           #label.size(0) 是一个数
                    correct += (predicted == labels).sum().item()   #predicted与labels对比后相同的值会为1，不同则会为0。.sum()将所有的值相加，得到的仍是tensor类别的int值。item()转成python数字

                print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
                fp.write('%.3f\n'%(100*correct/total))
            model.train()

    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

fp.close()
# Save the model checkpoint
torch.save(model.state_dict(), 'resnet1.ckpt')

model3 = model.eval()
model3.load_state_dict(torch.load('resnet1.ckpt'))

# 仅保存和加载模型参数(推荐使用)
#torch.save(model_object.state_dict(), 'params.pkl')
