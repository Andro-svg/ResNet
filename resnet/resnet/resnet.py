'''ResNet-18 Image classfication for cifar-10 with PyTorch

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

#Pytorch中神经网络模块化接口nn的了解
#torch.nn是专门为神经网络设计的模块化接口。nn构建于autograd之上，可以用来定义和运行神经网络。
#nn.Module是nn中十分重要的类,包含网络各层的定义及forward方法。
#只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现(利用Autograd)
#在forward函数中可以使用任何Variable支持的函数，毕竟在整个pytorch构建的图中，是Variable在流动。还可以使用if,for,print,log等python语法.

class ResidualBlock(nn.Module):   #继承nn.Module
    """
    子 module: Residual Block ---- ResNet 中一个跨层直连的单元
    """
    def __init__(self, inchannel, outchannel, stride=1):   #self是类的实例本身。直接实例化时，传入相应的参数
                                                           #init() 方法是所谓的对象的“构造函数”，负责在对象初始化时进行一系列的构建操作
        super(ResidualBlock, self).__init__()# 等价于nn.Module.__init__()
                                             #super继承，首先找到ResidualBlock的父类nn.Module，然后把类ResidualBlock的对象self转换为父类nn.Module的对象，然后“被转换”的父类对象调用自己的__init__函数

        # 一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数
        # 将网络的各个层组合到一起

        # main path
        # 不需要这么去想self代表对象本身，换个思维应该认为self是全局变量，如果变量前面加了self，
        # 那么在任何实例方法（非staticmethod和calssmethod）就都可以访问这个变量了，如果没有加self，只有在当前函数内部才能访问这个变量。

        #定义实例的函数left
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False), #输入维度，输出维度，卷积核大小，步长，padding,偏置
            nn.BatchNorm2d(outchannel),           # 批归一化 BatchNorm2d最常用于卷积网络中(防止梯度消失或爆炸)，设置的参数就是卷积的输出通道数
                                                  # 处理后的特征在数据集所有样本上的均值为0，方差为1。标准化处理输⼊数据使各个特征的分布相近：这往往更容易训练出有效的模型。
            nn.ReLU(inplace=True),                # ReLu层，利用in-place计算可以节省内（显）存，同时还可以省去反复申请和释放内存的时间。
                                                  # 例如 y=x + 1, x=y  先将x进行 + 1，操作后赋值给中间变量y，然后将y值赋给x，这需要内存存储变量y。当inplace = True时：
                                                  # 就是对从上层网络nn.Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存，不用多存储其他变量。
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),#是否需要偏置，参考https://blog.csdn.net/u013289254/article/details/98785869
            nn.BatchNorm2d(outchannel)
        )

        # #定义类的实例的函数shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:   # 如果输入和输出的通道不一致，或其步长不为 1，需要将二者转成一致
                                                     #默认跳远连接为identical连接，连接为输入，这是默认的。但是这时候就要求输入与输出大小一致，当stride不为1的时候，输出大小就和输入大小不一样了。所以要改变连接方式。
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )


    def forward(self, x):        #定义类的实例的函数（方法），向前传播
        out = self.left(x)       # main path
        out += self.shortcut(x)  #跳远连接
        out = F.relu(out)        #激活
        return out

class ResNet(nn.Module):
    """
    实现主 module: ResNet-18
    ResNet 包含多个 layer, 每个 layer 又包含多个 residual block (上面实现的类)
    因此, 用 ResidualBlock 实现 Residual 部分，用 _make_layer 函数实现 layer
    """
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()       #首先找到ResNet的父类nn.Module，然后把类ResNet的对象self转换为父类nn.Module的对象
        self.inchannel = 64
        # 最开始的操作
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # 输入深度为3(正好是彩色图片的3个通道)，输出深度为3，滤波器为3*3，步长为1，填充1层，
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # 四个 layer， 对应 2， 3， 4， 5 层， 每层有3，4，6，3个 residual block
        self.layer1 = self.make_layer(ResidualBlock, 64,  3, stride=1)  # 特征图大小不变
        self.layer2 = self.make_layer(ResidualBlock, 128, 4, stride=2)  # 特征图缩小1/2
        self.layer3 = self.make_layer(ResidualBlock, 256, 6, stride=2)  # 特征图缩小1/2
        self.layer4 = self.make_layer(ResidualBlock, 512, 3, stride=2)  # 特征图缩小1/2

        # 最后的全连接，分类时使用
        self.fc = nn.Linear(512, num_classes)        #softmax替换

    def make_layer(self, block, channels, num_blocks, stride):   #make_layer生成多个卷积层，形成一个大的模块
        """
        构建 layer, 每一个 layer 由多个 residual block 组成
        在 ResNet 中，每一个 layer 中只有两个 residual block
        """
        strides = [stride] + [1] * (num_blocks - 1)   # 创建stride列表
        layers = []                                   # 创建层的空列表
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))  #添加块
            self.inchannel = channels
        return nn.Sequential(*layers)
        # 时序容器 Modules 会以它们传入的顺序被添加到容器中

    def forward(self, x):
        # 最开始的处理
        out = self.conv1(x)
        # 四层 layer
        out = self.layer1(out)
        #相似度 卷积计算
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 全连接 输出分类信息
        out = F.avg_pool2d(out, 4)       # F.avg_pool2d()数据是四维输入F.avg_pool2d(input,kernel_size=(4,4))
        out = out.view(out.size(0), -1)  # view函数将张量x变形成一维向量形式，总特征数不变，为全连接层做准备
        out = self.fc(out)               # 全连接输出
        return out


def ResNet18():

    return ResNet(ResidualBlock)
