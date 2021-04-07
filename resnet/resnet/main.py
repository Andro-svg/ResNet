from resnet import ResNet18
from torch import  nn
from torch.nn import functional as F
import torch  as t
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
#import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"     #设置当前使用的GPU设备

#GPU
device=t.device("cuda" if t.cuda.is_available() else "cpu")
#device_ids=[0,1,2,3]


#参数准备
EPOCH = 1     # 遍历数据集次数
pre_epoch = 0    # 定义已经遍历数据集的次数
BATCH_SIZE = 128 #批处理尺寸(batch_size)
LR = 0.01        #学习率    先0.01，80%时0.1替换

#数据
#准备数据集并预处理
# 首先定义一个变换transform，利用的是上面提到的transforms模块中的Compose( )   Compose把多个步骤整合到一起
# 对训练集进行处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，再把图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),     #图像一半的概率翻转，一半的概率不翻转
                                           # 功能：依据概率p对PIL图片进行垂直翻转
                                           # 参数：p- 概率，默认值为0.5
    # 把多个变换组合在一起，可以看到这里面组合了ToTensor和Normalize这两个变换
    transforms.ToTensor(),          #将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

# 对测试集进行处理
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 训练集数据的加载
# 定义训练集，名为trainset，至于后面这一堆，其实就是一个类：
# torchvision.datasets.CIFAR10( )也是封装好了的，在前面提到的torchvision.datasets
path='E:\神经网络\资料\数据集\cifar-10-batches-py'
trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform_train) #定义训练数据集
trainloader = t.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取，shuffle=True，防止过拟合
                                                                                                      #随机打乱训练样本数据
                                                                                                      #不用子进程来加载数据，若为2，用2个子进程用来加载数据，这样可以更快的加载完数据
                                                                                                      #方便产生一个可迭代对象(iterator)，每次输出指定batch_size大小的Tensor
# 测试集数据的加载
# 对于测试集的操作和训练集一样
testset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform_test)
testloader = t.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0) #参数：
                                                                                            #dataset：Dataset类型，从其中加载数据
                                                                                            #batch_size：int，可选。每个batch加载多少样本
                                                                                            #由于是测试集，所以不用进行洗牌操作
                                                                                            #sampler：Sampler，可选。从数据集中采样样本的方法。
                                                                                            #num_workers：int，可选。加载数据时使用多少子进程。默认值为0，表示在主进程中加载数据。
#Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#使用的网络
model=ResNet18()
model=model.to(device)   #将模型加载到指定设备上

#损失函数与优化方式
#神经网络强大之处就在于反向传播，通过比较预测结果与真实结果， 修整网络参数。
#这里的比较就是损失函数，而修整网络参数就是优化器。
#这样充分利用了每个训练数据，使得网络的拟合和预测能力大大提高。
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9,0.999), weight_decay=5e-4) #Adam优化算法
                                                                                        #params(iterable)：可用于迭代优化的参数或者定义参数组的dicts。
                                                                                        #lr (float, optional) ：学习率(默认: 1e-3)
                                                                                        #betas (Tuple[float, float], optional)：用于计算梯度的平均和平方的系数(默认: (0.9, 0.999))
                                                                                        #eps (float, optional)：为了提高数值稳定性而添加到分母的一个项(默认: 1e-8)
                                                                                        #weight_decay (float, optional)：权重衰减(如L2惩罚)(默认: 0)
if __name__=='__main__':
    #model=ResNet()
    #initialize_weight(model)
    print("Start!")
# 分类网络的训练
    print(model)
    for epoch in range(pre_epoch,EPOCH):  # loop over the dataset multiple times 指定训练一共要循环几个epoches
        model.train()   #启用 BatchNormalization 和 Dropout

        #初始化损失，预测正确的图片数，总共的图片数
        sum_loss = 0.0
        correct = 0.0
        total = 0.0

        for i, data in enumerate(trainloader, 0):# 这里我们遇到了第一步中出现的trailoader，代码传入数据
                                                 # enumerate是python的内置函数，既获得索引也获得数据，枚举数据并从下标0开始
            inputs, labels = data                # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
            inputs, labels= inputs.to(device),labels.to(device)  #将数据和标签信息放到GPU上
            optimizer.zero_grad()                # 这里是对一个batch进行的操作，每个batch的操作应该是互不影响的，在进行训练时，根据每个batch累积的梯度，
                                                 # 神经网络的权重是可以调整的，在每个新的batch内梯度必须重新设置为零要把梯度重新归零，把loss关于weight的导数变成0，
                                                 # 因为反向传播过程中梯度会累加上一次循环的梯度
            length = len(trainloader)            # 训练数据集中batch的数量，每一批中有BACTCH_SIZE张图片

            #forward and backward
            outputs = model(inputs)            # 把数据输进model
            loss = criterion(outputs, labels)  # 用前面定义的criterion（交叉熵损失函数）计算损失值
            loss.backward()                    # loss进行反向传播
            optimizer.step()                   # 当执行反向传播之后，把优化器的参数进行更新，以便进行下一轮

            # print statistics 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()   #.item()将一个零维张量转换成浮点数
            _, predicted = t.max(outputs.data, 1)   #这个 _ , predicted是python的一种常用的写法，表示后面的函数其实会返回两个值
                                                    # 但是我们对第一个值不感兴趣，就写个_在那里，把它赋值给_就好，我们只关心第二个值predicted
                                                    #在第一维看 取出最大的数（丢弃）和最大数的位置（保留）后再与label相比即可进行测试
            total += labels.size(0)                            # 累计计算数据集大小
            correct += predicted.eq(labels.data).cpu().sum()   # 累积计算预测正确的数据集的大小
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '            #1个epoch等于使用训练集中的全部样本训练一次；1个iteration等于使用batchsize个样本训练一次；
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))



#分类网络的测试
        print("Waiting ")
        with t.no_grad(): #使一个tensor（命名为x）的requires_grad = True，由x得到的新tensor（命名为w-标量）requires_grad也为False，且grad_fn也为None,即不会对w求导
            #初始化预测正确的图片与所有图片数量
            orrect = 0
            total = 0
            for data in testloader: #每一批数据均经过测试
                model.eval()         #不启用 BatchNormalization 和 Dropout
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = t.max(outputs.data, 1)
                total += labels.size(0)                  # 累加图片总数
                correct += (predicted == labels).sum()   # 累加预测正确的图片数量
            print('测试分类准确率为：%.3f%%' % (100 * correct / total))