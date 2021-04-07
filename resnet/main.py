from resnet import ResNet18
from torch import  nn
from torch.nn import functional as F
import torch as t
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

#GPU
device=t.device("cuda" if t.cuda.is_available() else "cpu")
#device_ids=[0,1,2,3]


#参数准备
EPOCH = 1000  #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 128      #批处理尺寸(batch_size)
LR = 0.01        #学习率    先0.01，80%时0.1替换

#数据
#准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

path='E:\神经网络\资料\数据集\cifar-10-batches-py'
trainset = torchvision.datasets.CIFAR10(root='D:\\', train=True, download=False, transform=transform_train) #训练数据集
trainloader = t.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取


testset = torchvision.datasets.CIFAR10(root='D:\\', train=False, download=False, transform=transform_test)
testloader = t.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
#Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#网络
model=ResNet18()
# model=t.nn.DataParallel(model)             #device_ids=[0,1,2,3])   #n值？
model=model.to(device)

#损失函数与优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9,0.999), weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

if __name__=='__main__':
    #model=ResNet()
    #initialize_weight(model)
    print("Start!")
    for epoch in range(pre_epoch,EPOCH):
        model.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            length = len(trainloader)
            #forward and backward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            _, predicted = t.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))



        print("Waiting ")
        with t.no_grad():
            correct = 0
            total = 0
            for data in testloader:#每一批数据均经过测试
                model.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                    # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = t.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('测试分类准确率为：%.3f%%' % (100 * correct / total))

