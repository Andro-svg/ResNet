import torch
import matplotlib.pyplot as plt

"""生成随机数据"""
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x轴数据从-1到1，共100个数据，unsqueeze把一维的数据变为2维的数据
y = x.pow(2) + 0.2 * torch.rand(x.size())  # y=x*2，但是还要加上波动

"""保存神经网络"""


def save():
    net1 = torch.nn.Sequential(  # 快速创建神经网络
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)  # 使用优化器优化神经网络参数，lr为学习效率，SGD为随机梯度下降法
    loss_func = torch.nn.MSELoss()  # 均方差处理回归问题
    for t in range(1000):  # 循环训练
        prediction = net1(x)  # 输入x，得到预测值
        loss = loss_func(prediction, y)  # 计算损失，预测值和真实值的对比
        optimizer.zero_grad()  # 梯度先全部降为0
        loss.backward()  # 反向传递过程
        optimizer.step()  # 以学习效率0.5来优化梯度
    torch.save(net1, 'net.pkl')  # 方法一，保存整个神经网络
    torch.save(net1.state_dict(), 'net_params.pkl')  # 方法二，保存神经网络中结点的参数，效率更高
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)  # 1行3列第1张图
    plt.title('net1')
    plt.scatter(x.data.numpy(), y.data.numpy())  # 初始数据点
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)  # 预测的方程


"""加载神经网络"""


def restore_net():
    net2 = torch.load('net.pkl')  # 加载整个神经网络
    prediction = net2(x)  # 输入x，得到预测值
    plt.subplot(132)  # 1行3列第2张图
    plt.title('net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


"""加载神经网络参数"""


def restore_params():
    net3 = torch.nn.Sequential(  # 先建立一个一模一样的神经网络
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)  # 提取参数放入神经网络中
    )

    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)  # 输入x，得到预测值
    plt.subplot(133)  # 1行3列第3张图
    plt.title('net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()


save()
restore_net()
restore_params()
