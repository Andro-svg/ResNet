import matplotlib.pyplot as plt
import numpy as np



data1=np.loadtxt('Res5_1.txt')
data2=np.loadtxt('Res5_2.txt')
data3=np.loadtxt('Res5_3.txt')
data4=np.loadtxt('Res5_4.txt')
data5=np.loadtxt('Res5_5.txt')

plt.title('Results Compared')

plt.plot(data1,label='y1')
plt.plot(data2,label='y2')
plt.plot(data3,label='y3')
plt.plot(data4,label='y4')
plt.plot(data5,label='y5')
plt.legend()

plt.xlabel('epoch')
plt.ylabel('Acc')
plt.show()