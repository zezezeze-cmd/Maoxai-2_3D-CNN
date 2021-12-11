import codecs
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
f = codecs.open('loss_f1_recall_precision.txt', mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8'编码读取
line = f.readline()  # 以行的形式进行读取文件
list1 = []
xline = []
c = 0
while line:
    a = line.split(',')
    b = a[:1]  # 这是选取需要读取的位数
    list1.append(b)  # 将其添加在列表之中
    xline.append(c)
    line = f.readline()
    c += 1
f.close()

# for i in list1:
#     print(i)
# for i in xline:
#     print(i)

# plt.scatter(xline,list1,s=0.2,c = 'blue')


z1 = np.polyfit(xline, list1, 0)              # 曲线拟合，返回值为多项式的各项系数
p1 = np.poly1d(z1)                    # 返回值为多项式的表达式，也就是函数式子
print(p1)
y_pred = p1(xline)
plot2 = plt.plot(xline, y_pred, 'r')
plt.xlabel('train_per')
plt.ylabel('Loss')
plt.show()