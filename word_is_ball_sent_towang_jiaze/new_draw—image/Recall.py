
import codecs
import numpy as np

import matplotlib
matplotlib.rcParams['backend'] = 'SVG'

import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']   # 使得图像显示中文

if __name__ == "__main__":
    f = codecs.open('loss_f1_recall_precision.txt', mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8'编码读取
    line = f.readline()  # 以行的形式进行读取文件
    list1 = []
    xline = []
    while line:
        a = line.split(',')
        b = a[2:3]  # 这是选取需要读取的位数
        c = a[4:5]
        list1.append(b)  # 将其添加在列表之中
        xline.append(c)
        line = f.readline()
    f.close()
    xline_last = list(map(int, xline[-1]))

    origin_x = np.linspace(0, xline_last[0], len(list1))

    spline_x = np.linspace(0, xline_last[0], 100 * len(list1))
    spline_loss = make_interp_spline(origin_x, list1)(spline_x)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(spline_x, spline_loss,c='black')
    plt.xlabel('迭代次数 Iterations')
    plt.ylabel('召回率 Recall')
    plt.savefig('Recall.svg', dpi=300, format='svg')  # 保存svg
    plt.show()