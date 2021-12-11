# coding = utf-8

import codecs
import numpy as np
import matplotlib
matplotlib.rcParams['backend'] = 'SVG'
import matplotlib.pyplot as plt
from pylab import *
from scipy.interpolate import make_interp_spline

mpl.rcParams['font.sans-serif'] = ['SimHei']   # 使得图像显示中文

if __name__ == "__main__":
    with codecs.open('loss_f1_recall_precision.txt', mode='r', encoding='utf-8') as f:
        lines = f.readlines()

    loss = []
    length = []
    for line in lines:
        loss.append(float(line.split(",")[0]))
        length.append(float(line.split(",")[4]))
    print(length[-1])

    origin_x = np.linspace(0, length[-1], len(loss))
    spline_x = np.linspace(0, length[-1], 100*len(loss))
    spline_loss = make_interp_spline(origin_x, loss)(spline_x)

    # parameter = np.polyfit(origin_x, loss, 5)
    # loss_trendcy = parameter[0] * origin_x ** 5 + parameter[1] * origin_x ** 4 + parameter[2] * origin_x ** 3 \
    #                + parameter[3] * origin_x ** 2 + parameter[4] * origin_x + parameter[5]
    # plt.plot(origin_x, loss_trendcy+1, color='red')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'


    plt.plot(spline_x, spline_loss, c='black')
    plt.xlabel('迭代次数 Iterations')
    plt.ylabel('损失率 Loss')
    # plt.show()        # show尼玛
    plt.savefig('Loss.svg', dpi=300,format='svg')  # 保存svg
    plt.show()

