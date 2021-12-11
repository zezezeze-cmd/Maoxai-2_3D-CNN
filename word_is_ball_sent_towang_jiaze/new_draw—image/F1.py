import codecs
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


if __name__ == "__main__":
    f = codecs.open('loss_f1_recall_precision.txt', mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8'编码读取
    line = f.readline()  # 以行的形式进行读取文件
    list1 = []
    list2 = []

    while line:
        a = line.split(',')
        b = a[1:2]  # 这是选取需要读取的位数
        c = a[4:5]
        list1.append(b)  # 将其添加在列表之中
        list2.append(c)
        line = f.readline()
    f.close()
    mm = int(list2[-1])

    origin_x = np.linspace(0, len(list1) - 1, len(list1))
    spline_x = np.linspace(0, len(list1) - 1, 100 * len(list1))
    spline_loss = make_interp_spline(origin_x, list1)(spline_x)
    plt.plot(spline_x, spline_loss, c='black')
    plt.xlabel('Iteration/every 100 times')
    plt.ylabel('F1-score')
    plt.show()
