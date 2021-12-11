import codecs
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.rcParams['backend'] = 'SVG'
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.integrate import simps
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']   # 使得图像显示中文

if __name__ == "__main__":
    f = codecs.open('loss_f1_recall_precision.txt', mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8'编码读取
    line = f.readline()  # 以行的形式进行读取文件
    list_recall = []
    xline = []
    list_pre = []
    while line:
        a = line.split(',')
        b = a[2:3]  # 这是选取需要读取的位数
        c = a[3:4]
        for i in b:
            list_recall.append(i)
        for i in c:
            list_pre.append(i)
        line = f.readline()
    f.close()

    rec = []
    pre = []
    for i in list_recall:
        a = float(i)
        rec.append(a)


    for i in list_pre:
        pre.append(float(i))
    pre.reverse()
    # print(len(pre))
    # print(len(rec))
    # plt.scatter(rec, pre, s=0.1, c='black')
    # plt.plot(rec,pre,'k')

    origin_x = np.linspace(0, 1, len(rec))

    spline_x = np.linspace(0, 1, 100 * len(rec))
    spline_pre = make_interp_spline(origin_x, pre)(spline_x)


    "7th Spline interpolation"
    parameter = np.polyfit(origin_x, pre, 7)
    pre_rec_trendcy = parameter[0] * origin_x ** 7 + parameter[1] * origin_x ** 6 + parameter[2] * origin_x ** 5 \
                      + parameter[3] * origin_x ** 4 + parameter[4] * origin_x ** 3 + parameter[5] * origin_x ** 2 \
                      + parameter[6] * origin_x + parameter[7]

    mAP = simps(pre_rec_trendcy, origin_x, dx=0.001)
    print("AUC:%s" % mAP)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(origin_x, pre_rec_trendcy, color='black')
    # plt.plot(spline_x, spline_pre, c='black')


    plt.xlabel('召回率 Recall')
    plt.ylabel('精度 Precision')

    # sns.despine()   # python画图删除上边框和右边框
    plt.savefig('P-R.svg', dpi=300, format='svg')
    plt.show()

