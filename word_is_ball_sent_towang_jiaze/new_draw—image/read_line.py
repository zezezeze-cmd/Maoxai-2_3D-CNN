import codecs
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


if __name__ == "__main__":
    f = codecs.open('loss_f1_recall_precision.txt', mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8'编码读取
    line = f.readline()  # 以行的形式进行读取文件
    list_recall = []
    xline = []
    list_pre = []
    while line:
        a = line.split(',')
        b = a[2:3]  # 这是选取需要读取的位数
        for i in b:
            list_recall.append(i)

        line = f.readline()

        # a = line.split(',')
        c = a[3:4]  # 这是选取需要读取的位数
        for i in c:
            list_pre.append(i)

         # 将其添加在列表之中

        line = f.readline()
    f.close()
    rec = []
    pre = []
    # print(type(list_recall))
    # print(list_recall)


    for i in list_recall:
        a = float(i)
        rec.append(a)

    for i in list_pre:
        pre.append(float(i))
    pre.reverse()

    origin_x = np.linspace(0, len(rec) - 1, len(rec))

    spline_x = np.linspace(0, len(rec) - 1, 100 * len(rec))
    spline_pre = make_interp_spline(origin_x, pre)(spline_x)

    "5th Spline interpolation"
    # parameter = np.polyfit(origin_x, pre, 5)
    # pre_rec_trendcy = parameter[0] * origin_x ** 5 + parameter[1] * origin_x ** 4 + parameter[2] * origin_x ** 3 \
    #                + parameter[3] * origin_x ** 2 + parameter[4] * origin_x + parameter[5]

    "7th Spline interpolation"
    parameter = np.polyfit(origin_x, pre, 7)
    pre_rec_trendcy = parameter[0] * origin_x ** 7 + parameter[1] * origin_x ** 6 + parameter[2] * origin_x ** 5 \
                      + parameter[3] * origin_x ** 4 + parameter[4] * origin_x ** 3 + parameter[5] * origin_x ** 2 \
                      + parameter[6] * origin_x + parameter[7]

    # plt.plot(origin_x, pre_rec_trendcy, color='red')
    plt.plot(spline_x, spline_pre,c='black')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
