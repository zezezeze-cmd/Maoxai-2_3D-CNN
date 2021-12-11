# coding = utf-8

import codecs
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import make_interp_spline


if __name__ == "__main__":
    with codecs.open('loss_f1_recall_precision.txt', mode='r', encoding='utf-8') as f:
        lines = f.readlines()

    loss = []
    for line in lines:
        loss.append(float(line.split(",")[0]))

    origin_x = np.linspace(0, len(loss)-1, len(loss))

    spline_x = np.linspace(0, len(loss)-1, 100*len(loss))
    spline_loss = make_interp_spline(origin_x, loss)(spline_x)

    # parameter = np.polyfit(origin_x, loss, 5)
    # loss_trendcy = parameter[0] * origin_x ** 5 + parameter[1] * origin_x ** 4 + parameter[2] * origin_x ** 3 \
    #                + parameter[3] * origin_x ** 2 + parameter[4] * origin_x + parameter[5]
    # plt.plot(origin_x, loss_trendcy+1, color='red')

    plt.plot(spline_x, spline_loss)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

