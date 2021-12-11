import codecs
import numpy as np
import matplotlib.pyplot as plt
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

plt.scatter(rec,pre,s = 0.1,c = 'red')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
