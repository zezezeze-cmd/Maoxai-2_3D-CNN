import matplotlib.pyplot as plt

# with open('./loss_f1_recall_precision.txt','r') as f:
#     line = f.readline()
#     for i,result in enumerate(line):
#         if result==' ':
#             temp = line[i+1:i+11]

import codecs
import numpy as np
import matplotlib.pyplot as plt
f = codecs.open('loss_f1_recall_precision.txt', mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8'编码读取
line = f.readline()  # 以行的形式进行读取文件
list1 = []
xline = []
c = 0
while line:
    a = line.split(',')
    b = a[2:3]  # 这是选取需要读取的位数
    list1.append(b)  # 将其添加在列表之中
    xline.append(c)
    line = f.readline()
    c += 1
f.close()

plt.scatter(xline,list1,s=0.2,c = 'blue')
plt.xlabel('train_per')
plt.ylabel('Recall')
plt.show()