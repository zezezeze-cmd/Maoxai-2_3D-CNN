import matplotlib.pyplot as plt

with open('cccA_rec.txt','r') as f:
    line = f.readline()
    rec = [0.1]  #recall第一个值
    for i,result in enumerate(line):
        if result==' ':
            temp = line[i+1:i+11]
            rec.append(temp)

    X_rec = []
    for i in range(len(rec)+1):
        X_rec.append(i)
    X_rec.remove(0)


print(len(rec))

with open('cccB_pre.txt','r') as f:
    line = f.readline()
    pre = [0.9903973417862304]
    for i,result in enumerate(line):
        if result==' ':
            temp = line[i+1:i+11]
            pre.append(temp)

    X_pre = []
    for i in range(len(rec)+1):
        X_pre.append(i)
    X_pre.remove(0)


    pre_1 = []
    for i in pre:
        temp = float(i)
        temp1 = 1.0-temp
        pre_1.append(temp1)

F1 = []
for i,num in enumerate(rec):
    temppp = (2*float(num)*float(pre[i]))/(float(num)+float(pre[i]))
    F1.append(temppp)
print(len(F1))
print(len(X_pre))
# plt.scatter(X_pre, F1, s=0.2)
# plt.show()

# a = 0
# for i in F1:
#     if i>a:
#         a = i
# print(a)

# plt.title('prec')
# plt.scatter(X_pre, pre, s=0.2)
# plt.show()
#
# plt.title('1-prec')
# plt.scatter(X_pre,pre_1,s=0.2)
# plt.show()
#
# plt.title('1-pr-re')
# plt.scatter(rec,pre_1,s=0.2)
# plt.show()


rec_1 = []
for i in rec:
    rec_1.append(float(i))
pre_1 = []
for i in pre:
    pre_1.append(float(i))
print(rec_1)

for i,num in enumerate(pre_1):
    if num==0:
        print(i)
print('--------------------------')
for i,num in enumerate(rec_1):
    if num==0:
        print(i)

plt.xlabel('recall')
plt.ylabel('precious')

plt.scatter(rec_1,pre_1,s=0.2,c = 'black')
plt.show()