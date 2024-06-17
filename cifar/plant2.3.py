import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

# 加载数据
with open('plant_data/ysys/31-DP-CNN-NO-3.txt', 'r') as f1:
    data1 = f1.readlines()  # 加载第一个文件的数据

with open('plant_data/ysys/32-DP-KDCNN-ye-3.txt', 'r') as f2:
    data2 = f2.readlines()  # 加载第二个文件的数据

with open('plant_data/ysys/33-DP-FedPerCNN-ye-3.txt', 'r') as f3:
    data3 = f3.readlines()  # 加载第三个文件的数据

with open('plant_data/ysys/34-DP-LGCNN34-ye-3.txt', 'r') as f4:
    data4 = f4.readlines()  # 加载第四个文件的数据

# with open('plant_data/fedkd-dy4-88.42.txt', 'r') as f5:
#     data5 = f5.readlines()  # 加载第五个文件的数据

# 从每行数据中提取准确率值，分别保存到两个数组中
acc1 = [float(line.split()[-1]) for line in data1][:3]
acc2 = [float(line.split()[-1]) for line in data2][:3]
acc3 = [float(line.split()[-1]) for line in data3][:3]
acc4 = [float(line.split()[-1]) for line in data4][:3]
# acc5 = [float(line.split()[-1]) for line in data5][:100]

# 设置纵坐标范围为0到100
plt.ylim(80, 100)
y_ticks = [i * 10 for i in range(11)]
y_major_locator = FixedLocator(y_ticks)
plt.gca().yaxis.set_major_locator(y_major_locator)

# 使用自定义的x轴坐标点
rounds = [2, 5, 10]

# 绘制每个文件的准确率数据
plt.plot(rounds, acc1[:3], color='#ef8a46', marker='*', label='NO-DP')
plt.plot(rounds, acc2[:3], color='#FFD06E', marker='p', label='PDP-FD')
plt.plot(rounds, acc3[:3], color='#73BCD5', marker='s', label='PDP-FedPer')
plt.plot(rounds, acc4[:3], color='#386795', marker='o', label='PDP-LG-Fed')
# plt.plot(rounds, acc4[:3], color='red', marker='o', label='PDP-LG-Fed')
# plt.plot(rounds, acc4[:3], color='red', marker='o', label='PDP-LG-Fed')
# plt.plot(x_ticks, [acc1[3], acc2[3], acc3[3]], color='cornflowerblue', marker='^', label='personalization layer 4')
# 添加标签和图例
plt.xlabel('Privacy budget ε', fontsize='17')
plt.ylabel('Accuracy', fontsize='17')
plt.legend(loc='best', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=5)  # Adjust the bbox_to_anchor and ncol as needed
# plt.subplots_adjust(bottom=0.2)
# 保存图为svg格式，即矢量图格式
plt.savefig("picture/acc.svg", dpi=300, format="svg")

# # 保存图为eps格式
plt.savefig("picture/acc.eps", dpi=300, format="eps")

# 显示图表
plt.show()
