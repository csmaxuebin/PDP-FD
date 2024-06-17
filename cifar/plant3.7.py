import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

# 加载数据
with open('plant_data/dp2-CNN-96.5.txt', 'r') as f1:
    data1 = f1.readlines()  # 加载第一个文件的数据

with open('plant_data/pj-CNN-dp2-98.5.txt', 'r') as f2:
    data2 = f2.readlines()  # 加载第二个文件的数据

with open('plant_data/dddj-CNN-dp2-98.5.txt', 'r') as f3:
    data3 = f3.readlines()  # 加载第三个文件的数据

# 从每行数据中提取准确率值，分别保存到两个数组中
acc1 = [float(line.split()[-1]) for line in data1][:100]
acc2 = [float(line.split()[-1]) for line in data2][:100]
acc3 = [float(line.split()[-1]) for line in data3][:100]

# 使用步长为10的range函数生成x轴坐标
# rounds = list(range(1, 100, 5))
rounds = list(range(len(acc1)))

# 设置纵坐标范围为0到100
plt.ylim(0, 100)
y_ticks = [i * 10 for i in range(11)]
y_major_locator = FixedLocator(y_ticks)
plt.gca().yaxis.set_major_locator(y_major_locator)

# 绘制每个文件的准确率数据
plt.plot(rounds, acc1, color='#73BCD5', linestyle='-', label='PDP-FD (ε=2)')
plt.plot(rounds, acc2, color='#FFD06E', linestyle='-', label=' DP-FD (ε=2,Fixed)')
plt.plot(rounds, acc3, color='#ef8a46', linestyle='-', label=' DP-FD (ε=2,Uniform)')

# 添加标签和图例
plt.xlabel('Round')
plt.ylabel('Accuracy')
# plt.title('Training Results')
plt.legend()

# 保存图为svg格式，即矢量图格式
plt.savefig("picture/acc.svg", dpi=300,format="svg")

# # 保存图为eps格式
plt.savefig("picture/acc.eps", dpi=300, format="eps")

# 显示图表
plt.show()

