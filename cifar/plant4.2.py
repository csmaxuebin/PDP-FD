import matplotlib.pyplot as plt
import numpy as np

# 自定义横坐标值和纵坐标范围
# ======================dp5====================================
custom_x = ['71.59±0.4', '75.20±0.3', '80.36±0.4']
values1 = [0.88, 1.2, 3.25]
values2 = [1.3, 2.5, 4.2]
values3 = [1.44, 2.71, 3.98]

# 创建x坐标轴位置
x = np.arange(len(custom_x))

# 设置纵坐标范围
plt.ylim(0, 5)

# 创建三个柱状图，使用不同的颜色
# plt.bar(x - 0.25, values1, 0.25, label='PDP-FD', color='#73BCD5')
# plt.bar(x, values2, 0.25, label=' DP-FD (Fixed)', color='#FFD06E')
# plt.bar(x + 0.25, values3, 0.25, label=' DP-FD (Uniform)', color='#ef8a46')

plt.bar(x - 0.25, values1, 0.25, label='PDP-FD', color='None', hatch='//', edgecolor='#73BCD5')
plt.bar(x, values2, 0.25, label=' DP-FD (Fixed)', color='None', hatch='\\\\', edgecolor='#FFD06E')
plt.bar(x + 0.25, values3, 0.25, label=' DP-FD (Uniform)', color='None', hatch='xx', edgecolor='#ef8a46')

# 设置横坐标刻度位置和标签
plt.xticks(x, custom_x)

# 添加标签和图例
plt.xlabel('Accuracy', fontsize='17')
plt.ylabel('Privacy budget ε', fontsize='17')
plt.legend(loc='best', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 在每个柱状图上方添加数字标签
for i, v in enumerate(values1):
    plt.text(i - 0.25, v + 0.1, str(v), color='black', ha='center')

for i, v in enumerate(values2):
    plt.text(i, v + 0.1, str(v), color='black', ha='center')

for i, v in enumerate(values3):
    plt.text(i + 0.25, v + 0.1, str(v), color='black', ha='center')

# 自动调整布局，避免标签重叠
plt.tight_layout()

# 保存图为svg格式，即矢量图格式
plt.savefig("picture/acc.svg", dpi=300,format="svg")

# # 保存图为eps格式
plt.savefig("picture/acc.eps", dpi=300, format="eps")


# 显示图表
plt.show()
