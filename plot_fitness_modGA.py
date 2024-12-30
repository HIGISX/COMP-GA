import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('fitness_history-l2-mod.txt')

# 创建图形
plt.figure(figsize=(12, 6))

# 绘制全局最优适应度曲线和数据点
plt.plot(df['generation'], df['global_best'], 'b-', label='Global Best Fitness', alpha=0.5)  # 线条设置半透明
plt.scatter(df['generation'], df['global_best'], c='blue', s=20, alpha=0.6, label='Data Points')  # 添加散点

# 设置标题和标签
plt.title('Genetic Algorithm Evolution Curve', fontsize=14)
plt.xlabel('Generation', fontsize=12)
plt.ylabel('Fitness Value', fontsize=12)

# 添加图例
plt.legend(fontsize=10)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.7)

# 优化布局
plt.tight_layout()

# 保存图片
plt.savefig('evolution_curve-l2-mod.png', dpi=300, bbox_inches='tight')

# 显示图片
plt.show()