import matplotlib.pyplot as plt

# 读取数据
def read_fitness_history(file_path):
    generations = []
    best_fitness = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # 跳过表头
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:  # 检查是否有足够的部分
                continue  # 跳过无效行
            try:
                generations.append(int(parts[0]))  # 代数
                best_fitness.append(float(parts[1]))  # 当代最佳适应度
            except ValueError:
                print(f"跳过无效行: {line.strip()}")  # 输出无效行
                continue  # 跳过无效行
            
    return generations, best_fitness

# 绘制图表
def plot_fitness(generations, best_fitness):
    plt.figure(figsize=(12, 6))
    plt.plot(generations, best_fitness, marker='o', linestyle='-', color='b', label='Best Fitness per Generation')
    
    # 设置标题和标签
    plt.title('Fitness History', fontsize=16)
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Best Fitness', fontsize=14)
    
    # 显示网格
    plt.grid(True)
    
    # 添加图例
    plt.legend(fontsize=12)
    
    # 保存图表为图片
    plt.savefig('fitness_history_plot.png', bbox_inches='tight')  # 使用 bbox_inches='tight' 确保标签不被裁剪
    plt.show()  # 显示图表

if __name__ == "__main__":
    file_path = 'fitness_history-8.txt'  # 文件路径
    generations, best_fitness = read_fitness_history(file_path)
    plot_fitness(generations, best_fitness) 