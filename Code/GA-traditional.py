import numpy as np
import rasterio
from geopy.distance import geodesic
from pyproj import Geod
import json
import math
import random
from tqdm import tqdm  # 用于显示进度条

# 判断坐标是否在中国境内
def out_of_china(lon, lat):
    return not (72.004 <= lon <= 137.8347 and 0.8293 <= lat <= 55.8271)


# 转换纬度
def transform_lat(lon, lat):
    ret = -100.0 + 2.0 * lon + 3.0 * lat + 0.2 * lat * lat + 0.1 * lon * lat + 0.2 * math.sqrt(abs(lon))
    ret += (20.0 * math.sin(6.0 * lon * math.pi) + 20.0 * math.sin(2.0 * lon * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * math.pi) + 40.0 * math.sin(lat / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * math.pi) + 320 * math.sin(lat * math.pi / 30.0)) * 2.0 / 3.0
    return ret


# 转换经度
def transform_lon(lon, lat):
    ret = 300.0 + lon + 2.0 * lat + 0.1 * lon * lon + 0.1 * lon * lat + 0.1 * math.sqrt(abs(lon))
    ret += (20.0 * math.sin(6.0 * lon * math.pi) + 20.0 * math.sin(2.0 * lon * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lon * math.pi) + 40.0 * math.sin(lon / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lon / 12.0 * math.pi) + 300.0 * math.sin(lon / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


# GCJ-02坐标转WGS-84坐标
def gcj02_to_wgs84(lon, lat):
    if out_of_china(lon, lat):
        return lon, lat
    dlat = transform_lat(lon - 105.0, lat - 35.0)
    dlon = transform_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - 0.00669342162296594323 * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((6335552.717000426 * sqrtmagic) * math.pi)
    dlon = (dlon * 180.0) / ((6378245.0 / sqrtmagic * math.cos(radlat)) * math.pi)
    mgLat = lat + dlat
    mgLon = lon + dlon
    return lon * 2 - mgLon, lat * 2 - mgLat


# 转换已有竞争者坐标系为WGS84
def convert_coordinates(tea_stores):
    converted_stores = []
    for store in tea_stores:
        lon, lat = store['longitude'], store['latitude']
        lon_wgs, lat_wgs = gcj02_to_wgs84(lon, lat)
        converted_stores.append({
            "address": store["address"],
            "longitude": lon_wgs,
            "latitude": lat_wgs
        })
    return converted_stores


# 转换候选点坐标系为WGS84
def convert_candidate_coordinates(candidate_points):
    converted_candidates = []
    for point in candidate_points:
        lat, lon = point  # 这里是 [纬, 经度]
        lon_wgs, lat_wgs = gcj02_to_wgs84(lon, lat)  # 转换时传入 [经度, 纬度]
        converted_candidates.append({
            "longitude": lon_wgs,
            "latitude": lat_wgs
        })
    return converted_candidates



def count_nearby_stores(location, converted_stores, individual, max_distance=1000):
    """
    计算附近的竞争者数量，包括现有竞争者和同一方案中的其他设施
    
    参数:
    location: 当前位置
    converted_stores: 所有现有竞争者位置列表
    individual: 当前解决方案中的所有设施
    max_distance: 最大竞争距离（单位：米），默认1000米
    """
    count = 0
    loc1 = (location['longitude'], location['latitude'])
    
    # 计算现有竞争者
    for store in converted_stores:
        loc2 = (store['longitude'], store['latitude'])
        if abs(loc1[0] - loc2[0]) > 0.01 or abs(loc1[1] - loc2[1]) > 0.01:
            continue
            
        store_dict = {'longitude': loc2[0], 'latitude': loc2[1]}
        distance = calculate_distance(location, store_dict, 0, 0) * 1000
        
        if distance <= max_distance:
            count += 1
    
    # 计算同一方案中的其他设施
    for other_location, _ in individual:
        # 跳过当前位置自身
        if other_location['longitude'] == location['longitude'] and \
           other_location['latitude'] == location['latitude']:
            continue
            
        if abs(loc1[0] - other_location['longitude']) > 0.01 or \
           abs(loc1[1] - other_location['latitude']) > 0.01:
            continue
            
        distance = calculate_distance(location, other_location, 0, 0) * 1000
        
        if distance <= max_distance:
            count += 1
            
    return count



# 圆类，用于服务区的计算
class Circle:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

    def contains(self, point):
        px, py = point
        return (px - self.x) ** 2 + (py - self.y) ** 2 <= self.radius ** 2



# 使用Geod计算四面积
geod = Geod(ellps='WGS84')


def calculate_pixel_area(lon, lat, pixel_width, pixel_height):
    lon1, lat1 = lon, lat
    lon2, lat2 = lon + pixel_width, lat
    lon3, lat3 = lon + pixel_width, lat + pixel_height
    lon4, lat4 = lon, lat + pixel_height

    lons = [lon1, lon2, lon3, lon4, lon1]
    lats = [lat1, lat2, lat3, lat4, lat1]
    area, perimeter = geod.polygon_area_perimeter(lons, lats)

    return abs(area)


def calculate_fitness(individual, population_density, pixel_width, pixel_height, converted_stores, max_budget):
    """
    适应度计算函数
    
    吸引力计算公式：attraction_ij = (s_j * g(c_j)) / exp(d_ij)
    其中 g(c_j) = 1/ln(0.5*c_j+1)，c_j是竞争者数量
    """
    if not validate_individual(individual, max_budget):
        return float('-inf')
    
    total_fitness = 0
    rows, cols = population_density.shape
    
    # 预计算所有设施的竞争者数量
    competition_cache = {}
    for location_j, _ in individual:
        location_key = (location_j['longitude'], location_j['latitude'])
        if location_key not in competition_cache:
            competition_cache[location_key] = count_nearby_stores(location_j, converted_stores, individual)
    
    # 创建距离矩阵
    lats = np.arange(rows) * pixel_height
    lons = np.arange(cols) * pixel_width
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    
    # 初始化总吸引力矩阵
    total_attraction = np.zeros_like(population_density, dtype=float)
    
    # 计算所有设施点的吸引力之和
    for location_j, s_j in individual:
        location_key = (location_j['longitude'], location_j['latitude'])
        c_j = competition_cache[location_key]  # 获取竞争者数量
        
        # 计算g(c_j)：竞争影响因子
        # 当c_j = 0时，log(1) = 0，所以需要特殊处理
        denominator = np.log(0.5 * c_j + 1)
        g_c_j = 1 / denominator if denominator > 0 else 1.0  # 当没有竞争者时返回1
        
        # 计算到当前设施的距离
        d_ij = np.sqrt(
            ((lon_grid - location_j['longitude'])**2 + 
             (lat_grid - location_j['latitude'])**2)
        )
        
        # 累加当前设施的吸引力到总吸引力
        attraction_j = (s_j * g_c_j) / np.exp(d_ij)
        total_attraction += attraction_j
    
    # 计算最终适应度：ln(总吸引力) * 人口密度
    valid_mask = (total_attraction > 0) & np.isfinite(total_attraction)
    fitness_contributions = population_density * np.log(total_attraction, where=valid_mask)
    total_fitness = np.sum(fitness_contributions[valid_mask])
    
    return total_fitness if np.isfinite(total_fitness) else float('-inf')


def calculate_distance(point1, point2, pixel_width, pixel_height):
    """
    计算两点之间的距离（千米）
    """
    # 检查是否在同一栅格内
    if (abs(point1['longitude'] - point2['longitude']) < pixel_width and 
        abs(point1['latitude'] - point2['latitude']) < pixel_height):
        grid_size = (pixel_width + pixel_height) / 2
        return grid_size / 2
    
    # 算实际距离
    R = 6371  # 地球半径（千米）
    
    lat1 = math.radians(point1['latitude'])
    lon1 = math.radians(point1['longitude'])
    lat2 = math.radians(point2['latitude'])
    lon2 = math.radians(point2['longitude'])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c



# 打开人口密度tif文件，获取分辨率和转换信息
tif_path = 'D:/Academic/Task 14 COMP GA/worldpopbj/北京市_chn_ppp_2020_UNadj.tif'
with rasterio.open(tif_path) as dataset:
    population_density = dataset.read(1)
    transform = dataset.transform
    pixel_width = dataset.res[0]
    pixel_height = dataset.res[1]

# 加载已有银行和候选点的坐标
with open('D:/Academic/Task 14 COMP GA/北京市四环内银行.json', 'r', encoding='utf-8') as f:
    converted_stores = json.load(f)

with open('D:/Academic/Task 14 COMP GA/valid_candidates_2.json', 'r', encoding='utf-8') as f:
    candidate_points = json.load(f)

# 转换已有竞争者和候选点的坐标为WGS-84
#converted_stores = convert_coordinates(tea_stores)
converted_candidates = convert_candidate_coordinates(candidate_points)

# 在读取数据后，添加以下代码来筛选需求点
def filter_population_density(population_density, candidate_points, transform):
    """
    筛选包含候选点的栅格
    
    参数:
    population_density: 原始人口密度栅格
    candidate_points: 候选点列表
    transform: 栅格的地理变换参数
    
    返回:
    filtered_density: 筛选后的人口密度栅格 非候选点位置设为0
    """
    filtered_density = np.zeros_like(population_density)
    
    for point in candidate_points:
        # 将地理坐标转换为栅格索引
        row, col = ~transform * (point['longitude'], point['latitude'])
        row, col = int(row), int(col)
        
        # 确保索引在有效范围内
        if (0 <= row < population_density.shape[0] and 
            0 <= col < population_density.shape[1]):
            filtered_density[row, col] = population_density[row, col]
    
    return filtered_density

# 应筛选
population_density = filter_population_density(population_density, converted_candidates, transform)


import random
from tqdm import tqdm  # 用于显示进度条

# 义遗传算法参数
PARAMS = {
    'population_size': 80,        # 种群大小
    'num_generations': 200,       # 迭代次数
    'num_selected_points': 20,    # 选择的设施点数量
    'max_budget': 70             # 最大预算
}

# 将常用常量提取到文件顶部
SCALE_BUDGET = {
    1.0: 1,    # 规模1.0对应预算1
    2.5: 2,    # 规模2.5对应预算2
    7.0: 5     # 规模7.0对应预算5
}

SCALES = list(SCALE_BUDGET.keys())
MAX_ATTEMPTS = 100  # 最大尝试次数
DEFAULT_MAX_BUDGET = 70  # 默认最大预算



def initialize_individual(candidate_points, num_points, max_budget):
    """
    初始化单个个体
    """
    for _ in range(MAX_ATTEMPTS):
        individual = []
        current_budget = 0
        
        # 确保生成固定数量的点
        while len(individual) < num_points:
            # 随机选择位置和规模
            location = random.choice(candidate_points)
            scale = random.choice(SCALES)
            
            # 检查预算约束
            if current_budget + SCALE_BUDGET[scale] <= max_budget:
                individual.append((location, scale))
                current_budget += SCALE_BUDGET[scale]
            
            # 如果无法继续添加点，重新开始
            if len(individual) < num_points and current_budget >= max_budget:
                individual = []
                current_budget = 0
                continue
        
        # 验证生成的个体
        if validate_individual(individual, max_budget):
            return individual
            
    return None  # 如果多次尝试都失败，返回None



def initialize_population(candidate_points, population_size, num_selected_points, max_budget):
    """
    始化种群
    """
    population = []
    while len(population) < population_size:
        # 使用initialize_individual来创建有效的个体
        individual = initialize_individual(candidate_points, num_selected_points, max_budget)
        if individual:  # 确保成功创建了个体
            population.append(individual)
    
    assert len(population) == population_size, f"初始种群大小 {len(population)} 不等于目标大小 {population_size}"
    return population



def generate_offspring(parent_population, candidate_points, max_budget, crossover_rate=0.8, mutation_rate=0.2):
    """
    使用传统遗传算法生成子代
    
    参数:
    parent_population: 父代种群
    candidate_points: 候选点列表
    max_budget: 最大预算
    crossover_rate: 交叉概率,默认0.8
    mutation_rate: 变异概率,默认0.2
    """
    population_size = len(parent_population)
    offspring_population = []
    
    # 精英保留策略：保留10%的最优个体
    sorted_parents = sorted(parent_population, 
                          key=lambda x: calculate_fitness(x, population_density, pixel_width, pixel_height, converted_stores, max_budget),
                          reverse=True)
    num_elite = int(population_size * 0.1)
    offspring_population.extend(sorted_parents[:num_elite])
    
    # 生成剩余子代
    while len(offspring_population) < population_size:
        # 锦标赛选择两个父代
        parent1 = tournament_selection(parent_population)
        parent2 = tournament_selection(parent_population)
        
        # 交叉操作
        if random.random() < crossover_rate:
            child1, child2 = crossover(parent1, parent2)
        else:
            child1, child2 = parent1[:], parent2[:]
            
        # 变异操作
        if random.random() < mutation_rate:
            child1 = mutate(child1, candidate_points, max_budget)
        if random.random() < mutation_rate:
            child2 = mutate(child2, candidate_points, max_budget)
            
        # 验证并添加子代
        if validate_individual(child1, max_budget):
            offspring_population.append(child1)
        if len(offspring_population) < population_size and validate_individual(child2, max_budget):
            offspring_population.append(child2)
    
    return offspring_population[:population_size]

def tournament_selection(population, tournament_size=3):
    """
    锦标赛选择
    """
    tournament = random.sample(population, tournament_size)
    return max(tournament, 
              key=lambda x: calculate_fitness(x, population_density, pixel_width, pixel_height, converted_stores, max_budget=70))

def crossover(parent1, parent2):
    """
    单点交叉操作
    """
    if len(parent1) != len(parent2):
        return parent1, parent2
        
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    
    return child1, child2

def mutate(individual, candidate_points, max_budget):
    """
    变异操作:随机选择一个设施,有50%概率改变位置,50%概率改变规模
    """
    mutated = individual[:]
    mutation_idx = random.randint(0, len(individual) - 1)
    
    if random.random() < 0.5:  # 改变位置
        new_location = random.choice(candidate_points)
        mutated[mutation_idx] = (new_location, individual[mutation_idx][1])
    else:  # 改变规模
        current_budget = sum(SCALE_BUDGET[scale] for _, scale in individual) - SCALE_BUDGET[individual[mutation_idx][1]]
        available_scales = [s for s in SCALES if SCALE_BUDGET[s] + current_budget <= max_budget]
        if available_scales:
            new_scale = random.choice(available_scales)
            mutated[mutation_idx] = (individual[mutation_idx][0], new_scale)
            
    return mutated


def calculate_total_budget(individual):
    """
    计算个体的总预算
    """
    SCALE_BUDGET = {1.0: 1, 2.5: 2, 7.0: 5}
    return sum(SCALE_BUDGET[scale] for _, scale in individual)



# 辅助函数：修复重复位置
def fix_duplicate_locations(offspring, parent1, parent2):
    used_locations = []
    fixed_offspring = []
    
    for facility in offspring:
        location, scale = facility
        if location not in used_locations:
            used_locations.append(location)
            fixed_offspring.append(facility)
        else:
            # 从父代中选择未使用的位置
            all_parent_locations = [loc for loc, _ in parent1 + parent2]
            available_locations = [loc for loc in all_parent_locations 
                                if loc not in used_locations]
            if available_locations:
                new_location = random.choice(available_locations)
                used_locations.append(new_location)
                fixed_offspring.append((new_location, scale))
            
    return fixed_offspring



def validate_individual(individual, max_budget):
    """
    验证个体是否有效
    
    参数：
    individual: 待验证的个体
    
    返回：
    bool: 个体是否有效
    """
    try:
        # 检查个体结构
        if not individual or not isinstance(individual, list):
            return False
            
        # 检查每个设施的格式
        for facility in individual:
            if not isinstance(facility, tuple) or len(facility) != 2:
                return False
            location, scale = facility
            if not isinstance(scale, (int, float)) or scale not in SCALES:
                return False
                
        # 检查预算约束
        total_budget = sum(SCALE_BUDGET[scale] for _, scale in individual)
        if total_budget > max_budget:
            return False
            
        # 检查位置是否重复
        locations = [loc for loc, _ in individual]
        if len(set(tuple(loc.items()) for loc in locations)) != len(locations):
            return False
            
        return True
    except Exception as e:
        print(f"验证个体时出错: {e}")
        return False


# 遗传算法主循环
def genetic_algorithm(candidate_points, converted_stores, population_density, pixel_width, pixel_height,
                     population_size=50, num_generations=200, num_selected_points=20,
                     max_budget=DEFAULT_MAX_BUDGET):
    """
    优算法主函数
    
    参数:
    candidate_points: 候选点列表
    converted_stores: 现有竞争设施列表
    population_density: 人口密度栅格
    pixel_width: 栅格像素宽度
    pixel_height: 栅格像素高度
    population_size: 种群大小
    num_generations: 迭代次数
    num_selected_points: 选择的设施点数量
    max_budget: 最大预算限制
    
    返回:
    best_solution: 最优解
    """
    # 初始化种群
    population = initialize_population(candidate_points, population_size, num_selected_points, max_budget)
    
    # 记录全局最优解
    global_best_individual = None
    global_best_fitness = float('-inf')
    
    # 记录每代的统计数据
    generation_stats = []
    
    # 创建进度条
    pbar = tqdm(range(num_generations), desc="运行遗传算法")
    
    for generation in pbar:
        # 计算适应度
        fitness_scores = [
            calculate_fitness(individual, population_density, pixel_width, pixel_height, converted_stores, max_budget)
            for individual in population
        ]
        
        # 计算当前代的统计数据
        valid_scores = [score for score in fitness_scores if not np.isneginf(score)]
        avg_fitness = sum(valid_scores) / len(valid_scores) if valid_scores else float('-inf')
        current_best = max(fitness_scores)
        
        # 更新全���最优解
        if current_best > global_best_fitness:
            global_best_fitness = current_best
            global_best_individual = population[fitness_scores.index(current_best)]
        
        # 记录统计数据
        generation_stats.append({
            'generation': generation,
            'current_best': current_best,
            'avg_fitness': avg_fitness,
            'global_best': global_best_fitness
        })
        
        # 生成新一代
        population = generate_offspring(population, candidate_points, max_budget, crossover_rate=0.8, mutation_rate=0.2)
        
        # 更新进度条
        pbar.set_postfix({
            '当前最佳适应度': f'{current_best:.2f}',
            '全局最佳适应度': f'{global_best_fitness:.2f}'
        })
    
    # 保存统计数据到文件
    with open('fitness_history-22.txt', 'w', encoding='utf-8') as f:
        f.write("generation,current_best,avg_fitness,global_best\n")
        for stat in generation_stats:
            f.write(f"{stat['generation']},{stat['current_best']},{stat['avg_fitness']},{stat['global_best']}\n")
    
    return global_best_individual



# 使用遗传算法寻找最佳点位
print("开始运行遗传算法...")

try:
    best_candidates = genetic_algorithm(
        candidate_points=converted_candidates,
        converted_stores=converted_stores,
        population_density=population_density,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        **PARAMS
    )
    
    print("\n遗传算法完成！")
    print(f"选出的 {PARAMS['num_selected_points']} 个最佳候选点：")
    
    # 保存结果
    with open('best_candidates-22-traditional.txt', 'w', encoding='utf-8') as f:
        f.write(f"选出的 {PARAMS['num_selected_points']} 个最佳候选点及其规模：\n")
        for location, scale in best_candidates:
            f.write(f"经度: {location['longitude']}, 纬度: {location['latitude']}, 规模: {scale}\n")
            print(f"经度: {location['longitude']}, 纬度: {location['latitude']}, 规模: {scale}")

except Exception as e:
    print(f"\n程序运行出错: {e}")
    raise
 