import json
import math

def gcj02_to_wgs84(lon, lat):
    """
    将 GCJ-02 坐标转换为 WGS84 坐标
    """
    # 定义常量
    a = 6378245.0  # 地球半径
    ee = 0.00669342162296594323  # 椭球体偏心率平方

    # 计算
    if out_of_china(lon, lat):
        return lon, lat

    dlat = transform_lat(lon - 105.0, lat - 35.0)
    dlon = transform_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * math.pi)
    dlon = (dlon * 180.0) / (a / sqrtmagic * math.cos(radlat) * math.pi)
    mgLat = lat + dlat
    mgLon = lon + dlon
    return lon * 2 - mgLon, lat * 2 - mgLat

def out_of_china(lon, lat):
    return not (72.004 <= lon <= 137.8347 and 0.8293 <= lat <= 55.8271)

def transform_lat(lon, lat):
    ret = -100.0 + 2.0 * lon + 3.0 * lat + 0.2 * lat * lat + 0.1 * lon * lat + 0.2 * math.sqrt(abs(lon))
    ret += (20.0 * math.sin(6.0 * lon * math.pi) + 20.0 * math.sin(2.0 * lon * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * math.pi) + 40.0 * math.sin(lat / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * math.pi) + 320 * math.sin(lat * math.pi / 30.0)) * 2.0 / 3.0
    return ret

def transform_lon(lon, lat):
    ret = 300.0 + lon + 2.0 * lat + 0.1 * lon * lon + 0.1 * lon * lat + 0.1 * math.sqrt(abs(lon))
    ret += (20.0 * math.sin(6.0 * lon * math.pi) + 20.0 * math.sin(2.0 * lon * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lon * math.pi) + 40.0 * math.sin(lon / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lon / 12.0 * math.pi) + 300.0 * math.sin(lon / 30.0 * math.pi)) * 2.0 / 3.0
    return ret

def is_in_restricted_area(point, restricted_areas):
    """
    检查点是否在禁止区内
    """
    lat, lon = point
    for area in restricted_areas:
        if (area['lat_min'] <= lat <= area['lat_max'] and
            area['lon_min'] <= lon <= area['lon_max']):
            return True  # 点在禁止区内
    return False  # 点不在禁止区内

def filter_points(candidates_file, restricted_areas_file, output_file):
    """
    根据禁止区筛选候选点
    """
    # 读取禁止区数据
    with open(restricted_areas_file, 'r', encoding='utf-8') as f:
        restricted_areas = json.load(f)

    # 读取候选点数据
    with open(candidates_file, 'r', encoding='utf-8') as f:
        candidates = json.load(f)

    # 筛选不在禁止区内的候选点
    valid_candidates = []
    for point in candidates:
        lat, lon = point  # 假设输入格式为 [纬度, 经度]
        # 将 GCJ-02 坐标转换为 WGS84
        lon_wgs, lat_wgs = gcj02_to_wgs84(lon, lat)
        if not is_in_restricted_area([lat_wgs, lon_wgs], restricted_areas):
            valid_candidates.append(point)  # 保存原始 GCJ-02 坐标

    # 将有效候选点保存到新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(valid_candidates, f, ensure_ascii=False, indent=4)

    print(f"筛选后的候选点已保存到 {output_file} 文件中。")
    print(f"原始候选点数量：{len(candidates)}")
    print(f"筛选后候选点数量：{len(valid_candidates)}")

if __name__ == "__main__":
    candidates_file = 'valid_candidates.json'  # 候选点文件路径
    restricted_areas_file = '禁止区.json'  # 禁止区文件路径
    output_file = 'valid_candidates_2.json'  # 输出文件路径
    filter_points(candidates_file, restricted_areas_file, output_file)