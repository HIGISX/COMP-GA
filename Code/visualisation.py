import folium
import json
import re
from folium.plugins import Fullscreen, MeasureControl, MousePosition
from folium.features import DivIcon

def read_txt_points(file_path):
    """读取txt文件中的坐标点"""
    points = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 使用正则表达式提取经度、纬度和规模
            pattern = r'经度: ([\d.]+), 纬度: ([\d.]+), 规模: ([\d.]+)'
            match = re.search(pattern, line)
            if match:
                lon = float(match.group(1))  # 经度
                lat = float(match.group(2))  # 纬度
                scale = float(match.group(3))  # 规模
                points.append({
                    'location': [lat, lon],
                    'scale': scale
                })
    return points

def read_json_points(file_path):
    """读取json文件中的银行网点数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    points = []
    for bank in data:
        points.append({
            'location': [bank['latitude'], bank['longitude']],
            'name': bank['address']
        })
    return points

def create_beijing_boundary():
    """创建北京市边界的GeoJSON数据"""
    return {
        "type": "Feature",
        "properties": {"name": "京市"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [115.7178, 39.4915],
                [115.7178, 40.2436],
                [117.4297, 40.2436],
                [117.4297, 39.4915],
                [115.7178, 39.4915]
            ]]
        }
    }

def get_scale_radius(scale):
    """根据规模返回对应的圆圈半径"""
    if scale == 1.0:
        return 4  # 小规模
    elif scale == 2.5:
        return 6  # 中规模
    else:  # 7.0
        return 8  # 大规模

def create_beijing_map(txt_points=None, json_points=None):
    """创建北京地图并标注坐标点"""
    # 创建以北京为中心的地图
    beijing_center = [39.9042, 116.4074]
    m = folium.Map(
        location=beijing_center, 
        zoom_start=12,
        # 使用 CartoDB positron 作为底图（英文，清晰简洁）
        tiles='CartoDB positron',
        attr='CartoDB'
    )
    
    # 添加全屏控件
    Fullscreen().add_to(m)

    # 添加测量控件（包含比例尺功能）
    MeasureControl(
        position='bottomleft',
        primary_length_unit='meters',
        secondary_length_unit=None,
        primary_area_unit='sqmeters',
        secondary_area_unit=None
    ).add_to(m)

    # 添加鼠标位置显示
    MousePosition().add_to(m)

    # 添加北京市边界
    folium.GeoJson(
        create_beijing_boundary(),
        name='Beijing Boundary',  # 改为英文
        style_function=lambda x: {
            'color': '#000000',
            'weight': 2,
            'fillOpacity': 0,
            'dashArray': '5, 5'
        }
    ).add_to(m)

    # 添加txt文件中的点（推荐选址点，红色，小随规模变化）
    if txt_points:
        for point in txt_points:
            radius = get_scale_radius(point['scale'])
            folium.CircleMarker(
                location=point['location'],
                radius=radius,
                popup=f'Scale: {point["scale"]}',  # 改为英文
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.7,
                weight=2
            ).add_to(m)

    # 添加json文件中的银行网点（蓝色小圆点）
    if json_points:
        for point in json_points:
            folium.CircleMarker(
                location=point['location'],
                radius=3,
                popup=point['name'],
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.7,
                weight=1,
                opacity=0.8
            ).add_to(m)

    # 设置地图边界
    m.fit_bounds([[39.4915, 115.7178], [40.2436, 117.4297]])

    # 添加静态比例尺
    scale_length = 5000  # 5公里
    scale_x = 116.455  # 比例尺位置的经度
    scale_y = 39.855   # 比例尺位置的纬度
    
    # 创建比例尺线
    folium.PolyLine(
        locations=[[scale_y, scale_x], [scale_y, scale_x + scale_length/111319.9]],
        weight=5,
        color='black',
        opacity=0.8
    ).add_to(m)
    
    # 添加比例尺文字
    folium.map.Marker(
        [scale_y, scale_x + scale_length/(2*111319.9)],
        icon=DivIcon(
            icon_size=(150,20),
            icon_anchor=(75,0),
            html=f'<div style="font-size: 14px; color: black; font-weight: bold;">5 km</div>'
        )
    ).add_to(m)

    # 添加比例尺装饰
    folium.PolyLine(
        locations=[[scale_y, scale_x], [scale_y + 0.001, scale_x]],
        weight=5,
        color='black',
        opacity=0.8
    ).add_to(m)
    folium.PolyLine(
        locations=[[scale_y, scale_x + scale_length/111319.9], [scale_y + 0.001, scale_x + scale_length/111319.9]],
        weight=5,
        color='black',
        opacity=0.8
    ).add_to(m)

    # 添加自定义比例尺
    folium.map.CustomPane('scalebar').add_to(m)
    folium.map.LayerControl().add_to(m)
    
    # 添加图例
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; 
                right: 50px; 
                width: 180px;
                height: 120px;
                background-color: white;
                border: 2px solid grey;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                z-index: 1000;">
        <div style="margin-bottom: 10px;">
            <span style="display: inline-block;
                       width: 12px;
                       height: 12px;
                       border-radius: 50%;
                       background-color: blue;
                       margin-right: 8px;"></span>
            Existing Facilities
        </div>
        <div style="margin-bottom: 5px;">
            <span style="display: inline-block;
                       width: 8px;
                       height: 8px;
                       border-radius: 50%;
                       background-color: red;
                       margin-right: 8px;"></span>
            Small Scale
        </div>
        <div style="margin-bottom: 5px;">
            <span style="display: inline-block;
                       width: 12px;
                       height: 12px;
                       border-radius: 50%;
                       background-color: red;
                       margin-right: 8px;"></span>
            Medium Scale
        </div>
        <div>
            <span style="display: inline-block;
                       width: 16px;
                       height: 16px;
                       border-radius: 50%;
                       background-color: red;
                       margin-right: 8px;"></span>
            Large Scale
        </div>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))

    # 保存地图
    m.save('results-l2-mod.html')

if __name__ == "__main__":
    # 读取两种坐标点
    txt_points = read_txt_points(r'D:\Academic\Task 14 COMP GA\best_candidates-l2-mod.txt')
    json_points = read_json_points(r'D:\Academic\Task 14 COMP GA\北京市四环内银行.json')
    
    # 创建地图
    create_beijing_map(txt_points, json_points)