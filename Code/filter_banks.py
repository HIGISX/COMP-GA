import json
import folium
from folium.plugins import Fullscreen
from folium.features import DivIcon

def is_within_fourth_ring(lat, lon):
    """判断是否在四环以内"""
    # 四环的大致范围
    LAT_MIN = 39.86
    LAT_MAX = 39.98
    LON_MIN = 116.30
    LON_MAX = 116.50
    
    return LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX

def filter_banks(input_file, output_file):
    """筛选四环以内的银行"""
    with open(input_file, 'r', encoding='utf-8') as f:
        banks = json.load(f)
    
    # 筛选在四环内的银行
    filtered_banks = [
        bank for bank in banks
        if is_within_fourth_ring(bank['latitude'], bank['longitude'])
    ]
    
    # 保存筛选后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_banks, f, indent=4, ensure_ascii=False)
    
    print(f"原始银行数量: {len(banks)}")
    print(f"四环内银行数量: {len(filtered_banks)}")
    return filtered_banks

def create_bank_map(banks):
    """创建银行分布地图"""
    beijing_center = [39.9042, 116.4074]
    m = folium.Map(
        location=beijing_center,
        zoom_start=12,
        tiles='CartoDB positron',
        attr='CartoDB'
    )
    
    Fullscreen().add_to(m)

    # 为不同银行设置不同颜色
    bank_colors = {
        'ICBC': 'red',           # 工商银行
        'CCB': 'blue',           # 建设银行
        'ABC': 'green',          # 农业银行
        'BOC': 'purple',         # 中国银行
        'PSBC': 'orange',        # 邮政储蓄
        'Other Banks': 'gray'    # 其他银行
    }

    # 修改银行名称匹配规则，使用更完整的中文名称
    bank_keywords = {
        'ICBC': ['工商银行', '中国工商银行'],
        'CCB': ['建设银行', '中国建设银行'],
        'ABC': ['农业银行', '中国农业银行'],
        'BOC': ['中国银行'],
        'PSBC': ['邮政储蓄银行', '中国邮政储蓄银行']
    }

    # 添加银行点
    for bank in banks:
        # 确定银行类型和颜色
        bank_type = 'Other Banks'
        name = bank['address']  # 使用 address 字段
        
        # 使用关键词匹配
        for bank_code, keywords in bank_keywords.items():
            if any(keyword in name for keyword in keywords):
                bank_type = bank_code
                break
                
        color = bank_colors[bank_type]
        
        # 创建弹出窗口内容，包含银行名称和地址
        popup_content = f"{name}<br>{bank['address']}"
        
        folium.CircleMarker(
            location=[bank['latitude'], bank['longitude']],
            radius=4,
            popup=popup_content,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7
        ).add_to(m)

    # 添加英文图例
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
    <p><strong>Bank Types</strong></p>
    '''
    bank_names = {
        'ICBC': 'Industrial and Commercial Bank of China',
        'CCB': 'China Construction Bank',
        'ABC': 'Agricultural Bank of China',
        'BOC': 'Bank of China',
        'PSBC': 'Postal Savings Bank of China',
        'Other Banks': 'Other Banks'
    }
    for bank, color in bank_colors.items():
        legend_html += f'<p><i class="fa fa-circle fa-1x" style="color:{color}"></i> {bank_names[bank]}</p>'
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))

    # 添加静态比例尺
    scale_length = 5000  # 5公里
    scale_x = 116.455  # 比例尺位置的经度
    scale_y = 39.855   # 比例尺位置的纬度
    
    # 创建比例尺���
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

    # 保存地图
    m.save('competitor_banks_4th_ring.html')

if __name__ == "__main__":
    input_file = r'D:\Academic\Task 14\北京市国有银行.json'
    output_file = r'D:\Academic\Task 14\北京市四环内银行.json'
    
    # 筛选银行并创建地图
    filtered_banks = filter_banks(input_file, output_file)
    create_bank_map(filtered_banks) 