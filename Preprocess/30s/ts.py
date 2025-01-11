import os  # 导入操作系统模块，用于文件和目录操作
import pandas as pd  # 导入pandas库，用于数据处理和分析
import numpy as np  # 导入numpy库，用于数值计算
import re  # 导入正则表达式模块，用于字符串匹配

"""
该脚本用于处理HPC（高性能计数器）数据文件，提取相关信息并生成CSV格式的输出文件。
主要功能包括解析HPC数据文件、处理数据并将其转换为DataFrame格式，最后保存为CSV文件。
"""

def parse_hpc_file(file_path):
    # 提取文件名中的信息
    filename = os.path.basename(file_path)  # 获取文件名
    x, y = filename.split('_')[:2]  # 从文件名中提取x和y的值
    
    # 读取文件内容
    data = []  # 存储解析后的数据
    current_time = None  # 当前时间初始化为None
    current_values = {}  # 当前值初始化为空字典
    
    with open(file_path, 'r') as f:  # 打开文件进行读取
        for line in f:  # 遍历文件中的每一行
            if line.startswith('#') or not line.strip():  # 跳过注释行和空行
                continue
                
            parts = line.strip().split()  # 将行内容按空格分割
            if len(parts) < 3:  # 如果分割后的部分少于3，跳过该行
                continue
                
            # 处理<not counted>的情况
            if '<not counted>' in line:  # 检查是否包含<not counted>
                time = float(parts[0])  # 获取时间
                value = 0  # 设置值为0
                event = parts[3]  # 获取事件名称
            else:
                time = float(parts[0])  # 获取时间
                value = parts[1]  # 获取值
                event = parts[2]  # 获取事件名称
            
            # 转换数值
            if value != 0:  # 如果值不为0
                try:
                    value = int(value.replace(',', ''))  # 尝试将值转换为整数
                except ValueError:  # 如果转换失败
                    value = 0  # 设置值为0
                
            if current_time != time:  # 如果当前时间与读取的时间不同
                if current_time is not None:  # 如果当前时间不为None
                    data.append([current_time, current_values.copy()])  # 将当前时间和对应值添加到数据中
                current_time = time  # 更新当前时间
                current_values = {}  # 重置当前值
            
            current_values[event] = value  # 更新当前事件的值
            
        if current_time is not None:  # 如果当前时间不为None
            data.append([current_time, current_values.copy()])  # 添加最后一组数据
            
    return x, y, data  # 返回提取的x, y值和数据

def process_folder(input_dirs, output_file):
    """处理文件夹中的所有HPC数据文件并生成CSV"""
    # 20个HPC特征
    hpc_events = [
        'branch-instructions', 'branch-misses', 'cache-misses', 'cpu-cycles',
        'instructions', 'L1-dcache-loads', 'LLC-stores', 'iTLB-load-misses'
    ]
    
    all_data = {}  # 存储所有数据的字典
    
    # 处理每个文件
    for file in os.listdir(input_dirs):  # 遍历文件夹中的每个文件
        if file.endswith('.txt'):  # 只处理以.txt结尾的文件
            file_path = os.path.join(input_dirs, file)  # 获取文件的完整路径
            x, y, data = parse_hpc_file(file_path)  # 解析HPC文件
            id_key = f"{x}_{y}"  # 创建唯一的ID键
            
            if id_key not in all_data:  # 如果ID键不在all_data中
                all_data[id_key] = [{} for _ in range(100)]  # 初始化为100个时间点的空字典
                
            for time_idx, (time, values) in enumerate(data):  # 遍历解析后的数据
                if time_idx >= 100:  # 限制最大时间点
                    break
                all_data[id_key][time_idx].update(values)  # 更新对应时间点的值
    
    # 转换为DataFrame格式
    rows = []  # 存储行数据
    for id_key, time_series in all_data.items():  # 遍历所有数据
        x = id_key.split('_')[0]  # 提取x值
        for time_idx, values in enumerate(time_series):  # 遍历时间序列
            row = {
                'sample_id': id_key,  # ID键
                'timestamp_id': time_idx + 1,  # 保持从1开始
                'label': 1 if x == 'M' else 0  # 分类标记
            }
            
            for event in hpc_events:  # 遍历所有HPC事件
                row[event] = values.get(event, 0)  # 获取事件的值，如果不存在则填0
                
            rows.append(row)  # 将行数据添加到rows中
    
    # 创建DataFrame并排序
    df = pd.DataFrame(rows)  # 创建DataFrame
    columns = ['sample_id', 'timestamp_id', 'label'] + hpc_events  # 定义列顺序
    df = df[columns]  # 重新排列DataFrame的列顺序
    
    def extract_number(id_str):  # 提取ID中的数字
        match = re.search(r'(\d+)', id_str)  # 使用正则表达式提取数字
        return int(match.group(1)) if match else 0  # 返回提取的数字或0
    
    df['sort_key'] = df['sample_id'].apply(lambda x: (x.startswith('M'), extract_number(x)))  # 创建排序键
    df_sorted = df.sort_values(['sort_key', 'timestamp_id'])  # 按照排序键和时间戳排序
    df_sorted = df_sorted.drop('sort_key', axis=1)  # 删除排序键列
    
    # 保存CSV文件
    df_sorted.to_csv(output_file, index=False)  # 将DataFrame保存为CSV文件
    return df_sorted  # 返回排序后的DataFrame

def main():
    # 获取项目根目录的绝对路径
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 使用os.path.join()构建完整的路径
    input_dirs = os.path.join(project_root, 'Datasets', 'Original', '30s', 'tsc')
    output_file = os.path.join(project_root, 'Datasets', 'Processed', '30s', 'tsc.csv')
    
    # 创建输出目录(如果不存在)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print("正在处理HPC数据...")  # 打印处理信息
    df_sorted = process_folder(input_dirs, output_file)  # 处理文件夹中的数据
    print(f"已生成CSV文件: {output_file}")  # 打印生成的CSV文件路径

if __name__ == "__main__":
    main()  # 执行主函数
