import os  # 导入操作系统模块，用于文件和目录操作
import csv  # 导入CSV模块，用于读写CSV文件
import re  # 导入正则表达式模块，用于字符串匹配和处理
from tqdm import tqdm  # 导入tqdm模块，用于显示进度条

def extract_counter_value(line):
    """从给定的行中提取计数器的值
    Args:
        line: 包含计数器值的字符串行
    Returns:
        int: 提取的计数器值，如果无法提取则返回0
    """
    # 使用正则表达式匹配数字(可能包含逗号)或<not counted>
    match = re.search(r'^\s*([0-9,]+|\<not counted\>)', line)
    if match:
        value = match.group(1)  # 获取匹配的第一个分组
        if value == '<not counted>':  # 如果值为<not counted>
            return 0  # 返回0
        return int(value.replace(',', ''))  # 移除逗号并转换为整数
    return 0  # 如果没有匹配到任何值，返回0

def process_file(file_path):
    """处理单个性能计数器文件
    Args:
        file_path: 要处理的文件路径
    Returns:
        dict: 包含所有计数器名称和对应值的字典
    """
    counters = {}  # 初始化存储计数器值的字典
    with open(file_path, 'r') as f:  # 打开文件
        lines = f.readlines()  # 读取所有行
        for line in lines:  # 遍历每一行
            # 检查行中是否包含任何计数器名称
            if any(counter in line for counter in [
                'branch-instructions', 'branch-misses', 'cache-misses', 'cpu-cycles',
                'instructions', 'L1-dcache-loads', 'LLC-stores', 'iTLB-load-misses'
            ]):
                value = extract_counter_value(line)  # 提取计数器值
                counter_name = line.split()[1]  # 获取计数器名称
                counters[counter_name] = value  # 将计数器名称和值存入字典
    return counters  # 返回计数器字典

def get_sample_order(filename):
    """从文件名中提取样本类型和编号用于排序
    Args:
        filename: 文件名字符串
    Returns:
        tuple: (样本类型, 样本编号)
    """
    parts = filename.split('_')  # 按下划线分割文件名
    sample_type = parts[0]  # 获取样本类型(B或M)
    sample_num = int(parts[1].split('.')[0])  # 获取样本编号
    return (sample_type, sample_num)  # 返回排序元组

def main():
    """主函数：处理所有性能计数器文件并生成CSV输出"""
    # 获取项目根目录的绝对路径
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 使用os.path.join()构建完整的路径
    input_dirs = os.path.join(project_root, 'Datasets', 'Original', '20s', 'non_tsc')
    output_file = os.path.join(project_root, 'Datasets', 'Processed', '20s', 'non_tsc.csv')

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 定义要处理的性能计数器特征
    features = [
        'branch-instructions', 'branch-misses', 'cache-misses', 'cpu-cycles',
        'instructions', 'L1-dcache-loads', 'LLC-stores', 'iTLB-load-misses'
    ]

    with open(output_file, 'w', newline='') as f:  # 创建CSV输出文件
        writer = csv.writer(f)  # 创建CSV写入器
        writer.writerow(features + ['label'])  # 写入表头

        # 获取所有txt文件并按样本类型和编号排序
        all_files = sorted(
            [f for f in os.listdir(input_dirs) if f.endswith('.txt')],
            key=get_sample_order
        )

        # 处理每个文件
        for filename in tqdm(all_files, desc="处理样本文件"):  # 使用进度条显示处理进度
            is_malicious = 1 if filename.startswith('M_') else 0  # 确定样本类别
            file_path = os.path.join(input_dirs, filename)  # 构建完整文件路径
            
            counters = process_file(file_path)  # 处理文件获取计数器值
            # 构建输出行：所有特征的值加上类别标签
            row = [counters.get(feature, 0) for feature in features]
            row.append(is_malicious)
            writer.writerow(row)  # 写入CSV文件

if __name__ == '__main__':
    main()  # 执行主函数