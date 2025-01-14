# 导入必要的库
import pandas as pd  # 用于数据处理和分析,提供DataFrame等数据结构
import numpy as np  # 用于数值计算,提供多维数组支持
import torch  # PyTorch深度学习框架,用于构建和训练神经网络
import torch.nn as nn  # PyTorch神经网络模块,包含各类神经网络层
import torch.optim as optim  # PyTorch优化器,用于模型参数优化
from torch.utils.data import DataLoader, TensorDataset  # 用于数据加载和批处理
from sklearn.model_selection import train_test_split  # 用于将数据集划分为训练集和测试集
from sklearn.metrics import (  # 用于计算各种模型评估指标
    accuracy_score,  # 准确率:正确预测的样本比例
    precision_score,  # 精确率:正确预测为正例的比例
    recall_score,  # 召回率:正确识别出的正例比例
    f1_score,  # F1分数:精确率和召回率的调和平均
    roc_auc_score,  # ROC曲线下面积,用于评估分类器性能
    confusion_matrix,  # 混淆矩阵,展示预测结果的详细分布
    roc_curve  # 用于绘制ROC曲线的真正例率和假正例率
)
import matplotlib.pyplot as plt  # 用于绘制各类图表和可视化
import seaborn as sns  # 基于matplotlib的统计数据可视化
import matplotlib.font_manager as fm  # 用于管理matplotlib的字体设置
import os  # 用于处理文件和目录路径


def load_and_preprocess_data(train_path, test_path):
    """加载并预处理 HPC 时序数据
    
    该函数完成以下任务:
    1. 从CSV文件加载原始数据
    2. 提取样本ID、时间间隔、标签和特征
    3. 按样本ID组织时序数据
    4. 划分训练集(70%)、验证集(10%)和测试集(20%)
    5. 转换为PyTorch张量格式
    6. 创建DataLoader用于批处理
    
    Args:
        train_path (str): 训练数据文件的完整路径,包含HPC特征数据的CSV文件
        test_path (str): 测试数据文件的完整路径,包含HPC特征数据的CSV文件
        
    Returns:
        train_loader: 训练数据的DataLoader对象,batch_size=32
        val_loader: 验证数据的DataLoader对象,batch_size=32
        test_loader: 测试数据的DataLoader对象,batch_size=32
        y_test: 测试集的标签数组
        input_size: 输入特征的维度,即每个时间步的特征数量
    """
    # 加载 CSV 文件数据
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # 提取各个字段
    train_sample_ids = train_data.iloc[:, 0].values  # 第1列:样本ID,用于区分不同的样本
    test_sample_ids = test_data.iloc[:, 0].values  # 第1列:样本ID,用于区分不同的样本
    train_time_intervals = train_data.iloc[:, 1].values  # 第2列:时间间隔编号,表示采样时间点
    test_time_intervals = test_data.iloc[:, 1].values  # 第2列:时间间隔编号,表示采样时间点
    train_labels = train_data.iloc[:, 2].values  # 第3列:标签(0表示良性,1表示恶意)
    test_labels = test_data.iloc[:, 2].values  # 第3列:标签(0表示良性,1表示恶意)
    train_features = train_data.iloc[:, 3:].values  # 第4列及之后:HPC特征,包含多个性能计数器的值
    test_features = test_data.iloc[:, 3:].values  # 第4列及之后:HPC特征,包含多个性能计数器的值
    
    # 按样本ID分组处理数据
    train_unique_ids = np.unique(train_sample_ids)  # 获取所有不重复的样本ID
    test_unique_ids = np.unique(test_sample_ids)  # 获取所有不重复的样本ID
    train_sequences = []  # 存储每个样本的时序特征序列
    train_sequence_labels = []  # 存储每个样本的标签
    test_sequences = []  # 存储每个样本的时序特征序列
    test_sequence_labels = []  # 存储每个样本的标签

    # 遍历每个样本ID,组织其时序数据
    for sample_id in train_unique_ids:
        indices = train_sample_ids == sample_id  # 找出该样本ID的所有数据行
        train_sequences.append(train_features[indices])  # 添加该样本的完整时间序列
        train_sequence_labels.append(train_labels[indices][0])  # 添加该样本的标签(每个样本只取一个标签)
    for sample_id in test_unique_ids:
        indices = test_sample_ids == sample_id  # 找出该样本ID的所有数据行
        test_sequences.append(test_features[indices])  # 添加该样本的完整时间序列
        test_sequence_labels.append(test_labels[indices][0])  # 添加该样本的标签(每个样本只取一个标签)

    # 转换为NumPy数组便于后续处理
    train_sequences = np.array(train_sequences, dtype=np.float32)  # 转换特征序列为float32类型
    train_sequence_labels = np.array(train_sequence_labels, dtype=np.float32)  # 转换标签为float32类型
    test_sequences = np.array(test_sequences, dtype=np.float32)  # 转换特征序列为float32类型
    test_sequence_labels = np.array(test_sequence_labels, dtype=np.float32)  # 转换标签为float32类型

    # 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(
        train_sequences, 
        train_sequence_labels, 
        test_size=0.1,  # 10%用于验证集
        random_state=42  # 设置随机种子,确保结果可复现
    )

    X_test = test_sequences
    y_test = test_sequence_labels

    # 将NumPy数组转换为PyTorch张量
    X_train = torch.tensor(X_train, dtype=torch.float32)  # 训练特征
    y_train = torch.tensor(y_train, dtype=torch.float32)  # 训练标签
    X_val = torch.tensor(X_val, dtype=torch.float32)  # 验证特征
    y_val = torch.tensor(y_val, dtype=torch.float32)  # 验证标签
    X_test = torch.tensor(X_test, dtype=torch.float32)  # 测试特征
    y_test = torch.tensor(y_test, dtype=torch.float32)  # 测试标签

    # 创建DataLoader对象用于批处理训练
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),  # 将特征和标签打包
        batch_size=32,  # 每批32个样本
        shuffle=True  # 随机打乱训练数据
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), 
        batch_size=32,
        shuffle=False  # 验证集不需要打乱
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=32, 
        shuffle=False  # 测试集不需要打乱
    )

    # 获取输入特征维度
    input_size = X_train.shape[2]  # shape[2]表示每个时间步的特征数量
    return train_loader, val_loader, test_loader, y_test, input_size


class LSTMClassifier(nn.Module):
    """单向LSTM分类器模型
    
    该模型包含:
    1. LSTM层:处理时序特征,捕捉时间序列中的长期依赖关系
    2. 全连接层:将LSTM输出映射到分类空间
    3. Sigmoid激活:将输出压缩到0-1之间,表示二分类概率
    
    Args:
        input_size (int): 输入特征的维度,即每个时间步的特征数量
        hidden_size (int): LSTM隐藏层的维度,决定模型的容量
        num_classes (int): 分类类别数(二分类为1)
        
    Returns:
        模型输出的预测概率(0-1之间)
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        # LSTM层配置
        self.lstm = nn.LSTM(
            input_size,  # 输入特征维度
            hidden_size,  # 隐藏层维度
            batch_first=True  # 批次维度在前,即(batch, seq_len, feature)
        )
        # 全连接层,将hidden_size维的特征映射到num_classes维
        self.fc = nn.Linear(hidden_size, num_classes)
        # Sigmoid激活函数,用于二分类
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通过LSTM层,只使用最后一个时间步的隐藏状态
        _, (hn, _) = self.lstm(x)  # hn形状为(num_layers, batch, hidden_size)
        out = self.fc(hn[-1])  # 取最后一层的隐藏状态通过全连接层
        return self.sigmoid(out)  # 输出二分类概率


class BiLSTMClassifier(nn.Module):
    """双向LSTM分类器模型
    
    该模型包含:
    1. 双向LSTM层:同时从前向后和从后向前处理序列,可以捕捉双向的时序依赖
    2. 全连接层:将双向LSTM的输出映射到分类空间
    3. Sigmoid激活:输出二分类概率
    
    Args:
        input_size (int): 输入特征的维度,即每个时间步的特征数量
        hidden_size (int): LSTM隐藏层的维度,决定模型的容量
        num_classes (int): 分类类别数(二分类为1)
        
    Returns:
        模型输出的预测概率(0-1之间)
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(BiLSTMClassifier, self).__init__()
        # 双向LSTM层配置
        self.lstm = nn.LSTM(
            input_size,  # 输入特征维度
            hidden_size,  # 隐藏层维度
            batch_first=True,  # 批次维度在前
            bidirectional=True  # 设置为双向LSTM
        )
        # 全连接层,因为是双向所以输入维度是hidden_size*2
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        # Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通过双向LSTM层
        _, (hn, _) = self.lstm(x)
        # 连接前向和后向的最后隐藏状态
        # hn[-2]是前向LSTM的最后状态,hn[-1]是后向LSTM的最后状态
        hn = torch.cat((hn[-2], hn[-1]), dim=1)
        out = self.fc(hn)
        return self.sigmoid(out)


class Attention(nn.Module):
    """注意力机制模块
    
    实现了基于加法注意力的机制,用于对序列中不同时间步赋予不同的重要性权重。
    通过学习权重来确定哪些时间步的信息更重要。
    
    Args:
        hidden_size (int): 隐藏层的维度,因为是双向LSTM,所以实际使用的是hidden_size*2
        
    Returns:
        context: 注意力加权后的上下文向量
        attention_weights: 各时间步的注意力权重,用于可视化分析
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # 注意力层,将hidden_size*2维的输入映射到1维,用于计算权重
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output):
        # 计算注意力权重并通过softmax归一化
        # lstm_output的形状为(batch, seq_len, hidden_size*2)
        attention_weights = torch.softmax(
            self.attention(lstm_output),  # 计算每个时间步的权重分数
            dim=1  # 在序列长度维度上进行softmax
        )
        
        # 使用注意力权重对LSTM输出进行加权
        weighted_output = lstm_output * attention_weights
        # 在序列长度维度上求和,得到上下文向量
        context = weighted_output.sum(dim=1)
        return context, attention_weights


class BiLSTMAttentionClassifier(nn.Module):
    """带注意力机制的双向LSTM分类器模型
    
    该模型结合了双向LSTM和注意力机制的优点:
    1. 双向LSTM捕获序列的双向依赖关系
    2. 注意力机制关注重要的时间步
    3. 全连接层进行最终分类
    
    Args:
        input_size (int): 输入特征的维度,即每个时间步的特征数量
        hidden_size (int): LSTM隐藏层的维度,决定模型的容量
        num_classes (int): 分类类别数(二分类为1)
        
    Returns:
        outputs: 模型的预测概率(0-1之间)
        attention_weights: 注意力权重分布,用于分析模型关注点
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(BiLSTMAttentionClassifier, self).__init__()
        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
            bidirectional=True
        )
        # 注意力层
        self.attention = Attention(hidden_size)
        # 全连接层,输入维度为hidden_size*2(双向LSTM的输出)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        # Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通过双向LSTM层得到所有时间步的输出
        lstm_out, _ = self.lstm(x)
        # 应用注意力机制
        context, attention_weights = self.attention(lstm_out)
        # 通过全连接层进行分类
        out = self.fc(context)
        return self.sigmoid(out), attention_weights


def train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=0.001):
    """训练深度学习模型的函数
    
    完整的模型训练流程:
    1. 配置训练设备(GPU/CPU)
    2. 设定损失函数和优化器
    3. 训练循环:
       - 前向传播计算预测值
       - 计算损失
       - 反向传播计算梯度
       - 更新模型参数
    4. 验证评估当前模型性能
    
    Args:
        model: 待训练的模型(LSTM/BiLSTM/BiLSTM+Attention)
        train_loader: 训练数据的DataLoader,batch_size=32
        val_loader: 验证数据的DataLoader,batch_size=32
        num_epochs: 训练轮数,默认30轮
        learning_rate: 学习率,默认0.001,控制参数更新步长
        
    Returns:
        无返回值,但会打印每轮的训练损失和验证损失
    """
    # 设置计算设备,优先使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # 将模型移到指定设备
    print(f"使用设备: {device}")

    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二分类交叉熵损失
    optimizer = optim.Adam(
        model.parameters(),  # 优化模型所有参数
        lr=learning_rate  # 设置学习率
    )

    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()  # 设置为训练模式
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            # 将数据移到指定设备
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播,处理带注意力和不带注意力的模型
            outputs = model(X_batch)[0] if isinstance(model, BiLSTMAttentionClassifier) else model(X_batch)
            
            # 计算损失
            loss = criterion(outputs.squeeze(), y_batch)
            
            # 反向传播和参数更新
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数
            
            train_loss += loss.item()

        # 验证阶段
        model.eval()  # 设置为评估模式
        val_loss = 0.0
        with torch.no_grad():  # 不计算梯度
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)[0] if isinstance(model, BiLSTMAttentionClassifier) else model(X_val)
                val_loss += criterion(outputs.squeeze(), y_val).item()

        # 打印训练进度
        print(
            f"Epoch [{epoch+1}/{num_epochs}], "  # 当前轮数/总轮数
            f"Train Loss: {train_loss/len(train_loader):.4f}, "  # 平均训练损失
            f"Val Loss: {val_loss/len(val_loader):.4f}"  # 平均验证损失
        )


def plot_dataset_distribution(train_loader, val_loader, test_loader, result_dir):
    """绘制数据集样本分布的饼图并保存
    
    可视化训练集、验证集和测试集的样本数量分布:
    1. 计算各数据集的样本数量
    2. 创建饼图展示分布
    3. 设置图表样式和标签
    4. 保存可视化结果
    
    Args:
        train_loader: 训练集的DataLoader对象
        val_loader: 验证集的DataLoader对象
        test_loader: 测试集的DataLoader对象
        result_dir: 结果保存的目录路径
        
    Returns:
        无返回值,但会在result_dir目录下生成dataset_distribution.png饼图
    """
    # 获取各数据集的样本数
    train_size = len(train_loader.dataset)  # 训练集样本数
    val_size = len(val_loader.dataset)  # 验证集样本数
    test_size = len(test_loader.dataset)  # 测试集样本数

    sizes = [train_size, val_size, test_size]
    labels = ['Training Set', 'Validation Set', 'Test Set']

    # 创建饼图
    plt.figure(figsize=(8, 6))  # 设置图形大小
    plt.pie(
        sizes,  # 各部分的大小
        labels=[f"{label} ({size})" for label, size in zip(labels, sizes)],  # 标签显示名称和数量
        autopct='%1.1f%%',  # 显示百分比,保留一位小数
        startangle=90  # 起始角度为90度
    )
    plt.axis('equal')  # 保持饼图为圆形
    
    # 设置标题和字体
    plt.title(
        'Dataset Sample Distribution',
        fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/times.ttf')  # 使用Times New Roman字体
    )
    plt.rcParams['font.family'] = 'Times New Roman'  # 设置全局字体
    
    # 保存图片
    plt.savefig(os.path.join(result_dir, "dataset_distribution.png"))
    plt.close()  # 关闭图形,释放内存


def evaluate_and_visualize(model, test_loader, y_test, model_name, result_dir):
    """评估模型性能并进行可视化
    
    完整的模型评估和可视化流程:
    1. 在测试集上进行预测
    2. 计算各种性能评估指标
    3. 生成混淆矩阵可视化
    4. 对于带注意力的模型,可视化注意力权重
    5. 保存所有评估结果
    
    Args:
        model: 已训练的模型实例
        test_loader: 测试数据的DataLoader
        y_test: 测试集的真实标签
        model_name: 模型名称,用于结果标识
        result_dir: 结果保存的目录路径
        
    Returns:
        accuracy: 准确率,正确预测的比例
        precision: 精确率,正确预测为正例的比例
        recall: 召回率,正确识别出的正例比例
        f1: F1分数,精确率和召回率的调和平均
        auc: ROC曲线下面积,分类器的综合性能指标
    """
    # 设置设备并将模型移到设备上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    # 初始化结果列表
    y_true, y_pred, y_prob = [], [], []  # 存储真实标签、预测标签和预测概率
    attention_weights_list = []  # 存储注意力权重(如果模型包含注意力机制)

    # 在测试集上进行预测
    with torch.no_grad():  # 不计算梯度
        for X_test, y_batch in test_loader:
            X_test, y_batch = X_test.to(device), y_batch.to(device)
            
            # 处理不同类型的模型输出
            if isinstance(model, BiLSTMAttentionClassifier):
                outputs, attention_weights = model(X_test)
                attention_weights_list.append(attention_weights.cpu().squeeze().numpy())
            else:
                outputs = model(X_test)
            
            # 获取预测结果
            probs = outputs.squeeze()  # 预测概率
            predictions = (probs > 0.5).float()  # 二分类阈值0.5
            
            # 收集结果
            y_true.extend(y_batch.cpu().tolist())  # 真实标签
            y_pred.extend(predictions.cpu().tolist())  # 预测标签
            y_prob.extend(probs.cpu().tolist())  # 预测概率

    # 转换标签类型
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    # 计算性能指标
    accuracy = accuracy_score(y_true, y_pred)  # 准确率
    precision = precision_score(y_true, y_pred)  # 精确率
    recall = recall_score(y_true, y_pred)  # 召回率
    f1 = f1_score(y_true, y_pred)  # F1分数
    auc = roc_auc_score(y_true, y_prob)  # AUC值

    # 打印性能指标
    print(
        f"{model_name} - "
        f"Accuracy: {accuracy:.4f}, "
        f"Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, "
        f"F1-Score: {f1:.4f}, "
        f"AUC: {auc:.4f}"
    )

    # 将结果写入文本文件
    results_file = os.path.join(result_dir, "model_results.txt")
    with open(results_file, 'a') as f:
        f.write(
            f"{model_name} - "
            f"Accuracy: {accuracy:.4f}, "
            f"Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, "
            f"F1-Score: {f1:.4f}, "
            f"AUC: {auc:.4f}\n"
        )

    # 绘制并保存混淆矩阵
    cm = confusion_matrix(y_true, y_pred)  # 计算混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,  # 混淆矩阵数据
        annot=True,  # 显示数值
        fmt='d',  # 数值格式为整数
        cmap='Blues',  # 使用蓝色色图
        xticklabels=["Benign", "Malware"],  # x轴标签
        yticklabels=["Benign", "Malware"]  # y轴标签
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(result_dir, f"confusion_matrix_{model_name}.png"))
    plt.close()

    # 可视化注意力权重(仅适用于带注意力机制的模型)
    if attention_weights_list:
        plt.figure(figsize=(10, 6))
        # 绘制前5个样本的注意力权重分布
        for i, att_weights in enumerate(attention_weights_list[:5]):
            plt.plot(att_weights, label=f'Sample {i+1}')
        plt.title("Attention Weights Visualization")
        plt.xlabel("Time Step")
        plt.ylabel("Attention Weight")
        plt.legend()
        plt.savefig(os.path.join(result_dir, f"attention_weights_{model_name}.png"))
        plt.close()

    # 将结果保存到CSV文件
    results_df = pd.DataFrame({
        "Model": [model_name],
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1-Score": [f1],
        "AUC": [auc]
    })

    # 追加到CSV文件
    csv_file = os.path.join(result_dir, "model_evaluation_results.csv")
    results_df.to_csv(
        csv_file,
        mode='a',  # 追加模式
        header=not os.path.exists(csv_file),  # 如果文件不存在则写入表头
        index=False  # 不保存索引
    )

    return accuracy, precision, recall, f1, auc


def compare_models_performance(metrics, result_dir):
    """比较不同模型的性能并可视化
    
    创建模型性能对比的可视化图表:
    1. 整理各模型的性能指标数据
    2. 创建柱状图对比不同指标
    3. 添加数值标签和图例
    4. 设置图表样式
    5. 保存比较结果
    
    Args:
        metrics: 包含各模型性能指标的列表,每个元素是一个字典
        result_dir: 结果保存的目录路径
        
    Returns:
        无返回值,但会生成性能对比图并保存
    """
    # 创建数据框并打印比较结果
    df = pd.DataFrame(metrics)  # 将指标列表转换为DataFrame
    print("\nModel Performance Comparison:")
    print(df)
    df.set_index('Model', inplace=True)  # 将模型名称设为索引
    
    # 创建柱状图
    plt.figure(figsize=(14, 8))  # 设置图形大小
    ax = df[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(
        kind='bar',  # 柱状图类型
        title="Model Performance Metrics",
        ax=plt.gca(),
        color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 设置不同指标的颜色
    )
    
    # 设置坐标轴标签和字体
    plt.ylabel("Score", fontsize=14)
    plt.xticks(rotation=0, fontsize=12)  # x轴标签不旋转
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)

    # 在柱状图上添加数值标签
    for p in ax.patches:
        ax.annotate(
            f'{p.get_height():.2f}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='bottom',
            fontsize=12
        )

    # 设置图例和标题
    plt.legend(
        title="Metrics",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=14
    )
    plt.title("Model Performance Metrics", fontsize=16, fontweight='bold')
    
    # 添加网格线并调整布局
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    
    # 保存图表
    plt.savefig(
        os.path.join(result_dir, "model_performance_comparison.png"),
        bbox_inches='tight'
    )
    plt.close()


def plot_combined_roc_curve(models, test_loader, result_dir):
    """为所有模型绘制ROC曲线并保存
    
    在同一图表中绘制所有模型的ROC曲线:
    1. 获取每个模型的预测结果
    2. 计算TPR和FPR
    3. 绘制ROC曲线
    4. 计算AUC值
    5. 保存结果
    
    Args:
        models: 模型列表,每个元素为(model, model_name)元组
        test_loader: 测试数据的DataLoader
        result_dir: 结果保存的目录路径
        
    Returns:
        无返回值,但会生成ROC曲线图并保存
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plt.figure(figsize=(10, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    
    # 为每个模型绘制ROC曲线
    for model, model_name in models:
        model = model.to(device)
        y_true, y_prob = [], []
        
        # 获取模型预测结果
        with torch.no_grad():
            for X_test, y_batch in test_loader:
                X_test, y_batch = X_test.to(device), y_batch.to(device)
                if isinstance(model, BiLSTMAttentionClassifier):
                    outputs, _ = model(X_test)
                else:
                    outputs = model(X_test)
                
                probs = outputs.squeeze()
                y_true.extend(y_batch.cpu().tolist())
                y_prob.extend(probs.cpu().tolist())

        # 计算ROC曲线和AUC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})")

    # 设置图表样式
    plt.title("Combined ROC Curves for Different Models", fontsize=16)
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    
    # 保存图表
    plt.savefig(
        os.path.join(result_dir, "combined_roc_curves.png"),
        bbox_inches='tight'
    )
    plt.close()


# 设置全局字体样式
plt.rcParams['font.family'] = 'Times New Roman'


def process_dataset(train_path, test_path, result_path, hidden_size=64):
    """处理单个数据集的完整流程
    
    执行完整的数据处理、模型训练和评估流程:
    1. 加载和预处理数据
    2. 创建结果目录
    3. 绘制数据分布
    4. 训练多个模型
    5. 评估模型性能
    6. 生成比较结果
    
    Args:
        train_path: 训练数据集文件的路径
        test_path: 测试数据集文件的路径
        result_path: 结果保存的目录路径
        hidden_size: LSTM隐藏层的大小,默认64
        
    Returns:
        无返回值,但会生成多个结果文件和可视化图表
    """
    # 加载和预处理数据
    train_loader, val_loader, test_loader, y_test, input_size = load_and_preprocess_data(train_path, test_path)

    # 创建结果目录
    os.makedirs(result_path, exist_ok=True)

    # 创建评估结果CSV文件
    with open(os.path.join(result_path, "model_evaluation_results.csv"), 'w') as f:
        f.write("Model,Accuracy,Precision,Recall,F1-Score,AUC\n")

    # 绘制数据集分布
    plot_dataset_distribution(train_loader, val_loader, test_loader, result_path)

    # 定义要训练的模型列表
    models = [
        (LSTMClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=1), "LSTM"),
        (BiLSTMClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=1), "Bi-LSTM"),
        (BiLSTMAttentionClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=1), "Bi-LSTM + Attention")
    ]

    # 训练和评估每个模型
    metrics = []
    for model, model_name in models:
        print(f"Training {model_name}...")
        train_model(model, train_loader, val_loader)
        acc, prec, rec, f1, auc = evaluate_and_visualize(
            model, test_loader, y_test, model_name, result_path
        )
        metrics.append({
            "Model": model_name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })

    # 比较模型性能
    compare_models_performance(metrics, result_path)

    # 绘制ROC曲线
    plot_combined_roc_curve(models, test_loader, result_path)


def main():
    """
    镜像投毒攻击时序检测：
    （1）先在基于硬件性能计数器的容器异常行为检测模型上训练，得到最优模型
    （2）再在基于硬件性能计数器的容器镜像投毒攻击检测模型上测试，验证镜像投毒攻击检测效果
    """

    print("开始处理时分复用 10s_100ms 数据集...")
    print("="*50)
    process_dataset(
        train_path=r"F:\Workspace\code\HPC_Classification\Datasets\Processed\8Events\10s\8_per_1_time\10s_100ms.csv",
        test_path=r"F:\Workspace\code\Imgae_poison\Datasets\Processed\10s\tsc.csv",
        result_path=r"F:\Workspace\code\Imgae_poison\Results\10s\tsc"
    )

    print("\n" + "="*50)
    print("开始处理时分复用 20s_200ms 数据集...")
    print("="*50)
    process_dataset(
        train_path=r"F:\Workspace\code\HPC_Classification\Datasets\Processed\8Events\20s\8_per_1_time\20s_200ms.csv",
        test_path=r"F:\Workspace\code\Imgae_poison\Datasets\Processed\20s\tsc.csv",
        result_path=r"F:\Workspace\code\Imgae_poison\Results\20s\tsc"
    )

    print("\n" + "="*50)
    print("开始处理非时分复用 30s_300ms 数据集...")
    print("="*50)
    process_dataset(
        train_path=r"F:\Workspace\code\HPC_Classification\Datasets\Processed\8Events\30s\8_per_1_time\30s_300ms.csv",
        test_path=r"F:\Workspace\code\Imgae_poison\Datasets\Processed\30s\non_tsc.csv",
        result_path=r"F:\Workspace\code\Imgae_poison\Results\30s\tsc"
    )

if __name__ == "__main__":
    main()
