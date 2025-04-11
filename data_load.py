import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_soil_data(file_path, target_columns):
    """
    参数:
    - file_path: 包含土壤数据的文件路径（假设为CSV格式）
    - target_columns: 列表，包含8个目标土壤指标的列名

    返回:
    - X_train, X_test, y_train, y_test: 训练和测试集的特征和目标值，划分为8:2
    - wavelengths: 波长信息数组
    """
    # 读取CSV文件
    data = pd.read_csv(file_path)

    # 提取波长信息（波长在前4200列的列头中）
    wavelengths = data.columns[:4200].str.replace('spc.', '').astype(float)

    # 假设每个Record包含4200个数据点
    X = data.iloc[:, :4200].values  # 取前4200列作为特征
    y = data[target_columns].values  # 取目标列作为标签

    # 确保特征数据是浮点数类型
    X = X.astype('float32')

    # 分割数据集为训练集和测试集，训练集占80%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 将特征数据重塑为ResNet模型的输入形状
    # 对于1D卷积，我们需要(batch_size, channels, sequence_length)的形状
    X_train = X_train.reshape(-1, 1, 4200)  # 一个通道，序列长度为4200
    X_test = X_test.reshape(-1, 1, 4200)

    return X_train, X_test, y_train, y_test, wavelengths

if __name__ == "__main__":
    # 使用示例
    file_path = 'LUCAS.2009_abs.csv'
    target_columns = ['pH.in.CaCl2', 'pH.in.H2O', 'OC', 'CaCO3', 'N', 'P', 'K', 'CEC']
    X_train, X_test, y_train, y_test, wavelengths = load_soil_data(file_path, target_columns)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    print("wavelengths shape:", wavelengths.shape)
