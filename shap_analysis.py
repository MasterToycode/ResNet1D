import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from data_load import load_soil_data
from data_processing import preprocess_with_downsampling, process_spectra
from resnet1d_multitask import get_model

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_model_and_data():
    # 定义目标列
    target_columns = ['pH.in.CaCl2', 'pH.in.H2O', 'OC', 'CaCO3', 'N', 'P', 'K', 'CEC']

    # 加载数据
    X_train, X_test, y_train, y_test, wavelengths = load_soil_data('LUCAS.2009_abs.csv', target_columns)

    # 确保数据形状为 (n_samples, n_wavelengths)
    X_train, X_test = X_train.squeeze(), X_test.squeeze()

    # 预处理数据
    X_train = process_spectra(X_train, 'Abs-SG1-SNV')
    X_test = process_spectra(X_test, 'Abs-SG1-SNV')

    X_train, X_train_nwavelengths = preprocess_with_downsampling(X_train, wavelengths, 15)
    X_test, X_test_nwavelengths = preprocess_with_downsampling(X_test, wavelengths, 15)
    
    # 将数据形状调整为 (n_samples, 1, n_wavelengths)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        
    # 加载模型
    device = torch.device('cpu')
    model = get_model('C')
    model.load_state_dict(torch.load('code/models/set.pth', map_location=device))
    model.eval()
    
    return model, X_test, X_test_nwavelengths

def explain_predictions(model, X_test, wavelengths, n_background=50, n_samples=100):
    """
    使用KernelExplainer计算SHAP值，更适合CPU环境
    """
    # 定义一个包装函数来处理模型预测
    def model_predict(x):
        with torch.no_grad():
            x = torch.FloatTensor(x)
            if len(x.shape) == 2:
                x = x.reshape(x.shape[0], 1, -1)
            output = model(x)
            # 如果模型输出是元组，取第一个元素
            if isinstance(output, tuple):
                output = output[0]
            return output.numpy()

    # 准备背景数据和测试数据
    background_data = X_test[:n_background].squeeze()
    test_data = X_test[:n_samples].squeeze()

    # 创建KernelExplainer
    print("创建SHAP解释器...")
    explainer = shap.KernelExplainer(model_predict, background_data)
    
    # 计算SHAP值
    print("计算SHAP值（这可能需要几分钟时间）...")
    shap_values = explainer.shap_values(test_data, nsamples=100)

    # 处理多输出模型的SHAP值
    if isinstance(shap_values, list):
        # 取所有输出的平均值
        avg_shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        avg_shap_values = np.abs(shap_values)

    return avg_shap_values

def plot_top10_wavelengths(shap_values, wavelengths):
    """
    绘制贡献度排名前10的波段柱状图
    """
    # 计算每个波段的平均绝对SHAP值
    mean_shap = np.mean(np.abs(shap_values), axis=0)
    
    # 获取前10个波段的索引
    top10_idx = np.argsort(mean_shap)[-10:][::-1]
    top10_wavelengths = wavelengths[top10_idx]
    top10_values = mean_shap[top10_idx]
    
    # 创建柱状图
    plt.figure(figsize=(8, 6))
    plt.barh(range(10), top10_values, color='skyblue')
    plt.yticks(range(10), [f'{w:.1f}' for w in top10_wavelengths])
    plt.xlabel('mean(|SHAP value|) (average impact on model output magnitude)')
    plt.title('Top 10 Wavelengths by SHAP Value')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('shap_top10_wavelengths.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_shap_summary(shap_values, wavelengths):
    """
    绘制SHAP值的蜂窝图
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, features=wavelengths, feature_names=[f'{w:.1f}' for w in wavelengths], plot_type='dot', show=False)
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 加载模型和数据
    print("加载模型和数据...")
    model, X_test, wavelengths = load_model_and_data()
    
    # 计算SHAP值
    print("正在计算SHAP值...")
    shap_values = explain_predictions(model, X_test, wavelengths)
    
    # 绘制图表
    print("正在生成图表...")
    plot_shap_summary(shap_values, wavelengths)
    plot_top10_wavelengths(shap_values, wavelengths)
    print("分析完成！图表已保存为 'shap_summary_plot.png' 和 'shap_top10_wavelengths.png'")

if __name__ == "__main__":
    main()
