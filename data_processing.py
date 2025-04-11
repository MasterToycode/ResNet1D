import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from data_load import load_soil_data

def apply_sg_filter(spectra, window_length=15, polyorder=2, deriv=0):
    """
    应用Savitzky-Golay滤波器进行光谱平滑或求导
    参数:
    - spectra: 输入光谱数据，形状为(n_samples, n_wavelengths)
    - window_length: 窗口长度，必须是奇数
    - polyorder: 多项式最高阶数
    - deriv: 求导阶数，0表示平滑，1表示一阶导数，2表示二阶导数
    返回:
    - 处理后的光谱数据
    """
    return np.array([savgol_filter(spectrum, window_length, polyorder, deriv=deriv) 
                    for spectrum in spectra])


def apply_snv(spectra):
    """
    应用标准正态变量(SNV)转换 （标准正态变量变换）
    参数:
    - spectra: 输入光谱数据，形状为(n_samples, n_wavelengths)

    返回:
    - SNV处理后的光谱数据
    """
    # 对每个样本进行SNV转换
    spectra_snv = np.zeros_like(spectra)
    for i in range(spectra.shape[0]):
        spectrum = spectra[i]
        # 计算均值和标准差
        mean = np.mean(spectrum)
        std = np.std(spectrum)
        # 应用SNV转换
        spectra_snv[i] = (spectrum - mean) / std
    return spectra_snv




def process_spectra(spectra, method='Abs-SG0'):
    """
    根据指定方法处理光谱数据

    参数:
    - spectra: 输入光谱数据，形状为(n_samples, n_wavelengths)
    - method: 处理方法，可选值包括：
        'Abs-SG0': SG平滑
        'Abs-SG0-SNV': SG平滑+SNV
        'Abs-SG1': SG一阶导
        'Abs-SG1-SNV': SG一阶导+SNV
        'Abs-SG2': SG二阶导
        'Abs-SG2-SNV': SG二阶导+SNV
    
    返回:
    - 处理后的光谱数据
    """
    if method == 'Abs-SG0':
        return apply_sg_filter(spectra, deriv=0)
    elif method == 'Abs-SG0-SNV':
        sg_spectra = apply_sg_filter(spectra, deriv=0)
        return apply_snv(sg_spectra)
    elif method == 'Abs-SG1':
        return apply_sg_filter(spectra, deriv=1)
    elif method == 'Abs-SG1-SNV':
        sg_spectra = apply_sg_filter(spectra, deriv=1)
        return apply_snv(sg_spectra)
    elif method == 'Abs-SG2':
        return apply_sg_filter(spectra, deriv=2)
    elif method == 'Abs-SG2-SNV':
        sg_spectra = apply_sg_filter(spectra, deriv=2)
        return apply_snv(sg_spectra)
    else:
        raise ValueError(f"Unsupported method: {method}")




def remove_wavelength_bands(spectra, wavelengths):
    """
    移除400-499.5nm和2450-2499.5nm的波段
    
    参数:
    - spectra: 输入光谱数据，形状为(n_samples, n_wavelengths)
    - wavelengths: 波长值数组
    
    返回:
    - 处理后的光谱数据和对应的波长值
    """
    # 创建掩码，保留所需波段
    mask = ~((wavelengths >= 400) & (wavelengths <= 499.5) | 
             (wavelengths >= 2450) & (wavelengths <= 2499.5))
    
    # 应用掩码
    filtered_spectra = spectra[:, mask]
    filtered_wavelengths = wavelengths[mask]
    
    return filtered_spectra, filtered_wavelengths




def downsample_spectra(spectra, wavelengths, bin_size):
    """
    对光谱数据进行降采样
    参数:
    - spectra: 输入光谱数据，形状为(n_samples, n_wavelengths)
    - wavelengths: 波长值数组
    - bin_size: 降采样窗口大小（5nm、10nm或15nm）
    
    返回:
    - 降采样后的光谱数据和对应的波长值
    """
    # 计算每个bin的边界
    bins = np.arange(wavelengths[0], wavelengths[-1] + bin_size, bin_size)
    
    # 初始化结果数组
    n_bins = len(bins) - 1
    downsampled_spectra = np.zeros((spectra.shape[0], n_bins))
    downsampled_wavelengths = np.zeros(n_bins)
    
    # 对每个bin进行平均
    for i in range(n_bins):
        mask = (wavelengths >= bins[i]) & (wavelengths < bins[i+1])
        if np.any(mask):
            downsampled_spectra[:, i] = np.mean(spectra[:, mask], axis=1)
            downsampled_wavelengths[i] = np.mean([bins[i], bins[i+1]])
    
    return downsampled_spectra, downsampled_wavelengths





def preprocess_with_downsampling(spectra, wavelengths, bin_size=5):
    """
    完整的预处理流程：移除特定波段并进行降采样
    
    参数:
    - spectra: 输入光谱数据，形状为(n_samples, n_wavelengths)
    - wavelengths: 波长值数组
    - bin_size: 降采样窗口大小（5nm、10nm或15nm）
    
    返回:
    - 处理后的光谱数据和对应的波长值
    """
    # 首先移除指定波段
    filtered_spectra, filtered_wavelengths = remove_wavelength_bands(spectra, wavelengths)
    
    # 然后进行降采样
    downsampled_spectra, downsampled_wavelengths = downsample_spectra(
        filtered_spectra, filtered_wavelengths, bin_size)
    
    return downsampled_spectra, downsampled_wavelengths




def plot_processed_spectra_with_range(original_spectra, wavelengths=None):
    """
    绘制处理方法的光谱图，包括平均曲线和范围
    
    参数:
    - original_spectra: 原始光谱数据，形状为(n_samples, n_wavelengths)
    - wavelengths: 波长值，如果为None则使用索引值
    """
    methods = ['Abs-SG0', 'Abs-SG0-SNV', 'Abs-SG1', 
              'Abs-SG1-SNV', 'Abs-SG2', 'Abs-SG2-SNV']
    
    if wavelengths is None:
        wavelengths = np.arange(original_spectra.shape[1])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 布局：2行3列
    axes = axes.ravel()
    
    for i, method in enumerate(methods):
        processed = process_spectra(original_spectra, method)  # 获取处理后的数据
        mean_curve = np.mean(processed, axis=0)  # 平均光谱曲线
        min_curve = np.min(processed, axis=0)   # 最小值光谱
        max_curve = np.max(processed, axis=0)   # 最大值光谱
        
        # 绘制范围
        axes[i].fill_between(wavelengths, min_curve, max_curve, color='skyblue', alpha=0.3, label='Range')
        # 绘制平均曲线
        axes[i].plot(wavelengths, mean_curve, color='steelblue', label='Average Curve')
        
        # 设置标题和图例
        axes[i].set_title(f'({chr(97 + i)}) {method}', loc='center', fontsize=12)  # a, b, c...
        axes[i].set_xlabel('Wavelength/nm', fontsize=10)
        axes[i].set_ylabel('Absorbance', fontsize=10)
        axes[i].legend()
        axes[i].grid(True)
    
    # 调整布局
    plt.tight_layout(h_pad=2.5, w_pad=3.0)
    plt.show()







# 示例调用
if __name__ == '__main__':
    # 1. 加载数据
    file_path = 'LUCAS.2009_abs.csv'
    target_columns = ['pH.in.CaCl2', 'pH.in.H2O', 'OC', 'CaCO3', 'N', 'P', 'K', 'CEC']
    X_train, X_test, y_train, y_test ,wavelengths= load_soil_data(file_path, target_columns)

    # 2. 将数据重塑为2D
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    
    # 4. 展示原始数据的光谱处理结果
    print("\n=== 光谱预处理结果 ===")
    plot_processed_spectra_with_range(X_train_2d, wavelengths)
    
    # 5. 移除特定波段并进行不同程度的降采样
    print("\n=== 波段移除和降采样结果 ===")
    bin_sizes = [5, 10, 15]  # 不同的降采样窗口大小
    
    # 为不同的降采样结果创建一个新的图
    plt.figure(figsize=(15, 5))
    
    for i, bin_size in enumerate(bin_sizes):
        # 处理数据
        processed_spectra, processed_wavelengths = preprocess_with_downsampling(
            X_train_2d, wavelengths, bin_size)
        
        # 打印信息
        print(f"\n使用 {bin_size}nm 降采样:")
        print(f"处理后的光谱形状: {processed_spectra.shape}")
        print(f"波长数量: {len(processed_wavelengths)}")
        
        # 绘制降采样结果
        plt.subplot(1, 3, i+1)
        mean_curve = np.mean(processed_spectra, axis=0)
        std_curve = np.std(processed_spectra, axis=0)
        
        plt.plot(processed_wavelengths, mean_curve, 'b-', label=f'Mean ({bin_size}nm)')
        plt.fill_between(processed_wavelengths, 
                        mean_curve - std_curve, 
                        mean_curve + std_curve, 
                        color='skyblue', alpha=0.2, label='Standard Deviation Range')
        plt.title(f'Downsampling {bin_size}nm\n(Wavelengths: {len(processed_wavelengths)})')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Absorbance')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 6. 展示完整预处理流程的示例
    print("\n=== 完整预处理流程示例 ===")
    # 先进行光谱预处理
    processed_spectra = process_spectra(X_train_2d, method='Abs-SG0-SNV')
    # 然后进行波段移除和降采样
    final_spectra, final_wavelengths = preprocess_with_downsampling(
        processed_spectra, wavelengths, bin_size=10)
    print(f"最终处理后的数据形状: {final_spectra.shape}")
    print(f"最终波长数量: {len(final_wavelengths)}")
