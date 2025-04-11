import torch
import torch.nn as nn
import torchvision.models as models
import resnet1d

__all__ = ['ResNet1D_MultiTask', 'get_model']

class ResNet1D_MultiTask(resnet1d.ResNet1D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 获取特征维度
        in_features = self.dense.in_features

        # 移除原始的预测层
        delattr(self, 'dense')
                # 添加多任务预测头
        self.prediction_head = nn.Sequential(
            # 第一层：512 -> 256
            nn.Linear(in_features, in_features//2),
            nn.BatchNorm1d(in_features//2),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            # 第二层：256 -> 128
            nn.Linear(in_features//2, in_features//4),
            nn.BatchNorm1d(in_features//4),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            # 输出层：128 -> 8
            nn.Linear(in_features//4, 8)
        )
    def forward(self, x):
        # 获取特征提取器的输出
        out = x
        
        # first conv
        out = self.first_block_conv(out)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        # residual blocks
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            out = net(out)
            
        # 特征聚合
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)  # 全局平均池化   
        
        out=self.prediction_head(out)
        
        return out  # 输出 8 个指标的预测值


def get_model(model_type):
    if model_type == 'A':  # ResNet18
        return ResNet1D_MultiTask(
            in_channels=1,
            base_filters=32,  # 减小base_filters，降低显存占用
            kernel_size=3,  # 使用3x3卷积核
            stride=2,
            groups=1,
            n_block=8,  # ResNet18的配置
            n_classes=8
        )
    elif model_type == 'B':  # ResNet34
        return ResNet1D_MultiTask(
            in_channels=1,
            base_filters=32,  # 调整base_filters
            kernel_size=3,  # 使用3x3卷积核
            stride=2,
            groups=1,
            n_block=16,  # ResNet34的配置
            n_classes=8
        )
    elif model_type == 'C':  # ResNet50
        return ResNet1D_MultiTask(
            in_channels=1,
            base_filters=32,  # 调整base_filters
            kernel_size=3,  # 使用3x3卷积核
            stride=2,
            groups=1,
            n_block=24,  # ResNet50的配置
            n_classes=8
        )
    else:
        raise ValueError("Invalid model type. Choose 'A' for ResNet18, 'B' for ResNet34, or 'C' for ResNet50")

def print_model_info():
    """
    打印模型关键信息（简化版）
    """
    try:
        from torchsummary import summary
    except ImportError:
        print("请先安装torchsummary: pip install torchsummary")
        return
        
    import torch
    device = torch.device("cpu")
    
    model_types = ['A', 'B', 'C']
    model_names = {
        'A': 'ResNet18',
        'B': 'ResNet34',
        'C': 'ResNet50'
    }
    
    # 模型配置信息
    model_configs = {
        'A': {'n_block': 8, 'base_filters': 32, 'kernel_size': 3},
        'B': {'n_block': 16, 'base_filters': 32, 'kernel_size': 3},
        'C': {'n_block': 24, 'base_filters': 32, 'kernel_size': 3}
    }
    
    print("\n" + "="*50)
    print(f"{'LUCAS土壤光谱分析模型架构':^48}")
    print("="*50)
    print(f"{'输入: (batch_size=15228, channels=1, length=130)':^48}")
    print(f"{'输出: 8个土壤属性预测值':^48}")
    print("-"*50)
    
    for model_type in model_types:
        model = get_model(model_type).to(device)
        config = model_configs[model_type]
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n[Model {model_type}: {model_names[model_type]}]")
        print(f"网络深度: {config['n_block']} blocks")
        print(f"基础通道数: {config['base_filters']}")
        print(f"卷积核大小: {config['kernel_size']}")
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
        # 只打印主要层的信息
        main_layers = {}
        for name, module in model.named_children():
            params = sum(p.numel() for p in module.parameters())
            if params > 0 and params/total_params > 0.05:  # 只显示占比>5%的层
                main_layers[name] = params
        
        if main_layers:
            print("\n主要层结构:")
            for name, params in main_layers.items():
                print(f"  {name:15}: {params:,} ({params/total_params*100:.1f}%)")
        print("-"*50)

if __name__ == '__main__':
    print_model_info()