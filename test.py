import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from data_load import load_soil_data
from data_processing import process_spectra
from data_processing import preprocess_with_downsampling
from resnet1d_multitask import ResNet1D_MultiTask,get_model

bin_sizes = [5,10,15,20]  # 不同的降采样窗口大小
# 预处理数据
methods = ['Abs-SG0', 'Abs-SG0-SNV', 'Abs-SG1', 'Abs-SG1-SNV', 'Abs-SG2', 'Abs-SG2-SNV']
# 定义目标列
target_columns = ['pH.in.CaCl2', 'pH.in.H2O', 'OC', 'CaCO3', 'N', 'P', 'K', 'CEC']


for j in len(bin_sizes):
    # 加载数据
    X_train, X_test, y_train, y_test, wavelengths = load_soil_data('../LUCAS.2009_abs.csv', target_columns)

    # 确保数据形状为 (n_samples, n_wavelengths)
    X_train, X_test = X_train.squeeze(), X_test.squeeze()

    X_train= process_spectra(X_train,methods[5])
    X_test = process_spectra(X_test,methods[5])

    X_train,X_train_nwavelengths=preprocess_with_downsampling(X_train,wavelengths,bin_sizes[j])
    X_test,X_test_nwavelengths=preprocess_with_downsampling(X_test,wavelengths,bin_sizes[j])
    # 将数据形状调整为 (n_samples, 1, n_wavelengths)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    # 检查数据形状
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    assert X_train.shape[0] == y_train.shape[0], "Mismatch in number of samples between X_train and y_train"
    assert X_test.shape[0] == y_test.shape[0], "Mismatch in number of samples between X_test and y_test"

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


    # 模型参数设置
    model_name = 'C'  # 可以改为 'A' 或 'B'
    model = get_model(model_name)
    # 损失函数
    criterion = nn.SmoothL1Loss()
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.81)



    from sklearn.metrics import root_mean_squared_error, r2_score
    import numpy as np
    # 训练参数
    num_epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 初始化指标列表
    train_losses = []
    test_losses = []
    train_rmse = []
    train_r2 = []

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch_x, batch_y in train_loader:
            # 移动数据到设备
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # 前向传播
            outputs = model(batch_x)
            # 计算损失
            loss = criterion(outputs, batch_y)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # 收集预测和真实值
            all_preds.append(outputs.cpu().detach().numpy())
            all_targets.append(batch_y.cpu().detach().numpy())
            
            train_losses.append(total_loss / len(train_loader))
            
        # 更新学习率
        scheduler.step()  # 在每个epoch结束后调整学习率
        # 计算RMSE和R²
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        epoch_rmse = root_mean_squared_error(all_targets, all_preds)
        epoch_r2 = r2_score(all_targets, all_preds)
        train_rmse.append(epoch_rmse)
        train_r2.append(epoch_r2)
        
        # 在每个epoch结束后评估测试集
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                test_outputs = model(batch_x)
                loss = criterion(test_outputs, batch_y)
                test_loss += loss.item()
        test_losses.append(test_loss / len(test_loader))
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')
        #if epoch % 20 == 0: torch.save(model.state_dict(), f'models/now.pth')
        
        
        
        
        

    from sklearn.metrics import root_mean_squared_error, r2_score
    # 模型评估
    model.eval()
    total_test_loss = 0
    test_preds = []
    test_targets = []

    # 将列名和索引建立映射
    target_columns = ['pH.in.CaCl2', 'pH.in.H2O', 'OC', 'CaCO3', 'N', 'P', 'K', 'CEC']
    column_mapping = {i: col for i, col in enumerate(target_columns)}

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            test_outputs = model(batch_x)
            test_loss = criterion(test_outputs, batch_y)
            total_test_loss += test_loss.item()

            # 收集预测值和真实值
            test_preds.append(test_outputs.cpu().numpy())
            test_targets.append(batch_y.cpu().numpy())

        # 计算平均测试损失
        avg_test_loss = total_test_loss / len(test_loader)
        print(f'Average Test Loss: {avg_test_loss:.4f}')

        # 将预测值和真实值拼接在一起
        test_preds = np.concatenate(test_preds, axis=0)
        test_targets = np.concatenate(test_targets, axis=0)

        # 计算每个指标的 RMSE 和 R²
        from datetime import datetime
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'../results3/metrics_model{model_name}_{current_time}.txt'
        
        # 确保results目录存在
        import os
        if not os.path.exists('../results3'):
            os.makedirs('../results3')
        
        with open(results_file, 'w') as f:
            f.write(f"Results for Model {model_name} generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 50 + "\n")
            for i in range(test_targets.shape[1]):
                target_i = test_targets[:, i]
                pred_i = test_preds[:, i]

                # 计算 RMSE 和 R²
                rmse_i = np.sqrt(root_mean_squared_error(target_i, pred_i))
                r2_i = r2_score(target_i, pred_i)

                # 打印当前指标的结果
                result_line = f'Indicator {i + 1} ({column_mapping[i]}) - RMSE: {rmse_i:.4f}, R²: {r2_i:.4f}'
                print(result_line)
                #f.write(result_line + '\n')
                
            #f.write("\nAverage Test Loss: {:.4f}\n".format(avg_test_loss))
        
    # 绘制并保存训练和测试损失图
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Test Loss over Epochs (Model {model_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'../results3/loss_curves_model{model_name}_{current_time}_{bin_sizes[j]}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 绘制并保存指标图
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_rmse, label='Mean_Training RMSE')
    plt.plot(train_r2, label='Mean_Training R2')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title(f'Training Metrics (Model {model_name})')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Test Loss (Model {model_name})')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'../results3/training_metrics_model{model_name}_{current_time}_{bin_sizes[j]}.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nResults have been saved to: {results_file}")
    print(f"Figures have been saved to: ../results3/")