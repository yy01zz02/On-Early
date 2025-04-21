import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 参数设置
BATCH_SIZE = 32
FEATURE_DROPOUT = 0.3
HIDDEN_DROPOUT = 0.5
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# 前馈神经网络分类器
class FFHallucinationClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(FFHallucinationClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(FEATURE_DROPOUT)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(HIDDEN_DROPOUT)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return torch.sigmoid(x)

# RNN分类器
class RNNHallucinationClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(RNNHallucinationClassifier, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(HIDDEN_DROPOUT)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # 添加序列维度，因为RNN期望输入是[batch, seq_len, features]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        output, h_n = self.rnn(x)
        x = self.dropout(h_n[-1])
        x = self.fc(x)
        return torch.sigmoid(x)

# 生成分类器ROC得分
def gen_classifier_roc(X, y, model_type="FF", input_dim=None):
    X = np.array(X)
    y = np.array(y)
    if len(X) == 0 or len(y) == 0:
        print("数据为空，无法训练分类器")
        return 0.5, 0.5  # 返回随机猜测的AUC和准确率
        
    print(f"特征维度: {X.shape}, 标签维度: {y.shape}")
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 转换为张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 初始化模型
    if input_dim is None:
        input_dim = X.shape[1]
    
    if model_type == "FF":
        model = FFHallucinationClassifier(input_dim).to(device)
    else:
        model = RNNHallucinationClassifier(input_dim).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 训练模型
    model.train()
    for epoch in range(20):  # 10轮训练
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 5 == 0:
            print(f"轮次 {epoch+1}/20, 损失: {total_loss/len(train_loader):.4f}")
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_preds = (test_outputs > 0.5).float()
        roc_auc = roc_auc_score(y_test, test_outputs.cpu().numpy())
        accuracy = accuracy_score(y_test, test_preds.cpu().numpy())
        
    print(f"ROC AUC: {roc_auc:.4f}, 准确率: {accuracy:.4f}")
    return roc_auc, accuracy

def main():
    # 查找Qwen模型结果文件
    results_dir = Path('results')
    result_files = list(results_dir.glob('qwen_code_results_*.pickle'))
    
    if not result_files:
        print("未找到Qwen模型结果文件")
        return
    
    print(f"找到 {len(result_files)} 个Qwen模型结果文件")
    
    # 选择最新的结果文件
    result_file = sorted(result_files)[-1]
    print(f"使用最新的结果文件: {result_file}")
    
    # 加载结果
    with open(result_file, 'rb') as f:
        try:
            results = pickle.load(f)
            print(f"结果加载成功，包含 {len(results['code'])} 个样本")
        except Exception as e:
            print(f"结果加载失败: {e}")
            return
    
    # 检查结果格式
    required_fields = ['code', 'label', 'last_token_pos', 'last_token_softmax', 
                      'attributes_last_token', 'last_token_attention', 
                      'last_token_fully_connected']
    
    for field in required_fields:
        if field not in results:
            warnings.warn(f"结果缺少字段 '{field}'")
        elif len(results[field]) == 0:
            warnings.warn(f"字段 '{field}' 为空")
    
    # 检查样本数量是否一致
    sample_counts = {field: len(results[field]) for field in results if isinstance(results[field], list)}
    if len(set(sample_counts.values())) > 1:
        warnings.warn(f"各字段的样本数量不一致: {sample_counts}")
    
    # 准备分类任务
    labels = np.array(results['label'])
    
    # 分类结果
    classifier_results = {
        'layer_name': [],
        'feature_type': [],
        'roc_auc': [],
        'accuracy': []
    }
    
    # 1. 使用token归因
    if 'attributes_last_token' in results and len(results['attributes_last_token']) > 0:
        print("\n评估 Integrated Gradients 归因特征...")
        X_attr = results['attributes_last_token']
        X_attr_processed = []
        
        # 检查每个属性向量的长度是否与token数量匹配
        for i, attr in enumerate(X_attr):
            if len(attr) != results['last_token_pos'][i] + 1:
                # 如果不匹配，则填充或截断至合适长度
                if len(attr) < results['last_token_pos'][i] + 1:
                    padded_attr = np.zeros(results['last_token_pos'][i] + 1)
                    padded_attr[:len(attr)] = attr
                    X_attr_processed.append(padded_attr)
                else:
                    X_attr_processed.append(attr[:results['last_token_pos'][i] + 1])
            else:
                X_attr_processed.append(attr)
        
        # 找到最短长度并进行截断，保证所有的向量长度一致
        min_len = min(len(attr) for attr in X_attr_processed)
        X_attr_truncated = [attr[:min_len] for attr in X_attr_processed]
        
        roc_auc, accuracy = gen_classifier_roc(X_attr_truncated, labels)
        classifier_results['layer_name'].append('integrated_gradients')
        classifier_results['feature_type'].append('attribution')
        classifier_results['roc_auc'].append(roc_auc)
        classifier_results['accuracy'].append(accuracy)
    
    # 2. 使用最后一个token的softmax概率
    if 'last_token_softmax' in results and len(results['last_token_softmax']) > 0:
        print("\n评估 Softmax 概率特征...")
        X_softmax = results['last_token_softmax']
        
        roc_auc, accuracy = gen_classifier_roc(X_softmax, labels)
        classifier_results['layer_name'].append('softmax')
        classifier_results['feature_type'].append('probability')
        classifier_results['roc_auc'].append(roc_auc)
        classifier_results['accuracy'].append(accuracy)
    
    # 3. 使用全连接层特征
    if 'last_token_fully_connected' in results and len(results['last_token_fully_connected']) > 0:
        print("\n评估全连接层特征...")
        if len(results['last_token_fully_connected']) > 0 and results['last_token_fully_connected'][0] is not None:
            X_fc = results['last_token_fully_connected']
            
            if all(x is not None and len(x) > 0 for x in X_fc):
                roc_auc, accuracy = gen_classifier_roc(X_fc, labels)
                classifier_results['layer_name'].append('fully_connected')
                classifier_results['feature_type'].append('activation')
                classifier_results['roc_auc'].append(roc_auc)
                classifier_results['accuracy'].append(accuracy)
            else:
                print("全连接层特征中包含None或空数组，跳过评估")
        else:
            print("全连接层特征为空或第一个样本为None，跳过评估")
    
    # 4. 使用注意力层特征
    if 'last_token_attention' in results and len(results['last_token_attention']) > 0:
        print("\n评估注意力层特征...")
        if len(results['last_token_attention']) > 0 and results['last_token_attention'][0] is not None:
            X_attn = results['last_token_attention']
            
            if all(x is not None and len(x) > 0 for x in X_attn):
                roc_auc, accuracy = gen_classifier_roc(X_attn, labels)
                classifier_results['layer_name'].append('attention')
                classifier_results['feature_type'].append('activation')
                classifier_results['roc_auc'].append(roc_auc)
                classifier_results['accuracy'].append(accuracy)
            else:
                print("注意力层特征中包含None或空数组，跳过评估")
        else:
            print("注意力层特征为空或第一个样本为None，跳过评估")
    
    # 保存分类结果
    with open('results/qwen_classifier_results.pickle', 'wb') as f:
        pickle.dump(classifier_results, f)
    print(f"分类结果已保存至 'results/qwen_classifier_results.pickle'")
    
    # 打印分类结果摘要
    print("\n分类结果摘要:")
    for i in range(len(classifier_results['layer_name'])):
        layer = classifier_results['layer_name'][i]
        feature = classifier_results['feature_type'][i]
        roc = classifier_results['roc_auc'][i]
        acc = classifier_results['accuracy'][i]
        print(f"特征: {layer} ({feature}), ROC AUC: {roc:.4f}, 准确率: {acc:.4f}")

if __name__ == "__main__":
    main() 