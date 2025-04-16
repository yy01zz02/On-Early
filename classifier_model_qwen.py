import pickle
from pathlib import Path
import numpy as np
import scipy as sp

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import random

from tqdm import tqdm

# 硬件设置
gpu = "0"
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
batch_size = 128
dropout_mlp = 0.5
dropout_gru = 0.25
learning_rate = 1e-4
weight_decay = 1e-2

# 全连接网络分类器
class FFHallucinationClassifier(torch.nn.Module):
    def __init__(self, input_shape, dropout=dropout_mlp):
        super().__init__()
        self.dropout = dropout
        
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(input_shape, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(256, 2)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

# RNN分类器
class RNNHallucinationClassifier(torch.nn.Module):
    def __init__(self, dropout=dropout_gru):
        super().__init__()
        hidden_dim = 128
        num_layers = 4
        self.gru = torch.nn.GRU(1, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=False)
        self.linear = torch.nn.Linear(hidden_dim, 2)
    
    def forward(self, seq):
        if len(seq.shape) == 3:  # 如果输入是[batch_size, seq_len, input_size]
            gru_out, _ = self.gru(seq)
            return self.linear(gru_out[:, -1, :])
        else:  # 如果输入是[seq_len, input_size]
            seq = seq.unsqueeze(0)  # 添加batch维度
            gru_out, _ = self.gru(seq)
            return self.linear(gru_out[:, -1, :]).squeeze(0)  # 移除batch维度

# 生成分类器的ROC
def gen_classifier_roc(inputs, labels):
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels.astype(int), test_size=0.2, random_state=123)
    classifier_model = FFHallucinationClassifier(X_train.shape[1]).to(device)
    X_train = torch.tensor(X_train).to(torch.float).to(device)
    y_train = torch.tensor(y_train).to(torch.long).to(device)
    X_test = torch.tensor(X_test).to(torch.float).to(device)
    y_test = torch.tensor(y_test).to(torch.long).to(device)

    optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for _ in range(1001):
        optimizer.zero_grad()
        sample = torch.randperm(X_train.shape[0])[:batch_size]
        pred = classifier_model(X_train[sample])
        loss = torch.nn.functional.cross_entropy(pred, y_train[sample])
        loss.backward()
        optimizer.step()
    
    classifier_model.eval()
    with torch.no_grad():
        pred = torch.nn.functional.softmax(classifier_model(X_test), dim=1)
        prediction_classes = (pred[:,1]>0.5).type(torch.long).cpu()
        return roc_auc_score(y_test.cpu(), pred[:,1].cpu()), (prediction_classes.numpy()==y_test.cpu().numpy()).mean()

def main():
    results_pattern = "qwen_coder_results_*.pickle"
    inference_results = list(Path("./results/").rglob(results_pattern))
    print(f"查找结果文件 {results_pattern}，找到: {inference_results}")
    
    all_results = {}
    
    for idx, results_file in enumerate(tqdm(inference_results)):
        if str(results_file) not in all_results.keys():
            try:
                classifier_results = {}
                print(f"\n处理文件: {results_file}")
                with open(results_file, "rb") as infile:
                    results_data = pickle.load(infile)
                
                # 检查结果格式 - 如果是列表，将其转换为字典格式
                if isinstance(results_data, list):
                    print("检测到列表格式，转换为字典格式...")
                    results = {
                        'code': [],
                        'label': [],
                        'last_token_pos': [],
                        'last_token_fully_connected': [],
                        'last_token_attention': [],
                        'last_hidden_state': []
                    }
                    
                    for item in results_data:
                        for key in results.keys():
                            if key in item:
                                results[key].append(item[key])
                else:
                    results = results_data
                
                # 检查结果数据结构是否符合预期
                if 'label' not in results or len(results['label']) == 0:
                    print(f"警告: 文件 {results_file} 中没有找到或为空的'label'字段")
                    continue
                
                print(f"数据样本数: {len(results['label'])}")
                labels = np.array(results['label'])
                
                # 处理属性归因 (如果存在)
                if 'attributes_last_token' in results and results['attributes_last_token']:
                    print(f"处理RNN归因分类器...")
                    X_train, X_test, y_train, y_test = train_test_split(
                        results['attributes_last_token'], 
                        labels.astype(int), 
                        test_size=0.2, 
                        random_state=1234
                    )
                    
                    rnn_model = RNNHallucinationClassifier().to(device)
                    optimizer = torch.optim.AdamW(rnn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                    
                    # 打印一些调试信息
                    print(f"训练样本数: {len(X_train)}, 测试样本数: {len(X_test)}")
                    
                    for step in range(1001):
                        # 确保有足够的样本进行采样
                        sample_size = min(batch_size, len(X_train))
                        indices = random.sample(range(len(X_train)), sample_size)
                        
                        # 批量处理以提高效率
                        batch_x = [X_train[i] for i in indices]
                        batch_y = [y_train[i] for i in indices]
                        
                        # 转换为张量并移动到设备
                        y_sub = torch.tensor(batch_y).to(torch.long).to(device)
                        
                        # 单独处理每个属性序列并收集预测
                        optimizer.zero_grad()
                        batch_preds = []
                        for i, x in enumerate(batch_x):
                            x_tensor = torch.tensor(x).view(-1, 1).to(torch.float).to(device)
                            pred = rnn_model(x_tensor)
                            batch_preds.append(pred)
                        
                        preds = torch.stack(batch_preds)
                        
                        if step == 0:
                            print(f"预测形状: {preds.shape}, 标签形状: {y_sub.shape}")
                        
                        loss = torch.nn.functional.cross_entropy(preds, y_sub)
                        loss.backward()
                        optimizer.step()
                        
                        if step % 100 == 0:
                            print(f"步骤 {step}, 损失: {loss.item():.4f}")
                    
                    # 收集测试集预测
                    print(f"评估测试集...")
                    test_preds = []
                    with torch.no_grad():
                        for x in X_test:
                            x_tensor = torch.tensor(x).view(-1, 1).to(torch.float).to(device)
                            pred = rnn_model(x_tensor)
                            test_preds.append(pred)
                    
                    preds = torch.stack(test_preds)
                    preds = torch.nn.functional.softmax(preds, dim=1)
                    prediction_classes = (preds[:,1]>0.5).type(torch.long).cpu()
                    
                    # 计算指标
                    roc = roc_auc_score(y_test, preds[:,1].detach().cpu().numpy())
                    acc = (prediction_classes.numpy()==y_test).mean()
                    
                    print(f"RNN归因 ROC: {roc:.4f}, 准确率: {acc:.4f}")
                    
                    classifier_results['last_token_attribution_rnn_roc'] = roc
                    classifier_results['last_token_attribution_rnn_acc'] = acc
                
                # 处理最后一层的隐藏状态
                if 'last_hidden_state' in results and results['last_hidden_state']:
                    print(f"处理隐藏状态分类器...")
                    last_hidden_states = np.stack(results['last_hidden_state'])
                    hidden_roc, hidden_acc = gen_classifier_roc(last_hidden_states, labels)
                    classifier_results['last_hidden_state_roc'] = hidden_roc
                    classifier_results['last_hidden_state_acc'] = hidden_acc
                    print(f"隐藏状态 ROC: {hidden_roc:.4f}, 准确率: {hidden_acc:.4f}")
                
                # 处理全连接层
                if 'last_token_fully_connected' in results and len(results['last_token_fully_connected']) > 0:
                    print(f"处理全连接层分类器...")
                    # 首先检查维度
                    data_shape = np.array(results['last_token_fully_connected'][0]).shape
                    print(f"全连接层数据形状: {data_shape}")
                    
                    if len(data_shape) == 0 or (len(data_shape) == 1 and data_shape[0] == 0):
                        print("警告: 全连接层数据为空，跳过")
                    else:
                        # 如果是单层数据，直接处理
                        if len(data_shape) == 1:
                            layer_data = np.stack(results['last_token_fully_connected'])
                            layer_roc, layer_acc = gen_classifier_roc(layer_data, labels)
                            classifier_results[f'last_token_fully_connected_roc'] = layer_roc
                            classifier_results[f'last_token_fully_connected_acc'] = layer_acc
                            print(f"全连接层 ROC: {layer_roc:.4f}, 准确率: {layer_acc:.4f}")
                        else:
                            # 逐层处理
                            for layer in range(data_shape[0]):
                                layer_data = np.stack([i[layer] for i in results['last_token_fully_connected']])
                                layer_roc, layer_acc = gen_classifier_roc(layer_data, labels)
                                classifier_results[f'last_token_fully_connected_roc_{layer}'] = layer_roc
                                classifier_results[f'last_token_fully_connected_acc_{layer}'] = layer_acc
                                print(f"全连接层 {layer} ROC: {layer_roc:.4f}, 准确率: {layer_acc:.4f}")
                
                # 处理注意力层
                if 'last_token_attention' in results and len(results['last_token_attention']) > 0:
                    print(f"处理注意力层分类器...")
                    # 首先检查维度
                    data_shape = np.array(results['last_token_attention'][0]).shape
                    print(f"注意力层数据形状: {data_shape}")
                    
                    if len(data_shape) == 0 or (len(data_shape) == 1 and data_shape[0] == 0):
                        print("警告: 注意力层数据为空，跳过")
                    else:
                        # 如果是单层数据，直接处理
                        if len(data_shape) == 1:
                            layer_data = np.stack(results['last_token_attention'])
                            layer_roc, layer_acc = gen_classifier_roc(layer_data, labels)
                            classifier_results[f'last_token_attention_roc'] = layer_roc
                            classifier_results[f'last_token_attention_acc'] = layer_acc
                            print(f"注意力层 ROC: {layer_roc:.4f}, 准确率: {layer_acc:.4f}")
                        else:
                            # 逐层处理
                            for layer in range(data_shape[0]):
                                layer_data = np.stack([i[layer] for i in results['last_token_attention']])
                                layer_roc, layer_acc = gen_classifier_roc(layer_data, labels)
                                classifier_results[f'last_token_attention_roc_{layer}'] = layer_roc
                                classifier_results[f'last_token_attention_acc_{layer}'] = layer_acc
                                print(f"注意力层 {layer} ROC: {layer_roc:.4f}, 准确率: {layer_acc:.4f}")
                
                all_results[str(results_file)] = classifier_results.copy()
                print(f"处理文件 {results_file} 完成")
                
            except Exception as err:
                import traceback
                print(f"处理文件 {results_file} 时出错: {err}")
                print(traceback.format_exc())  # 打印完整的堆栈跟踪
    
    # 保存结果
    with open('./results/qwen_classifier_results.pickle', 'wb') as f:
        pickle.dump(all_results, f)
    
    # 打印结果摘要
    print("\n分类结果摘要:")
    for k, v in all_results.items():
        print(f"\n文件: {k}")
        for metric, value in v.items():
            print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main() 