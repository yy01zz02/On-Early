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
        gru_out, _ = self.gru(seq)  # 修正了原代码中的self.lstm错误为self.gru
        return self.linear(gru_out[:, -1, :])  # 修正了原代码中的索引错误

# 生成分类器的ROC
def gen_classifier_roc(inputs, labels):
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels.astype(int), test_size=0.2, random_state=123)
    classifier_model = FFHallucinationClassifier(X_train.shape[1]).to(device)
    X_train = torch.tensor(X_train).to(device)
    y_train = torch.tensor(y_train).to(torch.long).to(device)
    X_test = torch.tensor(X_test).to(device)
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
    inference_results = list(Path("./results/").rglob("*.pickle"))
    print("找到结果文件:", inference_results)
    
    all_results = {}
    
    for idx, results_file in enumerate(tqdm(inference_results)):
        if results_file not in all_results.keys():
            try:
                classifier_results = {}
                with open(results_file, "rb") as infile:
                    results = pickle.loads(infile.read())
                
                # 检查结果数据结构是否符合预期
                if 'label' not in results:
                    print(f"警告: 文件 {results_file} 中没有找到'label'字段")
                    continue
                
                labels = np.array(results['label'])
                
                # 处理属性归因
                if 'attributes_last_token' in results:
                    X_train, X_test, y_train, y_test = train_test_split(
                        results['attributes_last_token'], 
                        labels.astype(int), 
                        test_size=0.2, 
                        random_state=1234
                    )
                    
                    rnn_model = RNNHallucinationClassifier()
                    optimizer = torch.optim.AdamW(rnn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                    
                    for step in range(1001):
                        # 确保有足够的样本进行采样
                        sample_size = min(batch_size, len(X_train))
                        x_sub, y_sub = zip(*random.sample(list(zip(X_train, y_train)), sample_size))
                        y_sub = torch.tensor(y_sub).to(torch.long)
                        optimizer.zero_grad()
                        preds = torch.stack([rnn_model(torch.tensor(i).view(1, -1, 1).to(torch.float)) for i in x_sub])
                        loss = torch.nn.functional.cross_entropy(preds, y_sub)
                        loss.backward()
                        optimizer.step()
                    
                    preds = torch.stack([rnn_model(torch.tensor(i).view(1, -1, 1).to(torch.float)) for i in X_test])
                    preds = torch.nn.functional.softmax(preds, dim=1)
                    prediction_classes = (preds[:,1]>0.5).type(torch.long).cpu()
                    
                    classifier_results['last_token_attribution_rnn_roc'] = roc_auc_score(y_test, preds[:,1].detach().cpu().numpy())
                    classifier_results['last_token_attribution_rnn_acc'] = (prediction_classes.numpy()==y_test).mean()
                
                # 处理logits
                if 'logits' in results and 'last_token_pos' in results:
                    last_token_logits = np.stack([
                        sp.special.softmax(i[j]) for i, j in zip(results['logits'], results['last_token_pos'])
                    ])
                    last_token_logits_roc, last_token_logits_acc = gen_classifier_roc(last_token_logits, labels)
                    classifier_results['last_token_logits_roc'] = last_token_logits_roc
                    classifier_results['last_token_logits_acc'] = last_token_logits_acc
                
                # 处理全连接层
                if 'last_token_fully_connected' in results:
                    for layer in range(results['last_token_fully_connected'][0].shape[0]):
                        layer_data = np.stack([i[layer] for i in results['last_token_fully_connected']])
                        layer_roc, layer_acc = gen_classifier_roc(layer_data, labels)
                        classifier_results[f'last_token_fully_connected_roc_{layer}'] = layer_roc
                        classifier_results[f'last_token_fully_connected_acc_{layer}'] = layer_acc
                
                # 处理注意力层
                if 'last_token_attention' in results:
                    for layer in range(results['last_token_attention'][0].shape[0]):
                        layer_data = np.stack([i[layer] for i in results['last_token_attention']])
                        layer_roc, layer_acc = gen_classifier_roc(layer_data, labels)
                        classifier_results[f'last_token_attention_roc_{layer}'] = layer_roc
                        classifier_results[f'last_token_attention_acc_{layer}'] = layer_acc
                
                all_results[str(results_file)] = classifier_results.copy()
                print(f"处理文件 {results_file} 完成")
                
            except Exception as err:
                print(f"处理文件 {results_file} 时出错: {err}")
    
    # 保存结果
    with open('./results/classifier_results.pickle', 'wb') as f:
        pickle.dump(all_results, f)
    
    # 打印结果摘要
    print("\n分类结果摘要:")
    for k, v in all_results.items():
        print(f"\n文件: {k}")
        for metric, value in v.items():
            print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main() 