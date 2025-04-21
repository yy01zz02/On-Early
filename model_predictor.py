import torch
import numpy as np
import os
import glob
from pathlib import Path
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score
import pickle

# 定义模型类，与分类器训练文件中相同
class FFHallucinationClassifier(torch.nn.Module):
    def __init__(self, input_shape, dropout=0.5):
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

class RNNHallucinationClassifier(torch.nn.Module):
    def __init__(self, dropout=0.25):
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

class ModelPredictor:
    def __init__(self, models_dir='./models', device=None):
        self.models_dir = models_dir
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {
            'softmax': None,
            'attributes': None, 
            'fully_connected': None,
            'attention': None
        }
        self.load_models()
    
    def find_best_model(self, feature_type):
        """查找特定特征类型的最佳模型文件"""
        if feature_type == 'attributes':
            pattern = os.path.join(self.models_dir, f"codellama_rnn_{feature_type}*.pt")
        else:
            pattern = os.path.join(self.models_dir, f"codellama_ff_last_token_{feature_type}*.pt")
        
        model_files = glob.glob(pattern)
        if not model_files:
            return None
        
        # 按创建时间排序，获取最新的模型
        latest_model = max(model_files, key=os.path.getctime)
        return latest_model
    
    def load_models(self):
        """加载所有模型"""
        # 加载softmax模型
        softmax_model_file = self.find_best_model('softmax')
        if softmax_model_file:
            print(f"加载Softmax模型: {softmax_model_file}")
            model_info = torch.load(softmax_model_file, map_location=self.device)
            model = FFHallucinationClassifier(model_info['input_shape']).to(self.device)
            model.load_state_dict(model_info['model_state_dict'])
            model.eval()
            self.models['softmax'] = {
                'model': model,
                'info': model_info
            }
        
        # 加载attributes模型
        attributes_model_file = self.find_best_model('attributes')
        if attributes_model_file:
            print(f"加载IG属性归因模型: {attributes_model_file}")
            model_info = torch.load(attributes_model_file, map_location=self.device)
            model = RNNHallucinationClassifier().to(self.device)
            model.load_state_dict(model_info['model_state_dict'])
            model.eval()
            self.models['attributes'] = {
                'model': model,
                'info': model_info
            }
        
        # 加载fully_connected模型
        fc_model_file = self.find_best_model('fully_connected')
        if fc_model_file:
            print(f"加载全连接层模型: {fc_model_file}")
            model_info = torch.load(fc_model_file, map_location=self.device)
            model = FFHallucinationClassifier(model_info['input_shape']).to(self.device)
            model.load_state_dict(model_info['model_state_dict'])
            model.eval()
            self.models['fully_connected'] = {
                'model': model,
                'info': model_info
            }
        
        # 加载attention模型
        attn_model_file = self.find_best_model('attention')
        if attn_model_file:
            print(f"加载注意力层模型: {attn_model_file}")
            model_info = torch.load(attn_model_file, map_location=self.device)
            model = FFHallucinationClassifier(model_info['input_shape']).to(self.device)
            model.load_state_dict(model_info['model_state_dict'])
            model.eval()
            self.models['attention'] = {
                'model': model,
                'info': model_info
            }
    
    def predict_softmax(self, feature):
        """使用Softmax概率特征预测"""
        if self.models['softmax'] is None:
            raise ValueError("Softmax模型未加载")
        
        model = self.models['softmax']['model']
        feature_tensor = torch.tensor(feature, dtype=torch.float).to(self.device)
        
        with torch.no_grad():
            logits = model(feature_tensor.unsqueeze(0))
            probs = torch.nn.functional.softmax(logits, dim=1)
        
        return probs[0].cpu().numpy()
    
    def predict_attributes(self, feature):
        """使用IG属性归因特征预测"""
        if self.models['attributes'] is None:
            raise ValueError("IG属性归因模型未加载")
        
        model = self.models['attributes']['model']
        feature_tensor = torch.tensor(feature, dtype=torch.float).view(-1, 1).to(self.device)
        
        with torch.no_grad():
            logits = model(feature_tensor)
            probs = torch.nn.functional.softmax(logits, dim=0 if len(logits.shape) == 1 else 1)
        
        return probs.cpu().numpy() if len(probs.shape) == 1 else probs[0].cpu().numpy()
    
    def predict_fully_connected(self, feature):
        """使用全连接层特征预测"""
        if self.models['fully_connected'] is None:
            raise ValueError("全连接层模型未加载")
        
        model = self.models['fully_connected']['model']
        feature_tensor = torch.tensor(feature, dtype=torch.float).to(self.device)
        
        with torch.no_grad():
            logits = model(feature_tensor.unsqueeze(0))
            probs = torch.nn.functional.softmax(logits, dim=1)
        
        return probs[0].cpu().numpy()
    
    def predict_attention(self, feature):
        """使用注意力层特征预测"""
        if self.models['attention'] is None:
            raise ValueError("注意力层模型未加载")
        
        model = self.models['attention']['model']
        feature_tensor = torch.tensor(feature, dtype=torch.float).to(self.device)
        
        with torch.no_grad():
            logits = model(feature_tensor.unsqueeze(0))
            probs = torch.nn.functional.softmax(logits, dim=1)
        
        return probs[0].cpu().numpy()
    
    def ensemble_predict(self, features_dict):
        """结合所有特征进行集成预测"""
        probs = []
        weights = []
        
        # 使用各个模型的ROC作为权重
        if 'softmax' in features_dict and self.models['softmax']:
            probs.append(self.predict_softmax(features_dict['softmax']))
            weights.append(self.models['softmax']['info']['roc'])
        
        if 'attributes' in features_dict and self.models['attributes']:
            probs.append(self.predict_attributes(features_dict['attributes']))
            weights.append(self.models['attributes']['info']['roc'])
        
        if 'fully_connected' in features_dict and self.models['fully_connected']:
            probs.append(self.predict_fully_connected(features_dict['fully_connected']))
            weights.append(self.models['fully_connected']['info']['roc'])
        
        if 'attention' in features_dict and self.models['attention']:
            probs.append(self.predict_attention(features_dict['attention']))
            weights.append(self.models['attention']['info']['roc'])
        
        if not probs:
            raise ValueError("没有可用的模型进行预测")
        
        # 加权平均
        weights = np.array(weights) / sum(weights)
        weighted_probs = np.zeros_like(probs[0])
        for i, p in enumerate(probs):
            weighted_probs += p * weights[i]
        
        return weighted_probs

def test_models(predictor, test_file):
    """测试模型在测试数据上的表现"""
    print(f"测试模型使用: {test_file}")
    with open(test_file, 'rb') as f:
        test_data = pickle.load(f)
    
    labels = np.array(test_data['label'])
    
    # 测试各个模型
    results = {}
    
    # 1. 测试Softmax模型
    if 'last_token_softmax' in test_data and predictor.models['softmax']:
        print("测试Softmax概率模型...")
        predictions = []
        for feature in test_data['last_token_softmax']:
            pred = predictor.predict_softmax(feature)
            predictions.append(pred[1])  # 取正类概率
        
        auc = roc_auc_score(labels, predictions)
        predictions_binary = np.array(predictions) > 0.5
        acc = accuracy_score(labels, predictions_binary)
        results['softmax'] = {'auc': auc, 'accuracy': acc}
        print(f"Softmax模型 - AUC: {auc:.4f}, 准确率: {acc:.4f}")
    
    # 2. 测试IG属性归因模型
    if 'attributes_last_token' in test_data and predictor.models['attributes']:
        print("测试IG属性归因模型...")
        predictions = []
        for feature in test_data['attributes_last_token']:
            if len(feature) > 0:  # 确保特征不为空
                pred = predictor.predict_attributes(feature)
                predictions.append(pred[1])  # 取正类概率
        
        if predictions:
            valid_indices = [i for i, feat in enumerate(test_data['attributes_last_token']) if len(feat) > 0]
            valid_labels = labels[valid_indices]
            auc = roc_auc_score(valid_labels, predictions)
            predictions_binary = np.array(predictions) > 0.5
            acc = accuracy_score(valid_labels, predictions_binary)
            results['attributes'] = {'auc': auc, 'accuracy': acc}
            print(f"IG属性归因模型 - AUC: {auc:.4f}, 准确率: {acc:.4f}")
    
    # 3. 测试全连接层模型
    if 'last_token_fully_connected' in test_data and predictor.models['fully_connected']:
        print("测试全连接层模型...")
        predictions = []
        for feature in test_data['last_token_fully_connected']:
            pred = predictor.predict_fully_connected(feature)
            predictions.append(pred[1])  # 取正类概率
        
        auc = roc_auc_score(labels, predictions)
        predictions_binary = np.array(predictions) > 0.5
        acc = accuracy_score(labels, predictions_binary)
        results['fully_connected'] = {'auc': auc, 'accuracy': acc}
        print(f"全连接层模型 - AUC: {auc:.4f}, 准确率: {acc:.4f}")
    
    # 4. 测试注意力层模型
    if 'last_token_attention' in test_data and predictor.models['attention']:
        print("测试注意力层模型...")
        predictions = []
        for feature in test_data['last_token_attention']:
            pred = predictor.predict_attention(feature)
            predictions.append(pred[1])  # 取正类概率
        
        auc = roc_auc_score(labels, predictions)
        predictions_binary = np.array(predictions) > 0.5
        acc = accuracy_score(labels, predictions_binary)
        results['attention'] = {'auc': auc, 'accuracy': acc}
        print(f"注意力层模型 - AUC: {auc:.4f}, 准确率: {acc:.4f}")
    
    # 5. 测试集成模型
    print("测试集成模型...")
    predictions = []
    for i in range(len(labels)):
        features_dict = {}
        if 'last_token_softmax' in test_data and predictor.models['softmax']:
            features_dict['softmax'] = test_data['last_token_softmax'][i]
        
        if 'attributes_last_token' in test_data and predictor.models['attributes'] and len(test_data['attributes_last_token'][i]) > 0:
            features_dict['attributes'] = test_data['attributes_last_token'][i]
        
        if 'last_token_fully_connected' in test_data and predictor.models['fully_connected']:
            features_dict['fully_connected'] = test_data['last_token_fully_connected'][i]
        
        if 'last_token_attention' in test_data and predictor.models['attention']:
            features_dict['attention'] = test_data['last_token_attention'][i]
        
        if features_dict:
            try:
                pred = predictor.ensemble_predict(features_dict)
                predictions.append(pred[1])  # 取正类概率
            except Exception as e:
                print(f"样本 {i} 集成预测失败: {e}")
                # 使用首选模型（全连接层或注意力层）回退
                if 'fully_connected' in features_dict:
                    pred = predictor.predict_fully_connected(features_dict['fully_connected'])
                    predictions.append(pred[1])
                elif 'attention' in features_dict:
                    pred = predictor.predict_attention(features_dict['attention'])
                    predictions.append(pred[1])
                else:
                    predictions.append(0.5)  # 默认预测
    
    if predictions:
        auc = roc_auc_score(labels, predictions)
        predictions_binary = np.array(predictions) > 0.5
        acc = accuracy_score(labels, predictions_binary)
        results['ensemble'] = {'auc': auc, 'accuracy': acc}
        print(f"集成模型 - AUC: {auc:.4f}, 准确率: {acc:.4f}")
    
    return results

def predict_single_sample(predictor, features_dict):
    """预测单个样本"""
    results = {}
    
    # 单独的模型预测
    if 'softmax' in features_dict and predictor.models['softmax']:
        results['softmax'] = predictor.predict_softmax(features_dict['softmax'])
        print(f"Softmax模型预测 - 负类: {results['softmax'][0]:.4f}, 正类: {results['softmax'][1]:.4f}")
    
    if 'attributes' in features_dict and predictor.models['attributes']:
        results['attributes'] = predictor.predict_attributes(features_dict['attributes'])
        print(f"IG属性归因模型预测 - 负类: {results['attributes'][0]:.4f}, 正类: {results['attributes'][1]:.4f}")
    
    if 'fully_connected' in features_dict and predictor.models['fully_connected']:
        results['fully_connected'] = predictor.predict_fully_connected(features_dict['fully_connected'])
        print(f"全连接层模型预测 - 负类: {results['fully_connected'][0]:.4f}, 正类: {results['fully_connected'][1]:.4f}")
    
    if 'attention' in features_dict and predictor.models['attention']:
        results['attention'] = predictor.predict_attention(features_dict['attention'])
        print(f"注意力层模型预测 - 负类: {results['attention'][0]:.4f}, 正类: {results['attention'][1]:.4f}")
    
    # 集成预测
    try:
        results['ensemble'] = predictor.ensemble_predict(features_dict)
        print(f"集成模型预测 - 负类: {results['ensemble'][0]:.4f}, 正类: {results['ensemble'][1]:.4f}")
    except Exception as e:
        print(f"集成预测失败: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='运行CodeLlama幻觉检测模型预测')
    parser.add_argument('--models_dir', type=str, default='./models', help='模型文件夹路径')
    parser.add_argument('--test_file', type=str, help='测试数据文件路径')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    
    args = parser.parse_args()
    
    # 设置设备
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建预测器
    predictor = ModelPredictor(models_dir=args.models_dir, device=device)
    
    # 如果提供了测试文件，则测试模型
    if args.test_file:
        test_models(predictor, args.test_file)
    else:
        # 交互式测试
        print("\n没有提供测试文件。请使用以下命令测试模型:")
        print("python model_predictor.py --test_file path/to/test_file.pickle")
        print("\n可用模型:")
        for feature_type, model_info in predictor.models.items():
            if model_info:
                print(f"- {feature_type}: 加载成功 (ROC: {model_info['info']['roc']:.4f})")
            else:
                print(f"- {feature_type}: 未加载")

if __name__ == "__main__":
    main() 