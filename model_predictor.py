import torch
import numpy as np
import os
import glob
from pathlib import Path
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score
import pickle
import torch.serialization

# 允许numpy.core.multiarray.scalar作为可信全局变量
torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])

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
    """
    预测器类，用于加载和运行所有预训练模型
    """
    def __init__(self, models_dir='./models', device=None):
        """
        初始化预测器
        
        参数:
            models_dir: 包含所有模型的目录
            device: 计算设备 (例如 'cuda', 'cpu')
        """
        self.models_dir = Path(models_dir)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载所有可用模型
        self.models = self.load_models()
        
        # 加载模型ROC信息
        self.model_rocs = self.load_model_rocs()
        
        # 如果至少有一个模型加载成功，则初始化成功
        self.is_ready = len(self.models) > 0
        if not self.is_ready:
            print("警告: 没有成功加载任何模型，预测可能不可用。")
        else:
            print(f"成功加载了 {len(self.models)} 个模型。")
    
    def safe_load_model(self, path):
        """
        安全加载模型，兼容PyTorch不同版本
        """
        # 首先尝试使用weights_only=False加载
        try:
            return torch.load(path, map_location=self.device)
        except TypeError as e:
            # 如果是weights_only相关的错误，尝试不使用该参数
            if 'weights_only' in str(e):
                # torch低版本没有weights_only参数
                print(f"尝试兼容性模型加载: {path}")
                return torch.load(path, map_location=self.device)
            else:
                # 其他类型错误直接抛出
                raise
        except Exception as e:
            print(f"加载模型 {path} 时出错: {str(e)}")
            return None
    
    def load_models(self):
        """
        加载所有可用模型
        """
        models = {}
        
        # 模型类型和对应的文件名
        model_types = {
            'softmax': 'softmax_model.pt',
            'attributes': 'attributes_model.pt',
            'fully_connected': 'fully_connected_model.pt',
            'attention': 'attention_model.pt',
        }
        
        # 检查models目录是否存在
        if not self.models_dir.exists():
            print(f"警告: 模型目录 {self.models_dir} 不存在")
            return models
        
        # 尝试加载每种类型的模型
        for model_type, filename in model_types.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                try:
                    print(f"加载模型: {model_path}...")
                    model = self.safe_load_model(model_path)
                    if model is not None:
                        model.eval()  # 设置为评估模式
                        models[model_type] = model
                        print(f"成功加载模型: {model_type}")
                    else:
                        print(f"警告: 无法加载模型 {model_type}")
                except Exception as e:
                    print(f"加载模型 {model_type} 时出错: {str(e)}")
            else:
                print(f"模型文件不存在: {model_path}")
        
        return models
    
    def load_model_rocs(self):
        """
        加载模型ROC信息 (用于集成预测)
        """
        roc_file = self.models_dir / 'model_rocs.pickle'
        if roc_file.exists():
            try:
                with open(roc_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"加载ROC数据时出错: {str(e)}")
                return {}
        else:
            print(f"ROC数据文件不存在: {roc_file}")
            return {}
    
    def predict(self, features_dict):
        """
        使用所有可用模型进行预测
        
        参数:
            features_dict: 包含不同特征的字典，键为特征类型，值为特征向量
                           可能的键: 'softmax', 'attributes', 'fully_connected', 'attention'
        
        返回:
            predictions: 包含各个模型预测结果的字典，键为模型类型，值为预测概率 [class_0_prob, class_1_prob]
        """
        if not self.is_ready:
            print("预测器未准备就绪。")
            return None
        
        predictions = {}
        
        # 使用每个可用模型进行预测
        for model_type, model in self.models.items():
            # 检查是否有该模型对应的特征
            if model_type in features_dict and features_dict[model_type] is not None:
                feature = features_dict[model_type]
                
                # 将特征转换为张量并移动到正确的设备
                if isinstance(feature, np.ndarray):
                    feature_tensor = torch.FloatTensor(feature).to(self.device)
                else:
                    feature_tensor = feature.to(self.device)
                
                # 添加批次维度（如果需要）
                if len(feature_tensor.shape) == 1:
                    feature_tensor = feature_tensor.unsqueeze(0)
                
                # 进行预测
                with torch.no_grad():
                    output = model(feature_tensor)
                    probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()
                    predictions[model_type] = probabilities
        
        # 如果有多个模型预测结果，添加集成预测
        if len(predictions) > 1:
            ensemble_prediction = self.ensemble_predictions(predictions)
            if ensemble_prediction is not None:
                predictions['ensemble'] = ensemble_prediction
        
        return predictions
    
    def ensemble_predictions(self, predictions):
        """
        基于各个模型的ROC性能，集成多个模型的预测结果
        
        参数:
            predictions: 包含各个模型预测结果的字典
        
        返回:
            ensemble_probs: 集成后的预测概率 [class_0_prob, class_1_prob]
        """
        # 如果没有ROC数据，使用简单平均
        if not self.model_rocs:
            probs = np.array([pred for pred in predictions.values() if pred is not None])
            return np.mean(probs, axis=0)
        
        # 基于ROC权重进行集成
        weighted_probs = []
        weights = []
        
        for model_type, probs in predictions.items():
            if model_type in self.model_rocs:
                roc = self.model_rocs.get(model_type, 0.5)  # 默认ROC为0.5
                # 只有当ROC比随机猜测好时才使用该模型
                if roc > 0.55:
                    weighted_probs.append(probs * (roc - 0.5))
                    weights.append(roc - 0.5)
        
        # 如果没有足够好的模型，返回None
        if not weighted_probs:
            # 退化为简单平均
            probs = np.array([pred for pred in predictions.values() if pred is not None])
            return np.mean(probs, axis=0)
        
        # 计算加权平均
        weighted_sum = np.sum(weighted_probs, axis=0)
        weight_sum = np.sum(weights)
        
        # 归一化
        if weight_sum > 0:
            return weighted_sum / weight_sum
        else:
            # 退化为简单平均
            probs = np.array([pred for pred in predictions.values() if pred is not None])
            return np.mean(probs, axis=0)

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
        valid_indices = []
        for i, feature in enumerate(test_data['attributes_last_token']):
            if len(feature) > 0:  # 确保特征不为空
                try:
                    pred = predictor.predict_attributes(feature)
                    predictions.append(pred[1])  # 取正类概率
                    valid_indices.append(i)
                except Exception as e:
                    print(f"样本 {i} 预测失败: {e}")
        
        if predictions:
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
    """
    使用给定的预测器对单个样本进行预测
    
    参数:
        predictor: ModelPredictor实例
        features_dict: 特征字典
    
    返回:
        predictions: 预测结果字典
    """
    if not predictor.is_ready:
        print("预测器未准备就绪，无法进行预测。")
        return None
    
    # 验证特征
    valid_features = False
    for feature_type in ['softmax', 'attributes', 'fully_connected', 'attention']:
        if feature_type in features_dict and features_dict[feature_type] is not None:
            valid_features = True
            break
    
    if not valid_features:
        print("错误: 没有有效的特征可用于预测")
        return None
    
    # 进行预测
    try:
        predictions = predictor.predict(features_dict)
        return predictions
    except Exception as e:
        print(f"预测过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

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
                roc = predictor.model_rocs.get(feature_type, 'N/A')
                print(f"- {feature_type}: 加载成功 (ROC: {roc})")
            else:
                print(f"- {feature_type}: 未加载")

if __name__ == "__main__":
    main() 