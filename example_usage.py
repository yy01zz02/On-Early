import numpy as np
import pickle
import os
import sys
import torch
import torch.serialization
from pathlib import Path

# 允许numpy.core.multiarray.scalar作为可信全局变量
torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])

# 导入模型预测器
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_predictor import ModelPredictor, predict_single_sample

def load_example_data(pickle_file):
    """加载示例数据"""
    print(f"从 {pickle_file} 加载示例数据...")
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    # 打印数据基本信息
    print(f"数据包含 {len(data['label'])} 个样本")
    print(f"标签分布: 正例={sum(data['label'])}, 负例={len(data['label']) - sum(data['label'])}")
    
    # 验证数据结构
    expected_keys = ['code', 'label', 'last_token_pos', 
                     'last_token_softmax', 'attributes_last_token',
                     'last_token_attention', 'last_token_fully_connected']
    
    for key in expected_keys:
        if key in data:
            print(f"找到特征: {key} - 样本数: {len(data[key])}")
        else:
            print(f"警告: 未找到特征 {key}")
    
    return data

def predict_example(predictor, data, index=None):
    """预测一个示例"""
    if index is None:
        # 随机选择一个样本
        index = np.random.randint(0, len(data['label']))
    
    print(f"\n预测样本 {index} (标签: {'正例' if data['label'][index] == 1 else '负例'})")
    
    # 提取该样本的所有特征
    features_dict = {}
    if 'last_token_softmax' in data:
        features_dict['softmax'] = data['last_token_softmax'][index]
    
    if 'attributes_last_token' in data and len(data['attributes_last_token'][index]) > 0:
        features_dict['attributes'] = data['attributes_last_token'][index]
    
    if 'last_token_fully_connected' in data:
        features_dict['fully_connected'] = data['last_token_fully_connected'][index]
    
    if 'last_token_attention' in data:
        features_dict['attention'] = data['last_token_attention'][index]
    
    # 打印代码片段
    if 'code' in data:
        code = data['code'][index]
        print("\n代码片段:")
        print("-" * 50)
        print(code[:500] + "..." if len(code) > 500 else code)
        print("-" * 50)
    
    # 进行预测
    results = predict_single_sample(predictor, features_dict)
    
    # 打印详细结果
    if results:
        print("\n预测结果摘要:")
        best_prob = 0
        best_model = None
        
        for model_type, probs in results.items():
            if model_type != 'ensemble':
                if probs[1] > best_prob:
                    best_prob = probs[1]
                    best_model = model_type
        
        if best_model:
            print(f"最可信的单一模型: {best_model} (幻觉概率: {best_prob:.4f})")
        
        if 'ensemble' in results:
            ensemble_prob = results['ensemble'][1]
            print(f"集成模型: 幻觉概率 = {ensemble_prob:.4f}")
            
            # 最终判断
            is_hallucination = ensemble_prob > 0.5
            confidence = max(ensemble_prob, 1 - ensemble_prob)
            
            print(f"\n最终判断: {'存在幻觉' if is_hallucination else '无幻觉'} (置信度: {confidence:.4f})")
    else:
        print("\n未能获取有效的预测结果。")
    
    return results

def main():
    """主函数"""
    # 检查命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='模型预测器示例用法')
    parser.add_argument('--pickle_file', type=str, required=True, help='Pickle数据文件路径')
    parser.add_argument('--models_dir', type=str, default='./models', help='模型文件夹路径')
    parser.add_argument('--sample_index', type=int, help='要预测的样本索引 (如不指定则随机选择)')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    
    args = parser.parse_args()
    
    # 设置设备
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    data = load_example_data(args.pickle_file)
    
    # 创建预测器
    predictor = ModelPredictor(models_dir=args.models_dir, device=device)
    
    # 预测示例
    predict_example(predictor, data, args.sample_index)

if __name__ == "__main__":
    main() 