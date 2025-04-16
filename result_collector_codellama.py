import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pickle
import os
from datetime import datetime
from tqdm import tqdm
import random
import json
import pandas as pd
from pathlib import Path
from captum.attr import IntegratedGradients

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载模型和分词器
model_path = "/root/.cache/modelscope/hub/models/AI-ModelScope/CodeLlama-7b-hf"
print(f"正在加载模型: {model_path}")
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

print(f"模型类型: {type(model).__name__}")

# 检查模型结构
num_layers = len(model.model.layers)
print(f"模型层数: {num_layers}")

# 存储特征提取的钩子
fully_connected_forward_handles = {}
attention_forward_handles = {}

# 特征提取函数
def get_fully_connected_features(module, input, output):
    # 获取最后一层的输出
    last_hidden_state = output[0] if isinstance(output, tuple) else output
    # 存储最后一层的特征
    fully_connected_forward_handles['last_hidden_state'] = last_hidden_state.detach()

def get_attention_features(module, input, output):
    # 获取注意力输出
    attention_output = output[0] if isinstance(output, tuple) else output
    # 存储注意力特征
    attention_forward_handles['attention_output'] = attention_output.detach()

# 注册钩子
print("正在注册特征提取钩子...")
for name, module in model.named_modules():
    # 打印一些模块名称以进行调试
    if "down_proj" in name or "o_proj" in name:
        print(f"找到相关模块: {name}")
    
    # CodeLlama的MLP层
    if "mlp.down_proj" in name:
        layer_num = name.split(".")[2]  # 从名称中提取层号
        print(f"为MLP层 {layer_num} 注册钩子: {name}")
        fully_connected_forward_handles[name] = module.register_forward_hook(get_fully_connected_features)
    
    # CodeLlama的注意力层
    if "self_attn.o_proj" in name:
        layer_num = name.split(".")[2]  # 从名称中提取层号
        print(f"为注意力层 {layer_num} 注册钩子: {name}")
        attention_forward_handles[name] = module.register_forward_hook(get_attention_features)

# 加载代码样本
def load_code_samples(file_path):
    if file_path.endswith('.csv'):
        # 从CSV文件加载
        df = pd.read_csv(file_path)
        samples = []
        for _, row in df.iterrows():
            samples.append({
                'code': row['code'],
                'label': row['label']
            })
        return samples
    else:
        # 从JSON文件加载
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

# 归一化Integrated Gradients属性
def normalize_attributes(attributes):
    """归一化Integrated Gradients属性"""
    attributes = attributes.squeeze(0)
    norm = torch.norm(attributes, dim=1)
    attributes = norm / torch.sum(norm)  # 归一化使总和为1
    return attributes

# 模型前向传播函数，用于Integrated Gradients
def model_forward(inputs_embeds, model, input_ids):
    """模型前向传播函数，用于Integrated Gradients"""
    attention_mask = torch.ones_like(input_ids)
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    return torch.nn.functional.softmax(outputs.logits[:, -1, :], dim=-1)

# 特征提取函数
def extract_features(code, label):
    # 对代码进行编码
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs['input_ids'].to(device)
    
    # 获取最后一个token的位置
    last_token_pos = input_ids.shape[1] - 1
    
    # 前向传播
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
    
    # 获取logits并计算softmax概率
    logits = outputs.logits[0, -1].cpu().numpy()
    softmax_probs = torch.nn.functional.softmax(outputs.logits[0, -1], dim=-1).cpu().numpy()
    
    # 获取最后一层的MLP特征
    last_token_fully_connected = []
    for name in fully_connected_forward_handles.keys():
        if name != 'last_hidden_state' and 'last_hidden_state' in fully_connected_forward_handles:
            features = fully_connected_forward_handles['last_hidden_state'][0, -1].cpu().numpy()
            last_token_fully_connected.append(features)
    
    # 获取最后一层的注意力特征
    last_token_attention = []
    for name in attention_forward_handles.keys():
        if name != 'attention_output' and 'attention_output' in attention_forward_handles:
            features = attention_forward_handles['attention_output'][0, -1].cpu().numpy()
            last_token_attention.append(features)
    
    # 清理钩子存储，但保留钩子本身
    if 'last_hidden_state' in fully_connected_forward_handles:
        del fully_connected_forward_handles['last_hidden_state']
    if 'attention_output' in attention_forward_handles:
        del attention_forward_handles['attention_output']
    
    # 计算Integrated Gradients归因
    try:
        # 获取词嵌入
        embedder = model.model.embed_tokens
        input_embeds = embedder(input_ids).detach()
        
        # 目标token的id
        target_token_id = input_ids[0, last_token_pos].item()
        
        # 设置IG参数
        ig_steps = 32  # 积分步数
        ig = IntegratedGradients(model_forward)
        
        # 计算属性
        attributes = ig.attribute(
            input_embeds,
            target=target_token_id,
            additional_forward_args=(model, input_ids),
            n_steps=ig_steps
        )
        
        # 归一化属性
        normalized_attrs = normalize_attributes(attributes).detach().cpu().numpy()
    except Exception as e:
        print(f"计算IG归因时出错: {e}")
        normalized_attrs = np.zeros(last_token_pos + 1)  # 创建一个空的归因数组
    
    return {
        'code': code,
        'label': label,
        'last_token_pos': last_token_pos,
        'last_token_softmax': softmax_probs,
        'last_token_fully_connected': np.array(last_token_fully_connected) if last_token_fully_connected else np.array([]),
        'last_token_attention': np.array(last_token_attention) if last_token_attention else np.array([]),
        'attributes_last_token': normalized_attrs
    }

# 主函数
def main():
    # 设置随机种子
    set_seed(42)
    
    # 加载代码样本
    data_file = 'merged_data.csv'
    print(f"正在从 {data_file} 加载代码样本...")
    code_samples = load_code_samples(data_file)
    print(f"加载了 {len(code_samples)} 个代码样本")
    
    # 提取特征
    results = []
    for sample in tqdm(code_samples, desc="提取特征"):
        features = extract_features(sample['code'], sample['label'])
        results.append(features)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'results/codellama_code_results_{timestamp}.pickle'
    os.makedirs('results', exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"结果已保存到: {save_path}")

if __name__ == "__main__":
    main() 