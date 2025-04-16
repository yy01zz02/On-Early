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
import pandas as pd
from pathlib import Path
from captum.attr import IntegratedGradients
from functools import partial

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

# Integrated Gradients参数
ig_steps = 64
internal_batch_size = 4

# 存储特征提取的钩子
fully_connected_forward_handles = {}
attention_forward_handles = {}

# 特征提取函数
def get_fully_connected_features(module, input, output):
    # 获取全连接层的输出
    fc_output = output[0] if isinstance(output, tuple) else output
    # 存储激活值
    fully_connected_forward_handles['fc_output'] = fc_output.detach()

def get_attention_features(module, input, output):
    # 获取注意力层的输出
    attn_output = output[0] if isinstance(output, tuple) else output
    # 存储激活值
    attention_forward_handles['attn_output'] = attn_output.detach()

# 获取词嵌入层
def get_embedder(model):
    return model.model.embed_tokens

# 模型前向传播函数，用于Integrated Gradients
def model_forward(input_, model, extra_forward_args=None):
    outputs = model(inputs_embeds=input_)
    return torch.nn.functional.softmax(outputs.logits[:, -1, :], dim=-1)

# 归一化Integrated Gradients属性
def normalize_attributes(attributes):
    attributes = attributes.squeeze(0)
    norm = torch.norm(attributes, dim=1)
    attributes = norm / torch.sum(norm)  # 归一化为和为1
    return attributes

# 计算Integrated Gradients属性
def get_ig(code_content, token_pos, forward_func, tokenizer, embedder, model):
    input_ids = tokenizer.encode(code_content, return_tensors='pt').to(device)
    
    # 对目标位置的token进行归因分析
    target_token_id = input_ids[0, token_pos].item()
    
    # 获取词嵌入
    encoder_input_embeds = embedder(input_ids).detach()
    
    # 计算Integrated Gradients
    ig = IntegratedGradients(forward_func=forward_func)
    attributes = normalize_attributes(
        ig.attribute(
            encoder_input_embeds,
            target=target_token_id,
            n_steps=ig_steps,
            internal_batch_size=internal_batch_size
        )
    ).detach().cpu().numpy()
    
    return attributes

def main():
    # 设置随机种子
    set_seed(42)
    
    # 加载模型和分词器
    model_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Coder-7B"
    print(f"正在加载模型: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    print(f"模型类型: {type(model).__name__}")

    # 检查模型结构
    num_layers = len(model.model.layers)
    print(f"模型层数: {num_layers}")
    
    # 初始化embedder和forward_func
    embedder = get_embedder(model)
    forward_func = partial(model_forward, model=model)
    
    # 注册钩子
    print("正在注册特征提取钩子...")
    
    # 只注册最后一层的钩子
    layer_idx = num_layers - 1
    
    # 全连接层钩子
    fc_name = f"model.layers.{layer_idx}.mlp.down_proj"
    for name, module in model.named_modules():
        if fc_name in name:
            print(f"为全连接层注册钩子: {name}")
            fully_connected_forward_handles[name] = module.register_forward_hook(get_fully_connected_features)
            break
    
    # 注意力层钩子
    attn_name = f"model.layers.{layer_idx}.self_attn.o_proj"
    for name, module in model.named_modules():
        if attn_name in name:
            print(f"为注意力层注册钩子: {name}")
            attention_forward_handles[name] = module.register_forward_hook(get_attention_features)
            break
    
    # 加载代码样本
    data_file = 'merged_data.csv'
    print(f"正在从 {data_file} 加载代码样本...")
    df = pd.read_csv(data_file)
    code_samples = []
    for _, row in df.iterrows():
        code_samples.append({
            'code': row['code'],
            'label': row['label']
        })
    print(f"加载了 {len(code_samples)} 个代码样本")
    
    # 提取特征
    results = {
        'code': [],
        'label': [],
        'last_token_pos': [],
        'last_token_softmax': [],  # 最后一个token的softmax概率
        'attributes_last_token': [],  # IG特征归因
        'last_token_attention': [],  # 注意力得分
        'last_token_fully_connected': []  # 全连接层激活
    }
    
    for sample in tqdm(code_samples, desc="提取特征"):
        code = sample['code']
        label = sample['label']
        
        # 编码输入
        inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 获取最后一个token的位置
        last_token_pos = inputs['input_ids'].shape[1] - 1
        
        # 前向传播
        with torch.no_grad():
            outputs = model(**inputs)
            
            # 1. 获取softmax概率
            last_token_logits = outputs.logits[0, -1]
            last_token_softmax = F.softmax(last_token_logits, dim=-1).cpu().numpy()
            
            # 2. 获取全连接层激活 - 如果钩子已注册
            if 'fc_output' in fully_connected_forward_handles:
                fc_features = fully_connected_forward_handles['fc_output'][0, -1].cpu().numpy()
            else:
                fc_features = np.array([])
                
            # 3. 获取注意力得分 - 如果钩子已注册
            if 'attn_output' in attention_forward_handles:
                attn_features = attention_forward_handles['attn_output'][0, -1].cpu().numpy()
            else:
                attn_features = np.array([])
        
        # 4. 计算Integrated Gradients属性
        attributes = get_ig(code, last_token_pos, forward_func, tokenizer, embedder, model)
        
        # 存储结果
        results['code'].append(code)
        results['label'].append(label)
        results['last_token_pos'].append(last_token_pos)
        results['last_token_softmax'].append(last_token_softmax)
        results['attributes_last_token'].append(attributes)
        results['last_token_attention'].append(attn_features)
        results['last_token_fully_connected'].append(fc_features)
        
        # 清理钩子存储
        if 'fc_output' in fully_connected_forward_handles:
            del fully_connected_forward_handles['fc_output']
        if 'attn_output' in attention_forward_handles:
            del attention_forward_handles['attn_output']
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'results/qwen_coder_results_{timestamp}.pickle'
    os.makedirs('results', exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"结果已保存到: {save_path}")

if __name__ == "__main__":
    main()