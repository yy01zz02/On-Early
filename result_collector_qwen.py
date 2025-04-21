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
from scipy import sparse

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

# Integrated Gradients参数 - 减少计算步骤以提高效率
ig_steps = 16  # 减少步数
internal_batch_size = 8  # 增加批量大小
ig_sample_rate = 5  # 每5个样本计算一次IG

# 存储特征提取的钩子
features_dict = {
    'fc_output': None,
    'attn_output': None
}

# 特征提取函数
def get_fully_connected_features(module, input, output):
    # 获取全连接层的输出
    fc_output = output[0] if isinstance(output, tuple) else output
    # 存储激活值
    features_dict['fc_output'] = fc_output.detach()

def get_attention_features(module, input, output):
    # 获取注意力层的输出
    attn_output = output[0] if isinstance(output, tuple) else output
    # 存储激活值
    features_dict['attn_output'] = attn_output.detach()

# 获取词嵌入层 - 对于Qwen2.5模型，嵌入层路径是model.model.embed_tokens
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

# 注册钩子函数
def register_hooks(model, layer_idx):
    hooks = []
    
    # 全连接层钩子 - Qwen2.5模型的MLP输出投影层路径
    fc_name = f"model.layers.{layer_idx}.mlp.down_proj"
    for name, module in model.named_modules():
        if fc_name in name:
            print(f"为全连接层注册钩子: {name}")
            hooks.append(module.register_forward_hook(get_fully_connected_features))
            break
    
    # 注意力层钩子 - Qwen2.5模型的自注意力输出投影层路径
    attn_name = f"model.layers.{layer_idx}.self_attn.o_proj"
    for name, module in model.named_modules():
        if attn_name in name:
            print(f"为注意力层注册钩子: {name}")
            hooks.append(module.register_forward_hook(get_attention_features))
            break
    
    return hooks

# 获取top-k softmax概率
def get_top_k_softmax(logits, k=50):
    top_values, top_indices = torch.topk(logits, k)
    sparse_softmax = torch.zeros_like(logits)
    sparse_softmax.scatter_(0, top_indices, top_values)
    return F.softmax(sparse_softmax, dim=-1).cpu().numpy()

def main():
    # 设置随机种子
    set_seed(42)
    
    # 加载模型和分词器
    model_path = "Qwen/Qwen2.5-7B-Coder"
    print(f"正在加载模型: {model_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    except Exception as e:
        print(f"模型加载失败: {e}")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=False)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"分词器加载失败: {e}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    
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
    hooks = register_hooks(model, num_layers - 1)
    
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
        'last_token_softmax': [],  # 最后一个token的softmax概率 (top-50)
        'attributes_last_token': [],  # IG特征归因
        'last_token_attention': [],  # 注意力得分
        'last_token_fully_connected': []  # 全连接层激活
    }
    
    # 优化：批处理样本提高效率
    batch_size = 1  # 当前只能按1个处理，未来可优化为更大批量
    last_attributes = None  # 存储上一个有效的IG结果
    
    for idx, sample in enumerate(tqdm(code_samples, desc="提取特征")):
        code = sample['code']
        label = sample['label']
        
        # 编码输入
        inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 获取最后一个token的位置
        last_token_pos = inputs['input_ids'].shape[1] - 1
        
        # 前向传播 - 只执行一次
        with torch.no_grad():
            outputs = model(**inputs)
            
            # 1. 获取top-k softmax概率
            last_token_logits = outputs.logits[0, -1]
            last_token_softmax = get_top_k_softmax(last_token_logits, k=50)
            
            # 2. 获取全连接层激活
            fc_features = features_dict['fc_output'][0, -1].cpu().numpy() if features_dict['fc_output'] is not None else np.array([])
            
            # 3. 获取注意力得分
            attn_features = features_dict['attn_output'][0, -1].cpu().numpy() if features_dict['attn_output'] is not None else np.array([])
        
        # 4. 计算Integrated Gradients属性 - 间隔采样
        if idx % ig_sample_rate == 0:
            try:
                attributes = get_ig(code, last_token_pos, forward_func, tokenizer, embedder, model)
                last_attributes = attributes
            except Exception as e:
                print(f"计算IG时出错: {e}")
                attributes = last_attributes if last_attributes is not None else np.zeros(last_token_pos + 1)
        else:
            # 使用最近的IG结果，避免重复计算
            attributes = last_attributes if last_attributes is not None else np.zeros(last_token_pos + 1)
        
        # 存储结果
        results['code'].append(code)
        results['label'].append(label)
        results['last_token_pos'].append(last_token_pos)
        results['last_token_softmax'].append(last_token_softmax)
        results['attributes_last_token'].append(attributes)
        results['last_token_attention'].append(attn_features)
        results['last_token_fully_connected'].append(fc_features)
        
        # 清理特征字典，准备下一次迭代
        features_dict['fc_output'] = None
        features_dict['attn_output'] = None
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'results/qwen_code_results_{timestamp}.pickle'
    os.makedirs('results', exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"结果已保存到: {save_path}")
    
    # 清理内存
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 