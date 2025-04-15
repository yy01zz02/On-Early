import functools
from datetime import datetime
from typing import Any, Dict
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict, Counter
from functools import partial
import re
from captum.attr import IntegratedGradients

# IO
data_dir = Path(".") # Where our data files are stored
model_dir = Path(r"D:\DataSet\Qwen2.5-Coder-0.5B-Instruct") # 本地模型路径
results_dir = Path("./results/") # Directory for storing results
data_path = Path(r"D:\Experiment\PycharmProject\Probes-Code-infilling\collu-bench_data\data_only_buggy_one_line\humaneval.csv") # 数据集路径

# Hardware
gpu = "0"
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

# Integrated Grads
ig_steps = 64
internal_batch_size = 4

# Model
model_name = "Qwen2.5-Coder-0.5B-Instruct" # 修改为本地模型文件夹名称
layer_number = -1
use_half_precision = False  # 使用半精度以减少显存使用

# For storing results
fully_connected_hidden_layers = defaultdict(list)
attention_hidden_layers = defaultdict(list)
attention_forward_handles = {}
fully_connected_forward_handles = {}

# Qwen-Code model structure - 更新到Qwen2.5架构
model_num_layers = {"Qwen2.5-Coder-0.5B-Instruct": 24}
coll_str = "[0-9]+" if layer_number==-1 else str(layer_number)

# Model layer patterns for Qwen2.5模型架构
layer_patterns = {
    "Qwen2.5-Coder-0.5B-Instruct": (
        f".*model.layers.{coll_str}.mlp.down_proj",  # FFN output projection layer
        f".*model.layers.{coll_str}.self_attn.o_proj"  # Attention output projection layer
    ),
}

def save_fully_connected_hidden(layer_name, mod, inp, out):
    # 立即将输出转移到CPU并转换为numpy数组，释放GPU显存
    fully_connected_hidden_layers[layer_name].append(out.squeeze().detach().to(torch.float32).cpu().numpy())
    # 清理不需要的中间变量
    del out


def save_attention_hidden(layer_name, mod, inp, out):
    # 立即将输出转移到CPU并转换为numpy数组，释放GPU显存
    attention_hidden_layers[layer_name].append(out.squeeze().detach().to(torch.float32).cpu().numpy())
    # 清理不需要的中间变量
    del out


def get_newline_token(tokenizer):
    """Get the token ID for the newline character."""
    return tokenizer.encode('\n', add_special_tokens=False)[-1]


def load_code_data(data_path):
    """
    加载代码文件和标签
    data_path: CSV文件路径，包含code和label两列
    返回: 包含(代码内容, 标签)元组的列表
    """
    pd_frame = pd.read_csv(data_path)
    dataset = [(pd_frame.iloc[i]['code'], pd_frame.iloc[i]['label']) for i in range(len(pd_frame))]
    return dataset


def analyze_code(code_content, model, tokenizer):
    """分析代码内容，提取最后一个token的特征"""
    input_ids = tokenizer.encode(code_content, return_tensors='pt').to(device)
    
    # 获取最后一个token的位置
    last_token_pos = input_ids.shape[1] - 1
    
    # 前向传播获取特征
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
    
    # 清理不需要的中间变量
    del outputs
    torch.cuda.empty_cache()  # 清理GPU缓存
    
    return input_ids, logits, last_token_pos


def get_start_end_layer(model):
    """获取模型层的起始和结束索引，适配Qwen2.5结构"""
    # 获取模型层数
    layer_count = len(model.model.layers)
    layer_st = 0 if layer_number == -1 else layer_number
    layer_en = layer_count if layer_number == -1 else layer_number + 1
    return layer_st, layer_en


def collect_fully_connected(token_pos, layer_start, layer_end):
    """收集全连接层的激活值"""
    layer_name = layer_patterns[model_name][0][2:].split(coll_str)
    newline_activation = np.stack([fully_connected_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][token_pos] \
                                for i in range(layer_start, layer_end)])
    return newline_activation


def collect_attention(token_pos, layer_start, layer_end):
    """收集注意力层的激活值"""
    layer_name = layer_patterns[model_name][1][2:].split(coll_str)
    newline_activation = np.stack([attention_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][token_pos] \
                                for i in range(layer_start, layer_end)])
    return newline_activation


def normalize_attributes(attributes: torch.Tensor) -> torch.Tensor:
    """归一化Integrated Gradients属性"""
    attributes = attributes.squeeze(0)
    norm = torch.norm(attributes, dim=1)
    attributes = norm / torch.sum(norm)  # Normalize the values so they add up to 1
    return attributes


def model_forward(input_: torch.Tensor, model, extra_forward_args: Dict[str, Any]) -> torch.Tensor:
    """模型前向传播函数，用于Integrated Gradients"""
    output = model(inputs_embeds=input_, **extra_forward_args)
    return torch.nn.functional.softmax(output.logits[:, -1, :], dim=-1)


def get_embedder(model):
    """获取模型的词嵌入层，适配Qwen2.5结构"""
    return model.model.embed_tokens


def get_ig(code_content, token_pos, forward_func, tokenizer, embedder, model):
    """
    计算Integrated Gradients属性
    token_pos: 目标token位置（最后一个token）
    """
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
    
    # 清理不需要的中间变量
    del encoder_input_embeds
    del ig
    torch.cuda.empty_cache()  # 清理GPU缓存
    
    return attributes


def compute_and_save_results():
    """
    主函数：加载模型和数据，提取特征并保存结果
    """
    # 加载数据集
    dataset = load_code_data(data_path)

    # 加载模型和分词器
    print(f"正在加载模型：{model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    # 使用半精度加载模型以减少显存使用
    if use_half_precision:
        print("使用半精度加载模型以减少显存使用")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.float16  # 使用半精度
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map=device,
            trust_remote_code=True
        )
    
    # 设置模型为评估模式
    model.eval()
    
    # 打印模型结构以便调试
    print("模型类型:", type(model))
    print("模型结构预览:", model.__class__.__name__)
    
    # 获取词嵌入层
    embedder = get_embedder(model)
    
    # 初始化前向传播函数
    forward_func = partial(model_forward, model=model, extra_forward_args={})
    
    # 注册钩子函数，收集隐层激活值
    layer_start, layer_end = get_start_end_layer(model)
    print(f"处理层范围: {layer_start} - {layer_end}")
    
    # 注册全连接层钩子
    fc_pattern = layer_patterns[model_name][0][2:].split(coll_str)
    for i in range(layer_start, layer_end):
        layer_name = f'{fc_pattern[0]}{i}{fc_pattern[1]}'
        for name, module in model.named_modules():
            if re.search(f'{fc_pattern[0]}{i}{fc_pattern[1]}$', name):
                fully_connected_forward_handles[layer_name] = module.register_forward_hook(
                    partial(save_fully_connected_hidden, layer_name))
                print(f"注册全连接层钩子: {name}")
                break
    
    # 注册注意力层钩子
    attn_pattern = layer_patterns[model_name][1][2:].split(coll_str)
    for i in range(layer_start, layer_end):
        layer_name = f'{attn_pattern[0]}{i}{attn_pattern[1]}'
        for name, module in model.named_modules():
            if re.search(f'{attn_pattern[0]}{i}{attn_pattern[1]}$', name):
                attention_forward_handles[layer_name] = module.register_forward_hook(
                    partial(save_attention_hidden, layer_name))
                print(f"注册注意力层钩子: {name}")
                break
    
    # 存储结果
    results = {
        'code': [],
        'last_token_fully_connected': [], 
        'last_token_attention': [],
        'attributes_last_token': [],
        'logits': [],
        'last_token_pos': [],
        'label': []
    }
    
    # 处理每个代码内容
    for code_content, label in tqdm(dataset):
        # 清空之前的激活值
        fully_connected_hidden_layers.clear()
        attention_hidden_layers.clear()
        
        try:
            # 分析代码，提取特征
            input_ids, logits, last_token_pos = analyze_code(code_content, model, tokenizer)
            
            # 收集全连接层和注意力层的激活值
            last_token_fully_connected = collect_fully_connected(last_token_pos, layer_start, layer_end)
            last_token_attention = collect_attention(last_token_pos, layer_start, layer_end)
            
            # 计算Integrated Gradients属性
            attributes = get_ig(code_content, last_token_pos, forward_func, tokenizer, embedder, model)
            
            # 存储结果
            results['code'].append(code_content)
            results['last_token_fully_connected'].append(last_token_fully_connected)
            results['last_token_attention'].append(last_token_attention)
            results['attributes_last_token'].append(attributes)
            results['logits'].append(logits.squeeze().detach().cpu().numpy())
            results['last_token_pos'].append(last_token_pos)
            results['label'].append(label)
            
            # 清理不需要的中间变量
            del input_ids, logits, last_token_fully_connected, last_token_attention, attributes
            torch.cuda.empty_cache()  # 清理GPU缓存
            
        except Exception as e:
            print(f"处理样本时出错: {e}")
            continue
    
    # 移除钩子
    for handle in fully_connected_forward_handles.values():
        handle.remove()
    for handle in attention_forward_handles.values():
        handle.remove()
    
    # 保存结果
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f'qwen_code_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pickle'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"结果已保存到 {results_file}")
    print(f"共处理 {len(results['code'])} 个样本")
    return results_file


if __name__ == "__main__":
    compute_and_save_results()