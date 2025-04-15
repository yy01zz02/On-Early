import pickle
import numpy as np
from pathlib import Path
import argparse
import sys

def analyze_pickle_file(pickle_path, verbose=False):
    """
    分析pickle文件中的数据形状并打印信息
    
    Args:
        pickle_path: pickle文件路径
        verbose: 是否打印详细信息
    """
    print(f"分析文件: {pickle_path}")
    
    try:
        # 加载pickle文件
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, dict):
            print(f"警告: pickle文件包含的不是字典类型，而是 {type(data)}")
            return
        
        # 打印基本信息
        print("\n基本信息:")
        print(f"字典键: {list(data.keys())}")
        print(f"数据样本数: {len(data.get('code', []))}")
        
        # 分析每个键的数据形状
        print("\n数据形状分析:")
        for key in data.keys():
            if key == 'code':
                print(f"  {key}: {len(data[key])} 个样本")
                if verbose and len(data[key]) > 0:
                    print(f"    示例: \n{data[key][0][:200]}...")
            elif key == 'label':
                print(f"  {key}: {len(data[key])} 个样本")
                if len(data[key]) > 0:
                    label_counts = {}
                    for label in data[key]:
                        label_counts[label] = label_counts.get(label, 0) + 1
                    print(f"    标签分布: {label_counts}")
            elif key == 'last_token_pos':
                print(f"  {key}: {len(data[key])} 个样本")
                if len(data[key]) > 0:
                    print(f"    范围: {min(data[key])} ~ {max(data[key])}")
            elif key == 'logits':
                print(f"  {key}: {len(data[key])} 个样本")
                if len(data[key]) > 0:
                    print(f"    形状: {data[key][0].shape}")
            elif key in ['last_token_fully_connected', 'last_token_attention']:
                print(f"  {key}: {len(data[key])} 个样本")
                if len(data[key]) > 0:
                    shape_str = f"{data[key][0].shape}"
                    print(f"    形状: {shape_str}")
                    print(f"    第一维是层数: {data[key][0].shape[0]} 层")
                    print(f"    第二维是特征维度: {data[key][0].shape[1]}")
                    
                    # 计算每层的平均值和标准差
                    if verbose:
                        for layer in range(data[key][0].shape[0]):
                            layer_data = np.stack([sample[layer] for sample in data[key]])
                            print(f"    层 {layer}: 均值={np.mean(layer_data):.4f}, 标准差={np.std(layer_data):.4f}")
            elif key == 'attributes_last_token':
                print(f"  {key}: {len(data[key])} 个样本")
                if len(data[key]) > 0:
                    print(f"    形状: {np.array(data[key][0]).shape}")
            else:
                print(f"  {key}: {len(data[key])} 个样本")
                if len(data[key]) > 0 and hasattr(data[key][0], 'shape'):
                    print(f"    形状: {data[key][0].shape}")
                elif len(data[key]) > 0:
                    print(f"    类型: {type(data[key][0])}")
        
        # 检查样本数是否一致
        sample_counts = {key: len(value) for key, value in data.items() if isinstance(value, list)}
        if len(set(sample_counts.values())) != 1:
            print("\n警告: 不同特征的样本数不一致")
            for key, count in sample_counts.items():
                print(f"  {key}: {count}")
        
        print("\n分析完成")
    
    except Exception as e:
        print(f"分析文件时出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='分析特征提取pickle文件的形状')
    parser.add_argument('--file', type=str, help='要分析的pickle文件路径')
    parser.add_argument('--all', action='store_true', help='分析results目录下的所有pickle文件')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细信息')
    args = parser.parse_args()
    
    if args.file:
        analyze_pickle_file(args.file, args.verbose)
    elif args.all:
        pickle_files = list(Path("./results/").rglob("*.pickle"))
        if not pickle_files:
            print("未找到任何pickle文件")
            return
        
        print(f"找到 {len(pickle_files)} 个pickle文件")
        for pickle_file in pickle_files:
            print("\n" + "="*50)
            analyze_pickle_file(pickle_file, args.verbose)
    else:
        # 如果没有指定文件，尝试查找最新的pickle文件
        pickle_files = list(Path("./results/").rglob("qwen_code_results_*.pickle"))
        if not pickle_files:
            print("未找到任何qwen_code_results_*.pickle文件，请指定要分析的文件路径")
            parser.print_help()
            return
        
        # 按修改时间排序，取最新的文件
        latest_file = max(pickle_files, key=lambda p: p.stat().st_mtime)
        print(f"分析最新的pickle文件: {latest_file}")
        analyze_pickle_file(latest_file, args.verbose)

if __name__ == "__main__":
    main() 