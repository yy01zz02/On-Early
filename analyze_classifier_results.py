import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import os
import matplotlib

# 配置中文字体支持
def setup_chinese_fonts():
    """设置中文字体支持"""
    # 尝试设置不同的中文字体，以适应不同系统环境
    try:
        # 检查系统是否有中文字体
        font_list = matplotlib.font_manager.findSystemFonts()
        chinese_fonts = []
        
        # 常见中文字体列表
        chinese_font_names = [
            'SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi',  # Windows
            'PingFang SC', 'STHeiti', 'Heiti SC', 'Hiragino Sans GB',  # macOS
            'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC',  # Linux
            'Noto Sans SC', 'Source Han Sans CN', 'Source Han Sans SC'  # 开源字体
        ]
        
        # 查找系统中可用的中文字体
        for font_path in font_list:
            try:
                font = matplotlib.font_manager.FontProperties(fname=font_path)
                font_family = font.get_name()
                if any(chinese_name in font_family for chinese_name in chinese_font_names):
                    chinese_fonts.append(font_path)
            except:
                continue
        
        # 如果找到了中文字体，设置默认字体
        if chinese_fonts:
            plt.rcParams['font.family'] = matplotlib.font_manager.FontProperties(fname=chinese_fonts[0]).get_name()
            print(f"已设置中文字体: {matplotlib.font_manager.FontProperties(fname=chinese_fonts[0]).get_name()}")
            return True
        
        # 如果没有找到中文字体，尝试使用sans-serif字体族
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Heiti SC', 
                                            'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print("尝试使用sans-serif字体族显示中文")
        return True
    
    except Exception as e:
        print(f"设置中文字体时出错: {e}")
        print("将使用默认字体，中文可能无法正确显示")
        return False

def load_pickle_results(pickle_path):
    """加载分类器结果pickle文件"""
    try:
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"无法加载文件 {pickle_path}: {e}")
        return None

def extract_model_name(file_path):
    """从结果文件路径提取模型名称"""
    file_name = os.path.basename(file_path)
    if "qwen" in file_name.lower():
        return "Qwen2.5-Coder"
    elif "codellama" in file_name.lower():
        return "CodeLlama"
    else:
        return "未知模型"

def extract_features_and_metrics(results):
    """从结果中提取不同特征的性能指标"""
    all_data = []
    
    for file_path, metrics in results.items():
        model_name = extract_model_name(file_path)
        
        # 按特征类型分组
        feature_groups = {
            "IG特征": {}, 
            "Softmax概率": {},
            "全连接层": {},
            "注意力层": {},
            "其他特征": {}
        }
        
        # 提取所有指标
        for metric_name, value in metrics.items():
            if "attributes_last_token" in metric_name:
                feature_type = "IG特征"
                layer = "N/A"
                metric_type = "ROC AUC" if "roc" in metric_name else "准确率"
            elif "softmax" in metric_name:
                feature_type = "Softmax概率"
                layer = "N/A"
                metric_type = "ROC AUC" if "roc" in metric_name else "准确率"
            elif "fully_connected" in metric_name:
                feature_type = "全连接层"
                if "_roc_" in metric_name:
                    layer = metric_name.split("_roc_")[1]
                    metric_type = "ROC AUC"
                elif "_acc_" in metric_name:
                    layer = metric_name.split("_acc_")[1]
                    metric_type = "准确率"
                else:
                    layer = "N/A"
                    metric_type = "ROC AUC" if "roc" in metric_name else "准确率"
            elif "attention" in metric_name:
                feature_type = "注意力层"
                if "_roc_" in metric_name:
                    layer = metric_name.split("_roc_")[1]
                    metric_type = "ROC AUC"
                elif "_acc_" in metric_name:
                    layer = metric_name.split("_acc_")[1]
                    metric_type = "准确率"
                else:
                    layer = "N/A"
                    metric_type = "ROC AUC" if "roc" in metric_name else "准确率"
            else:
                feature_type = "其他特征"
                layer = "N/A"
                metric_type = "ROC AUC" if "roc" in metric_name else "准确率"
            
            # 记录每一项数据
            all_data.append({
                "模型": model_name,
                "文件": file_path,
                "特征类型": feature_type,
                "层": layer,
                "指标类型": metric_type,
                "值": value
            })
    
    return pd.DataFrame(all_data)

def plot_feature_comparison(df, output_dir):
    """绘制不同特征类型的性能比较图"""
    plt.figure(figsize=(12, 8))
    
    # 按特征类型和指标类型分组，计算平均值
    grouped = df.groupby(["特征类型", "指标类型"])["值"].mean().reset_index()
    
    # 透视表以便绘图
    pivot_data = grouped.pivot(index="特征类型", columns="指标类型", values="值")
    
    # 绘制条形图
    ax = pivot_data.plot(kind="bar", figsize=(10, 6))
    plt.title("不同特征类型的分类性能", fontsize=16)
    plt.ylabel("分数", fontsize=14)
    plt.xlabel("特征类型", fontsize=14)
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="指标类型")
    
    # 在条形上显示数值
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_comparison.png", dpi=300)
    plt.close()

def plot_layer_performance(df, feature_type, output_dir):
    """绘制特定特征类型各层的性能图"""
    # 过滤出特定特征类型的数据，且只选择有层信息的数据
    layer_data = df[(df["特征类型"] == feature_type) & (df["层"] != "N/A")].copy()
    
    if len(layer_data) == 0:
        print(f"没有找到{feature_type}的层级数据")
        return
    
    # 确保层是数字并排序
    layer_data.loc[:, "层"] = pd.to_numeric(layer_data["层"], errors="coerce")
    layer_data = layer_data.dropna(subset=["层"]).sort_values("层")
    
    plt.figure(figsize=(12, 8))
    
    # 绘制每一层的ROC和准确率
    for metric_type in ["ROC AUC", "准确率"]:
        metric_data = layer_data[layer_data["指标类型"] == metric_type]
        if len(metric_data) > 0:
            plt.plot(metric_data["层"], metric_data["值"], 
                     marker='o', linestyle='-', 
                     label=f"{metric_type}")
    
    plt.title(f"{feature_type}各层的分类性能", fontsize=16)
    plt.xlabel("层索引", fontsize=14)
    plt.ylabel("分数", fontsize=14)
    plt.ylim(0.5, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.tight_layout()
    
    # 确保输出文件名没有空格和特殊字符
    feature_type_filename = feature_type.replace(" ", "_").replace("/", "_")
    plt.savefig(f"{output_dir}/{feature_type_filename}_layer_performance.png", dpi=300)
    plt.close()

def plot_model_comparison(df, output_dir):
    """绘制不同模型的性能比较图"""
    # 确保有多个模型可比较
    if df["模型"].nunique() <= 1:
        print("只有一个模型，跳过模型比较图")
        return
    
    plt.figure(figsize=(12, 8))
    
    # 计算每个模型每种特征类型的平均性能
    model_data = df.groupby(["模型", "特征类型", "指标类型"])["值"].mean().reset_index()
    
    # 只使用ROC AUC指标进行模型比较
    roc_data = model_data[model_data["指标类型"] == "ROC AUC"]
    
    # 透视表以便绘图
    pivot_data = roc_data.pivot(index="模型", columns="特征类型", values="值")
    
    # 绘制条形图
    ax = pivot_data.plot(kind="bar", figsize=(10, 6))
    plt.title("不同模型的分类性能比较 (ROC AUC)", fontsize=16)
    plt.ylabel("ROC AUC", fontsize=14)
    plt.xlabel("模型", fontsize=14)
    plt.ylim(0.5, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="特征类型")
    
    # 在条形上显示数值
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=300)
    plt.close()

def generate_summary_table(df, output_dir):
    """生成性能摘要表格"""
    # 计算每个特征类型和指标类型的平均性能和最大性能
    summary = df.groupby(["特征类型", "指标类型"]).agg({
        "值": ["mean", "max"]
    }).reset_index()
    
    # 格式化表格
    summary.columns = ["特征类型", "指标类型", "平均值", "最大值"]
    summary["平均值"] = summary["平均值"].round(4)
    summary["最大值"] = summary["最大值"].round(4)
    
    # 保存为CSV
    summary.to_csv(f"{output_dir}/summary_table.csv", index=False, encoding="utf-8-sig")
    
    # 打印表格
    print("\n性能摘要:")
    print(summary.to_string(index=False))
    
    return summary

def run_analysis(result_files, output_dir="./analysis_results"):
    """运行完整分析流程"""
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # 加载所有结果文件
    for file_path in result_files:
        results = load_pickle_results(file_path)
        if results:
            all_results.update(results)
    
    if not all_results:
        print("没有找到有效的结果数据")
        return
    
    print(f"已加载 {len(all_results)} 个结果文件")
    
    # 提取特征和指标
    df = extract_features_and_metrics(all_results)
    
    # 保存处理后的数据
    df.to_csv(f"{output_dir}/processed_results.csv", index=False, encoding="utf-8-sig")
    
    # 生成摘要表格
    summary = generate_summary_table(df, output_dir)
    
    # 绘制比较图
    plot_feature_comparison(df, output_dir)
    plot_model_comparison(df, output_dir)
    
    # 为不同特征类型绘制层性能图
    for feature_type in ["全连接层", "注意力层"]:
        plot_layer_performance(df, feature_type, output_dir)
    
    print(f"分析完成，结果保存到 {output_dir} 目录")
    
    # 找出最佳表现特征
    best_feature = df.loc[df["值"].idxmax()]
    print(f"\n最佳表现特征:")
    print(f"  特征类型: {best_feature['特征类型']}")
    print(f"  层: {best_feature['层']}")
    print(f"  指标类型: {best_feature['指标类型']}")
    print(f"  值: {best_feature['值']:.4f}")
    print(f"  模型: {best_feature['模型']}")

def main():
    parser = argparse.ArgumentParser(description="分析分类器结果")
    parser.add_argument("--files", nargs="+", help="要分析的结果pickle文件", default=None)
    parser.add_argument("--output", help="分析结果输出目录", default="./analysis_results")
    args = parser.parse_args()
    
    # 设置中文字体
    setup_chinese_fonts()
    
    if args.files:
        result_files = [Path(f) for f in args.files]
    else:
        # 自动查找结果文件
        result_files = list(Path("./results/").glob("*classifier_results*.pickle"))
    
    if not result_files:
        print("未找到结果文件。请使用--files参数指定文件路径。")
        return
    
    print(f"将分析以下结果文件: {[str(f) for f in result_files]}")
    run_analysis(result_files, args.output)

if __name__ == "__main__":
    main() 