import pandas as pd

# 读取两个CSV文件
csv_file_1 = 'humaneval.csv'
csv_file_2 = 'mbpp.csv'

# 使用pandas读取CSV文件
df1 = pd.read_csv(csv_file_1)
df2 = pd.read_csv(csv_file_2)

# 合并两个DataFrame
merged_df = pd.concat([df1, df2])

# 去除重复项，保留第一条出现的记录
merged_df.drop_duplicates(inplace=True)

# 保存为新的CSV文件
output_file = 'merged_data.csv'
merged_df.to_csv(output_file, index=False)

print(f"合并完成，去重后已保存为 {output_file}")