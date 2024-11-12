import csv

# 打开CSV文件
with open('compression_results.csv', mode='r', encoding='utf-8') as file:
    # 创建一个csv阅读器
    reader = csv.reader(file)
    
    # 遍历CSV文件中的每一行
    for row in reader:
        print(row)  # row是一个列表，包含了行中的每个字段