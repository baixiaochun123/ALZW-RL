import requests
import os
import pandas as pd

# 1.下载刘易斯·卡罗尔的《爱丽丝梦游仙境》
url = 'https://www.gutenberg.org/files/11/11-0.txt'
response = requests.get(url)

# 将内容保存到 'data/input.txt'
with open('1text.txt', 'w', encoding='utf-8') as f:
    f.write(response.text)

#2.示例：生成正弦波时间序列
import numpy as np
import os

# 如果不存在，则创建 'data' 目录
if not os.path.exists('data'):
    os.makedirs('data')

# 生成合成时间序列数据（正弦波）
time_steps = np.linspace(0, 100, 10000)
time_series = np.sin(time_steps)

# 将数值数据转换为字符串格式
data_str = ','.join(map(str, time_series))

# 将数据保存到 'data/input.txt'
with open('2num.txt', 'w') as f:
    f.write(data_str)

sample_text = """
在机器学习领域，数据是解锁算法潜力的关键。
强化学习算法通过与环境交互来实现目标。
像 LZW 这样的压缩算法在不丢失信息的情况下减小数据的大小。
"""

# 将样本文本保存到 'data/input.txt'
with open('4test.txt', 'w', encoding='utf-8') as f:
    f.write(sample_text)


texts = []
urls = [
    'https://www.gutenberg.org/files/1342/1342-0.txt',   # 傲慢与偏见
    'https://www.gutenberg.org/files/84/84-0.txt',       # 科学怪人
    'https://www.gutenberg.org/files/98/98-0.txt'        # 双城记
]

for url in urls:
    response = requests.get(url)
    texts.append(response.text)

combined_text = '\n'.join(texts)

# 将合并的文本保存到 'data/input.txt'
with open('5train.txt', 'w', encoding='utf-8') as f:
    f.write(combined_text)
