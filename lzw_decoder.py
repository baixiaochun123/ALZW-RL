# lzw_decoder.py

import sys
from sys import argv
from struct import unpack
import time
import math
import numpy as np
from lzw_agent import RLAgent  # 导入 RLAgent
import os

# 初始化 RL 代理
state_size = 2  # 状态维度：处理时间、字典大小比例
action_size = 4  # 动作数量
rl_agent = RLAgent(state_size, action_size)

# 加载代理的模型
model_path = 'models/rl_agent_model.pth'
if os.path.exists(model_path):
    rl_agent.load(model_path)
else:
    print("No pre-trained model found. Starting with a new model.")

# 初始化变量
input_file = argv[1]
compressed_data = []
with open(input_file, 'rb') as f:
    while True:
        rec = f.read(2)
        if len(rec) != 2:
            break
        (data,) = unpack('>H', rec)
        compressed_data.append(data)

dictionary_size = 256
maximum_table_size = rl_agent.initial_dictionary_size()
dictionary = {i: bytes([i]) for i in range(dictionary_size)}
string = b""
decompressed_data = bytearray()
position = 0  # 数据指针
start_time = time.time()
action = None  # 初始动作

# 加载编码器的动作日志，以同步参数
actions_file = input_file.split(".")[0] + "_actions.npy"
if os.path.exists(actions_file):
    actions_log = np.load(actions_file)
else:
    print("No actions log found. Decoding may not be synchronized.")
    actions_log = []

action_index = 0  # 动作日志的索引

while position < len(compressed_data):
    code = compressed_data[position]
    if code in dictionary:
        entry = dictionary[code]
    elif code == dictionary_size:
        # 特殊情况，定义为 string + first byte of string
        entry = string + bytes([string[0]])
        dictionary[code] = entry
    else:
        raise ValueError(f"Bad compressed code: {code}")
    
    decompressed_data += entry
    
    if string:
        if dictionary_size < maximum_table_size:
            dictionary[dictionary_size] = string + bytes([entry[0]])
            dictionary_size += 1
        else:
            pass  # 字典已满，可以实现字典替换策略
    
    string = entry
    position += 1

    # 在与编码器相同的间隔或条件下，从动作日志中获取动作
    if position % 100 == 0 or position == len(compressed_data):
        if action_index < len(actions_log):
            action = actions_log[action_index]
            action_index += 1
        else:
            # 如果没有更多的动作，默认不做调整
            action = 3

        # 根据动作调整参数
        if action == 0:  # 增加字典大小
            maximum_table_size += 512
        elif action == 1:  # 减少字典大小
            maximum_table_size = max(256, maximum_table_size - 512)
            if dictionary_size > maximum_table_size:
                # 需要缩减字典
                # 可以在此处实现字典修剪逻辑
                dictionary_size = maximum_table_size
        elif action == 2:  # 重置字典
            dictionary_size = 256
            dictionary = {i: bytes([i]) for i in range(dictionary_size)}
        elif action == 3:  # 不做任何调整
            pass

# 将解压缩的数据写入文件（以字节形式）
output_file = input_file.split(".")[0] + "_decompressed.bin"
with open(output_file, 'wb') as f:
    f.write(decompressed_data)

# 输出解码完成信息
total_processing_time = time.time() - start_time
print(f"Decoding completed.")
print(f"Total Processing Time: {total_processing_time:.2f} seconds")