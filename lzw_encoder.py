# lzw_encoder.py

import sys
from sys import argv
from struct import pack
import time
import math
import numpy as np
import os

def encode(input_file, use_rl=True):
    # 初始化变量
    data = open(input_file, 'rb').read()  # 以字节形式读取数据
    dictionary_size = 256
    maximum_table_size = 4096  # 初始最大字典大小
    dictionary = {bytes([i]): i for i in range(dictionary_size)}
    string = b""
    compressed_data = []
    position = 0  # 数据指针
    total_input_size = len(data) * 8  # 假设每个字节8位
    total_output_bits = 0
    start_time = time.time()
    
    # 如果使用 RL 代理，初始化代理
    if use_rl:
        from lzw_agent import RLAgent
        state_size = 3
        action_size = 4
        rl_agent = RLAgent(state_size, action_size)
        model_path = 'models/rl_agent_model.pth'
        if os.path.exists(model_path):
            rl_agent.load(model_path)
        else:
            print("No pre-trained model found. Starting with a new model.")
        actions_log = []

    while position < len(data):
        symbol = bytes([data[position]])
        string_plus_symbol = string + symbol

        if string_plus_symbol in dictionary:
            string = string_plus_symbol
        else:
            if string in dictionary:
                code = dictionary[string]
            else:
                # 如果 string 不在字典中，设置为第一个字节
                code = dictionary[string[0:1]]
            compressed_data.append(code)
            bits_required = int(math.ceil(math.log2(dictionary_size)))
            total_output_bits += bits_required

            if dictionary_size < maximum_table_size:
                dictionary[string_plus_symbol] = dictionary_size
                dictionary_size += 1
            else:
                pass  # 字典已满

            string = symbol

        position += 1

        # 如果使用 RL 代理，在间隔处获取动作
        if use_rl and (position % 100 == 0 or position == len(data)):
            # 计算状态
            if total_output_bits == 0:
                compression_ratio = 0
            else:
                compression_ratio = total_input_size / total_output_bits
            current_time = time.time()
            processing_time = (current_time - start_time) / position  # 每个字节的平均处理时间
            state = np.array([
                compression_ratio / 10,
                processing_time * 1000,
                dictionary_size / maximum_table_size
            ])

            # 获取动作
            action = rl_agent.get_action(state)
            actions_log.append(action)

            # 根据动作调整参数
            if action == 0:
                maximum_table_size += 512
            elif action == 1:
                maximum_table_size = max(256, maximum_table_size - 512)
                if dictionary_size > maximum_table_size:
                    dictionary_size = maximum_table_size
            elif action == 2:
                dictionary_size = 256
                dictionary = {bytes([i]): i for i in range(dictionary_size)}
                string = b""  # 重置字符串
            elif action == 3:
                pass

    # 处理剩余的字符串
    if string in dictionary:
        code = dictionary[string]
        compressed_data.append(code)
        bits_required = int(math.ceil(math.log2(dictionary_size)))
        total_output_bits += bits_required

    # 将压缩数据写入文件
    output_file = input_file.split(".")[0] + "_compressed.lzw"
    with open(output_file, 'wb') as f:
        for code in compressed_data:
            f.write(pack('>H', int(code)))  # 假设代码在2字节内

    # 保存动作日志（如果使用 RL）
    if use_rl:
        actions_file = input_file.split(".")[0] + "_actions.npy"
        np.save(actions_file, actions_log)

    # 计算压缩比和处理时间
    final_compression_ratio = total_input_size / total_output_bits if total_output_bits != 0 else 0
    total_processing_time = time.time() - start_time

    # 输出结果
    print(f"Compression completed. Use RL: {use_rl}")
    print(f"Final Compression Ratio: {final_compression_ratio:.2f}")
    print(f"Total Processing Time: {total_processing_time:.2f} seconds")

    return final_compression_ratio, total_processing_time

if __name__ == "__main__":
    if len(argv) < 2:
        print("Usage: python lzw_encoder.py <input_file> [--no-rl]")
        sys.exit(1)
    input_file = argv[1]
    use_rl = True
    if len(argv) > 2 and argv[2] == '--no-rl':
        use_rl = False
    encode(input_file, use_rl)