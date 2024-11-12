# lzw_evaluate.py

import os
import time
import zlib
import gzip
import bz2
import lzma
import brotli
import lz4.frame
import snappy
import matplotlib.pyplot as plt
import csv
import torch
import sys

# 添加当前目录到 PYTHONPATH，以确保可以正确导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from lzw_agent import RLAgent  # 确保 lzw_agent.py 在当前目录或 PYTHONPATH 中
except ModuleNotFoundError as e:
    print(f"错误: 无法导入 RLAgent 类: {e}")
    print("请确保 'lzw_agent.py' 文件位于当前目录或 PYTHONPATH 中。")
    sys.exit(1)

def read_file(file_path):
    """读取文件内容并返回字节数据。"""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        print(f"成功读取文件: {file_path} (大小: {len(data)} 字节)")
        return data
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return None

def write_file(file_path, data):
    """将字节数据写入文件。"""
    try:
        with open(file_path, 'wb') as f:
            f.write(data)
        print(f"成功写入文件: {file_path} (大小: {len(data)} 字节)")
    except Exception as e:
        print(f"写入文件 {file_path} 时出错: {e}")

def calculate_compression_ratio(original_data, compressed_data):
    """计算压缩比：原始大小 / 压缩后大小。"""
    original_size = len(original_data)
    compressed_size = len(compressed_data)
    if compressed_size == 0:
        print("警告: 压缩后的数据大小为0，无法计算压缩比。")
        return 0
    ratio = original_size / compressed_size
    return ratio

def lzw_decompress(compressed_data):
    """使用 LZW 算法解压缩数据。"""
    max_table_size = 4096  # 最大表大小
    # 将字节数据转换为整数列表，每个代码由2个字节组成
    if len(compressed_data) % 2 != 0:
        print("警告: 压缩数据长度不是2的倍数，可能存在损坏。")
        return b''

    compressed_codes = [int.from_bytes(compressed_data[i:i+2], byteorder='big') for i in range(0, len(compressed_data), 2)]
    table = {i: bytes([i]) for i in range(256)}
    string = table[compressed_codes[0]]
    decompressed_data = bytearray(string)
    code = 256
    for k in compressed_codes[1:]:
        if k in table:
            entry = table[k]
        elif k == code:
            entry = string + bytes([string[0]])
        else:
            print(f"错误: 压缩码 {k} 不在字典中。")
            return b''
        decompressed_data += entry
        if len(table) < max_table_size:
            table[code] = string + bytes([entry[0]])
            code += 1
        string = entry
    return bytes(decompressed_data)

def lzw_compress(data):
    """使用 LZW 算法压缩数据。"""
    max_table_size = 4096  # 最大表大小
    table = {bytes([i]): i for i in range(256)}
    string = b""
    compressed_data = []
    code = 256
    for symbol in data:
        symbol = bytes([symbol])
        string_plus_symbol = string + symbol
        if string_plus_symbol in table:
            string = string_plus_symbol
        else:
            compressed_data.append(table[string])
            if len(table) < max_table_size:
                table[string_plus_symbol] = code
                code += 1
            string = symbol
    if string:
        compressed_data.append(table[string])
    # 将整数列表转换为字节数据，每个代码由2个字节组成
    compressed_bytes = b"".join(int.to_bytes(code, length=2, byteorder='big') for code in compressed_data)
    return compressed_bytes

def decompress_rl_model(compressed_data):
    """
    使用 RL 模型的动作序列和长度信息进行无损解压缩。
    
    Args:
        compressed_data (bytes): 包含动作序列、长度信息和压缩数据的字节数据。
    
    Returns:
        bytes: 解压后的原始数据。
    """
    try:
        if len(compressed_data) < 4:
            print("错误: 压缩数据长度不足以包含动作序列长度。")
            return None
        
        # 读取动作序列长度
        idx = 0
        actions_length = int.from_bytes(compressed_data[idx:idx+4], byteorder='big')
        idx += 4
        
        # 读取动作序列
        actions = list(compressed_data[idx:idx+actions_length])
        idx += actions_length
        
        # 读取每个动作对应的数据长度
        lengths = []
        for _ in range(actions_length):
            if idx + 4 > len(compressed_data):
                print("错误: 压缩数据长度不足以包含长度信息。")
                return None
            length = int.from_bytes(compressed_data[idx:idx+4], byteorder='big')
            lengths.append(length)
            idx += 4
        
        # 剩余部分为实际的压缩数据
        compressed_bytes = compressed_data[idx:]
        byte_idx = 0  # 当前读取压缩数据的位置
        decompressed_data = bytearray()
        
        for i, action in enumerate(actions):
            length = lengths[i]
            print(f"处理动作 {i}: 动作编号 {action}, 数据长度 {length}")
            if byte_idx + length > len(compressed_bytes):
                print("错误: 压缩数据不足以解压当前动作的数据。")
                return None
            data_segment = compressed_bytes[byte_idx:byte_idx+length]
            byte_idx += length
            
            # 根据动作编号进行解压
            if action == 0:
                # 动作0: 直接添加原始数据
                decompressed_data.extend(data_segment)
                print(f"动作0: 添加 {length} 个字节的数据")
            elif action == 1:
                # 动作1: 使用 zlib 解压缩
                decompressed_state = zlib.decompress(data_segment)
                decompressed_data.extend(decompressed_state)
                print(f"动作1: 使用 zlib 解压缩，得到 {len(decompressed_state)} 个字节的数据")
            elif action == 2:
                # 动作2: 使用 bz2 解压缩
                decompressed_state = bz2.decompress(data_segment)
                decompressed_data.extend(decompressed_state)
                print(f"动作2: 使用 bz2 解压缩，得到 {len(decompressed_state)} 个字节的数据")
            elif action == 3:
                # 动作3: 不添加任何数据（在无损压缩中不建议）
                print(f"动作3: 不添加任何数据")
            else:
                # 未知动作，报错
                print(f"错误: 未知的动作编号 {action}")
                return None
                
        print("RLAgent 解压缩完成。")
        print(f"解压后数据大小: {len(decompressed_data)} 字节")
        return bytes(decompressed_data)
    except Exception as e:
        print(f"RLAgent 解压缩时出错: {e}")
        return None

def decompress_data(algorithm, compressed_data):
    """使用指定的算法解压缩数据，并返回解压后的数据。"""
    try:
        if algorithm == 'gorilla':
            try:
                import gorillacompression as gc
                print("开始使用 Gorilla 解压缩...")
                decompressed_data = gc.ValuesEncoder.decode_all({'encoded': compressed_data})['decoded']
                print("Gorilla 解压缩完成。")
                return decompressed_data
            except ModuleNotFoundError:
                print("错误: 未找到 'gorillacompression' 模块。请确保已正确安装或提供该模块。")
                return None
            except Exception as e:
                print(f"Gorilla 解压缩时出错: {e}")
                return None
        elif algorithm == 'snappy':
            try:
                print("开始使用 Snappy 解压缩...")
                decompressed_data = snappy.uncompress(compressed_data)
                print("Snappy 解压缩完成。")
                return decompressed_data
            except Exception as e:
                print(f"Snappy 解压缩时出错: {e}")
                return None
        elif algorithm == 'zlib':
            try:
                print("开始使用 Zlib 解压缩...")
                decompressed_data = zlib.decompress(compressed_data)
                print("Zlib 解压缩完成。")
                return decompressed_data
            except Exception as e:
                print(f"Zlib 解压缩时出错: {e}")
                return None
        elif algorithm == 'bz2':
            try:
                print("开始使用 Bz2 解压缩...")
                decompressed_data = bz2.decompress(compressed_data)
                print("Bz2 解压缩完成。")
                return decompressed_data
            except Exception as e:
                print(f"Bz2 解压缩时出错: {e}")
                return None
        elif algorithm == 'gzip':
            try:
                print("开始使用 Gzip 解压缩...")
                decompressed_data = gzip.decompress(compressed_data)
                print("Gzip 解压缩完成。")
                return decompressed_data
            except Exception as e:
                print(f"Gzip 解压缩时出错: {e}")
                return None
        elif algorithm == 'lzma':
            try:
                print("开始使用 Lzma 解压缩...")
                decompressed_data = lzma.decompress(compressed_data)
                print("Lzma 解压缩完成。")
                return decompressed_data
            except Exception as e:
                print(f"Lzma 解压缩时出错: {e}")
                return None
        elif algorithm == 'brotli':
            try:
                print("开始使用 Brotli 解压缩...")
                decompressed_data = brotli.decompress(compressed_data)
                print("Brotli 解压缩完成。")
                return decompressed_data
            except Exception as e:
                print(f"Brotli 解压缩时出错: {e}")
                return None
        elif algorithm == 'lz4':
            try:
                print("开始使用 Lz4 解压缩...")
                decompressed_data = lz4.frame.decompress(compressed_data)
                print("Lz4 解压缩完成。")
                return decompressed_data
            except Exception as e:
                print(f"Lz4 解压缩时出错: {e}")
                return None
        elif algorithm == 'lzw':
            try:
                print("开始使用 LZW 解压缩...")
                decompressed_data = lzw_decompress(compressed_data)
                print("LZW 解压缩完成。")
                return decompressed_data
            except Exception as e:
                print(f"LZW 解压缩时出错: {e}")
                return None
        elif algorithm == 'rl_model':
            try:
                print("开始使用 RL 模型解压缩...")
                decompressed_data = decompress_rl_model(compressed_data)
                if decompressed_data is not None:
                    print("RL 模型解压缩完成。")
                else:
                    print("RL 模型解压缩失败。")
                return decompressed_data
            except Exception as e:
                print(f"RL 模型解压缩时出错: {e}")
                return None
        else:
            raise ValueError(f"未知的压缩算法: {algorithm}")
    except Exception as e:
        print(f"使用 {algorithm} 解压缩时发生未捕捉的错误: {e}")
        return None

def compress_data(algorithm, data, rl_agent=None):
    """使用指定的算法压缩数据，并返回压缩后的数据和耗时。"""
    start_time = time.process_time()
    compressed = None
    try:
        if algorithm == 'gorilla':
            try:
                import gorillacompression as gc
                print("开始使用 Gorilla 压缩...")
                compressed = gc.ValuesEncoder.encode_all(data)['encoded']
                print("Gorilla 压缩完成。")
            except ModuleNotFoundError:
                print("错误: 未找到 'gorillacompression' 模块。请确保已正确安装或提供该模块。")
                return None, 0
            except Exception as e:
                print(f"Gorilla 压缩时出错: {e}")
                return None, 0
        elif algorithm == 'snappy':
            try:
                print("开始使用 Snappy 压缩...")
                compressed = snappy.compress(data)
                print("Snappy 压缩完成。")
            except Exception as e:
                print(f"Snappy 压缩时出错: {e}")
                return None, 0
        elif algorithm == 'zlib':
            try:
                print("开始使用 Zlib 压缩...")
                compressed = zlib.compress(data)
                print("Zlib 压缩完成。")
            except Exception as e:
                print(f"Zlib 压缩时出错: {e}")
                return None, 0
        elif algorithm == 'bz2':
            try:
                print("开始使用 Bz2 压缩...")
                compressed = bz2.compress(data)
                print("Bz2 压缩完成。")
            except Exception as e:
                print(f"Bz2 压缩时出错: {e}")
                return None, 0
        elif algorithm == 'gzip':
            try:
                print("开始使用 Gzip 压缩...")
                compressed = gzip.compress(data)
                print("Gzip 压缩完成。")
            except Exception as e:
                print(f"Gzip 压缩时出错: {e}")
                return None, 0
        elif algorithm == 'lzma':
            try:
                print("开始使用 Lzma 压缩...")
                compressed = lzma.compress(data)
                print("Lzma 压缩完成。")
            except Exception as e:
                print(f"Lzma 压缩时出错: {e}")
                return None, 0
        elif algorithm == 'brotli':
            try:
                print("开始使用 Brotli 压缩...")
                compressed = brotli.compress(data)
                print("Brotli 压缩完成。")
            except Exception as e:
                print(f"Brotli 压缩时出错: {e}")
                return None, 0
        elif algorithm == 'lz4':
            try:
                print("开始使用 Lz4 压缩...")
                compressed = lz4.frame.compress(data)
                print("Lz4 压缩完成。")
            except Exception as e:
                print(f"Lz4 压缩时出错: {e}")
                return None, 0
        elif algorithm == 'lzw':
            try:
                print("开始使用 LZW 压缩...")
                compressed = lzw_compress(data)
                print("LZW 压缩完成。")
            except Exception as e:
                print(f"LZW 压缩时出错: {e}")
                return None, 0
        elif algorithm == 'rl_model':
            if rl_agent is None:
                raise ValueError("RL 模型代理未提供。")
            try:
                print("开始使用 RL 模型进行压缩...")
                compressed = rl_agent.compress(data)
                if compressed:
                    print("RL 模型压缩完成。")
                else:
                    print("RL 模型压缩失败，返回空数据。")
            except Exception as e:
                print(f"RL 模型压缩时出错: {e}")
                return None, 0
        else:
            raise ValueError(f"未知的压缩算法: {algorithm}")
    except Exception as e:
        print(f"使用 {algorithm} 压缩时发生未捕捉的错误: {e}")
        return None, 0
    elapsed_time = time.process_time() - start_time
    return compressed, elapsed_time

def load_rl_model(model_path):
    """从指定路径加载训练好的 RL 模型。"""
    try:
        # 根据 RLAgent 的构造函数提供必要的参数
        state_size = 5  # 根据训练时的实际情况调整
        action_size = 4  # 根据训练时的实际情况调整
        rl_agent = RLAgent(state_size, action_size)
        print(f"加载 RL 模型从: {model_path}...")
        rl_agent.load(model_path)  # 调用 RLAgent 的 load 方法
        rl_agent.eval()
        print("RL 模型加载成功。")
        
        # 打印模型的 state_dict
        print("模型的 state_dict:")
        for param_tensor in rl_agent.state_dict():
            print(f"{param_tensor}\t{rl_agent.state_dict()[param_tensor].size()}")
        
        return rl_agent
    except FileNotFoundError:
        print(f"RL 模型文件未找到: {model_path}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"加载 RL 模型时出错 (可能是模型架构不匹配): {e}")
        sys.exit(1)
    except Exception as e:
        print(f"加载 RL 模型时出错: {e}")
        sys.exit(1)

def main():
    # 加载训练好的 RL 模型
    model_path = './models/rl_agent_model.pth'  # 根据实际路径调整
    rl_agent = load_rl_model(model_path)

    # 初始化压缩算法列表
    algorithms = [
        'gorilla', 'snappy', 'zlib', 'bz2', 'gzip',
        'lzma', 'brotli', 'lz4', 'lzw', 'rl_model'  # 包含 LZW 和 RL 模型
    ]
    
    ratios = {alg: [] for alg in algorithms}
    times = {alg: [] for alg in algorithms}
    # files = ['data/input.txt','data/2num.txt']  # 根据需要添加更多文件
    files = ['data/1text.txt']  # 根据需要添加更多文件

    for file_path in files:
        print(f"\n正在处理文件: {file_path}")
        if not os.path.exists(file_path):
            print(f"文件未找到: {file_path}")
            for alg in algorithms:
                ratios[alg].append(0)
                times[alg].append(0)
            continue

        # 读取文件内容
        original_data = read_file(file_path)
        if original_data is None:
            for alg in algorithms:
                ratios[alg].append(0)
                times[alg].append(0)
            continue

        print(f"原始大小: {len(original_data)} 字节")

        compressed_results = {}
        for alg in algorithms:
            try:
                if alg == 'rl_model':
                    compressed_data, elapsed = compress_data(alg, original_data, rl_agent=rl_agent)
                else:
                    compressed_data, elapsed = compress_data(alg, original_data)
                
                if compressed_data is not None:
                    ratio = calculate_compression_ratio(original_data, compressed_data)
                    print(f"{alg.capitalize()} 压缩 - 压缩比: {ratio:.2f}, 时间: {elapsed:.6f} 秒")
                else:
                    ratio = 0
                    elapsed = 0
                    print(f"{alg.capitalize()} 压缩失败，压缩比设为0，时间设为0秒。")
                
                ratios[alg].append(ratio)
                times[alg].append(elapsed)
                compressed_results[alg] = compressed_data
            except Exception as e:
                print(f"{alg.capitalize()} 压缩失败: {e}")
                ratios[alg].append(0)
                times[alg].append(0)
                compressed_results[alg] = None
        
        # 解压缩并验证
        for alg in algorithms:
            if alg == 'rl_model':
                # RL 模型的解压缩
                print(f"\n正在解压缩使用 {alg} 压缩的数据...")
                compressed_data = compressed_results.get(alg)
                if compressed_data is None:
                    print(f"{alg.capitalize()} 压缩数据不存在，跳过解压缩。")
                    continue
                decompressed_data = decompress_rl_model(compressed_data)
                
                if decompressed_data is not None:
                    decompressed_file_path = f'decompressed_{alg}.bin'
                    write_file(decompressed_file_path, decompressed_data)
                    
                    # 验证解压缩数据与原始数据是否一致
                    if decompressed_data == original_data:
                        print(f"{alg.capitalize()} 解压缩验证成功。")
                    else:
                        print(f"{alg.capitalize()} 解压缩验证失败：解压缩数据与原始数据不一致。")
                else:
                    print(f"{alg.capitalize()} 解压缩失败。")
            elif alg == 'lzw':
                # LZW 解压缩
                print(f"\n正在解压缩使用 {alg} 压缩的数据...")
                compressed_data = compressed_results.get(alg)
                if compressed_data is None:
                    print(f"{alg.capitalize()} 压缩数据不存在，跳过解压缩。")
                    continue
                decompressed_data = lzw_decompress(compressed_data)
                
                if decompressed_data is not None:
                    decompressed_file_path = f'decompressed_{alg}.bin'
                    write_file(decompressed_file_path, decompressed_data)
                    
                    # 验证解压缩数据与原始数据是否一致
                    if decompressed_data == original_data:
                        print(f"{alg.capitalize()} 解压缩验证成功。")
                    else:
                        print(f"{alg.capitalize()} 解压缩验证失败：解压缩数据与原始数据不一致。")
                else:
                    print(f"{alg.capitalize()} 解压缩失败。")
            else:
                # 标准压缩算法的解压缩
                print(f"\n正在解压缩使用 {alg} 压缩的数据...")
                compressed_data = compressed_results.get(alg)
                if compressed_data is None:
                    print(f"{alg.capitalize()} 压缩数据不存在，跳过解压缩。")
                    continue
                decompressed_data = decompress_data(alg, compressed_data)
                
                if decompressed_data is not None:
                    decompressed_file_path = f'decompressed_{alg}.bin'
                    write_file(decompressed_file_path, decompressed_data)
                    
                    # 验证解压缩数据与原始数据是否一致
                    if decompressed_data == original_data:
                        print(f"{alg.capitalize()} 解压缩验证成功。")
                    else:
                        print(f"{alg.capitalize()} 解压缩验证失败：解压缩数据与原始数据不一致。")
                else:
                    print(f"{alg.capitalize()} 解压缩失败。")

    # 打印压缩结果
    print("\n压缩比:")
    for alg in algorithms:
        print(f"{alg}: {ratios[alg]}")

    print("\n压缩时间:")
    for alg in algorithms:
        print(f"{alg}: {times[alg]}")

    # 将结果保存到 CSV
    try:
        with open('compression_results_text.csv', 'w', newline='') as csvfile:
            fieldnames = ['file'] + [f"{alg}_ratio" for alg in algorithms] + [f"{alg}_time" for alg in algorithms]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i, file_path in enumerate(files):
                row = {'file': file_path}
                for alg in algorithms:
                    row[f"{alg}_ratio"] = ratios[alg][i]
                    row[f"{alg}_time"] = times[alg][i]
                writer.writerow(row)
        print("压缩结果已保存到 'compression_results.csv'")
    except Exception as e:
        print(f"保存压缩结果到 CSV 时出错: {e}")

    # 绘制压缩比图表
    try:
        plt.figure(figsize=(14, 7))
        for alg in algorithms:
            plt.plot(files, ratios[alg], marker='o', label=alg.capitalize())
        plt.xlabel("文件")
        plt.ylabel("压缩比")
        plt.title("不同算法的压缩比")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('compression_ratios.png')
        plt.show()
    except Exception as e:
        print(f"绘制压缩比图表时出错: {e}")

    # 绘制压缩时间图表
    try:
        plt.figure(figsize=(14, 7))
        for alg in algorithms:
            plt.plot(files, times[alg], marker='o', label=alg.capitalize())
        plt.xlabel("文件")
        plt.ylabel("压缩时间 (秒)")
        plt.title("不同算法的压缩时间")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('compression_times.png')
        plt.show()
    except Exception as e:
        print(f"绘制压缩时间图表时出错: {e}")

if __name__ == '__main__':
    main()