# test_rl_decompress.py

import sys
import os
from lzw_agent import RLAgent
from lzw_evaluate import decompress_rl_model, compress_data, load_rl_model, read_file, write_file

def test_rl_decompression():
    model_path = './models/rl_agent_model.pth'
    data_path = 'data/2num.txt'  # 替换为您的测试文件

    if not os.path.exists(model_path):
        print(f"模型文件未找到: {model_path}")
        sys.exit(1)
    if not os.path.exists(data_path):
        print(f"数据文件未找到: {data_path}")
        sys.exit(1)

    # 加载 RL 模型
    state_size = 5
    action_size = 4
    rl_agent = RLAgent(state_size, action_size)
    try:
        rl_agent.load(model_path)
        rl_agent.eval()
        print("RL 模型加载成功。")
    except Exception as e:
        print(f"加载 RL 模型时出错: {e}")
        sys.exit(1)

    # 读取原始数据
    original_data = read_file(data_path)
    if original_data is None:
        print("读取数据失败。")
        sys.exit(1)
    print(f"原始数据大小: {len(original_data)} 字节")

    # 压缩数据
    compressed_data, compress_time = compress_data('rl_model', original_data, rl_agent=rl_agent)
    if compressed_data is None:
        print("RL 模型压缩失败。")
        sys.exit(1)
    print(f"RL 模型压缩完成，压缩后大小: {len(compressed_data)} 字节, 时间: {compress_time:.6f} 秒")

    # 解压缩数据
    decompressed_data = decompress_rl_model(compressed_data)
    if decompressed_data is None:
        print("RL 模型解压缩失败。")
        sys.exit(1)
    print(f"RL 模型解压缩完成，解压后大小: {len(decompressed_data)} 字节, 时间: {compress_time:.6f} 秒")

    # 保存解压缩数据
    decompressed_file_path = 'decompressed_rl_model.bin'
    write_file(decompressed_file_path, decompressed_data)

    # 验证解压缩数据与原始数据是否一致
    if decompressed_data == original_data:
        print("RL 模型解压缩验证成功：解压缩数据与原始数据一致。")
    else:
        print("RL 模型解压缩验证失败：解压缩数据与原始数据不一致。")

if __name__ == '__main__':
    test_rl_decompression()