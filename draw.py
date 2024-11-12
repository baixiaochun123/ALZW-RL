# draw.py
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def drawline(ratio, x):
    print(x)
    plt.figure(figsize=(12, 6))
    plt.plot(x, ratio['gorilla'], marker='o', label='Gorilla')
    plt.plot(x, ratio['s'], marker='o', label='Snappy')
    plt.plot(x, ratio['z'], marker='o', label='Zlib')
    plt.plot(x, ratio['b'], marker='o', label='Bz2')
    plt.plot(x, ratio['gz'], marker='o', label='Gzip')
    plt.plot(x, ratio['lzma'], marker='o', label='LZMA')
    plt.plot(x, ratio['brotli'], marker='o', label='Brotli')
    plt.plot(x, ratio['lz4'], marker='o', label='LZ4')
    plt.plot(x, ratio['influx'], marker='x', label='InfluxDB')
    plt.plot(x, ratio['my'], marker='*', label='RL-method')

    plt.ylim(0, max([max(v) for v in ratio.values() if v]) * 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.xticks(x, ['mit100','mit101','mit102','mit103','mit104','mit105'], rotation=45)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15, right=0.75)
    plt.xlabel("Datasets")  # X轴标签
    plt.ylabel("Compression Ratio")  # Y轴标签
    plt.title("Time-Series Compression Ratios")  # 标题
    plt.grid(True)
    plt.show()

def drawtime(time_dict, x):
    print(x)
    plt.figure(figsize=(12, 6))
    plt.plot(x, time_dict['gorilla'], marker='o', label='Gorilla')
    plt.plot(x, time_dict['s'], marker='o', label='Snappy')
    plt.plot(x, time_dict['z'], marker='o', label='Zlib')
    plt.plot(x, time_dict['b'], marker='o', label='Bz2')
    plt.plot(x, time_dict['gz'], marker='o', label='Gzip')
    plt.plot(x, time_dict['lzma'], marker='o', label='LZMA')
    plt.plot(x, time_dict['brotli'], marker='o', label='Brotli')
    plt.plot(x, time_dict['lz4'], marker='o', label='LZ4')
    plt.plot(x, time_dict['influx'], marker='x', label='InfluxDB')
    plt.plot(x, time_dict['my'], marker='*', label='RL-method')

    plt.ylim(0, max([max(v) for v in time_dict.values() if v]) * 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.xticks(x, ['mit100','mit101','mit102','mit103','mit104','mit105'], rotation=45)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15, right=0.75)
    plt.xlabel("Datasets")  # X轴标签
    plt.ylabel("Compression Time (s)")  # Y轴标签
    plt.title("Time-Series Compression Times")  # 标题
    plt.grid(True)
    plt.show()