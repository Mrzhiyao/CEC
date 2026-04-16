import asyncio
import subprocess
import psutil
import os
import time
import torch
import torch.nn as nn
import time
def monitor_resources(interval=0.1):
    """
    定期监控 CPU、内存、GPU 和显存的使用情况。

    参数:
        interval: 采样间隔（以秒为单位），默认为 1 秒。

    返回:
        列表: 包含每次采样的 CPU、内存、GPU 和显存使用情况的列表。
    """
    # 初始化列表，存储采样结果
    cpu_usages = [] #表示 CPU 使用率的百分比
    memory_usages = [] #表示当前已经使用的内存大小（以字节为单位）。
    gpu_usages = [] #表示 GPU 使用率的百分比
    memory_utilizations = [] #表示显存（GPU 内存）使用率的百分比。
    memory_used = [] #表示当前已经使用的显存大小（以mb为单位）。

    # 在模型运行期间循环采样
    # 获取当前进程的 CPU 和内存使用情况
    process = psutil.Process(os.getpid())
    cpu_usage = process.cpu_percent(interval=interval)
    memory_info = process.memory_info()
    memory_usage = memory_info.rss  # 常驻集大小（以字节为单位）

    # 调用 nvidia-smi 命令，查询 GPU 和显存使用情况
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True)
    data = result.stdout.strip().split('\n')[0]
    gpu_utilization, memory_utilization, memory_used_val = map(float, data.split(','))

    # 将采样结果添加到列表中
    cpu_usages.append(cpu_usage)
    memory_usages.append(memory_usage)
    gpu_usages.append(gpu_utilization)
    memory_utilizations.append(memory_utilization)
    memory_used.append(memory_used_val)

    # 等待下一个采样间隔
    # await asyncio.sleep(interval)

    # 输出平均值
    print(f"CPU 使用率 : {cpu_usage:.2f}%")
    print(f"已经使用的内存大小 : {(memory_usage / 1048576):.2f} MB")
    print(f"GPU 使用率 : {gpu_utilization:.2f}%")
    print(f"显存（GPU 内存）使用率 : {memory_utilization:.2f}%")
    print(f"已经使用的显存大小 : {(memory_used_val):.2f} MB")
    print(gpu_utilization,memory_utilization,memory_used_val)
    
    return gpu_utilization,memory_utilization,memory_used_val

# monitor_resources()
