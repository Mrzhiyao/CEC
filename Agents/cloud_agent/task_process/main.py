from transform_without_result import task_get
from agents import Agents
from tasks import get_tasks, get_tasks_conclusion
import time
from crewai import Crew, Process
from insert_embedding import insert_result
import os
from pymilvus import  MilvusClient
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from node_send import send_file
from node_send_host_new import send_host_file
from node_send_answer import send_host_answer
from read_task import read_task
from read_sequence import read_sequence
from mv_file import mvfile_filename
from time_order import get_time_file
from threading_task_processing import task_processing
import threading

import subprocess
import os 

os.environ["OTEL_SDK_DISABLED"] = "true"

def set_read_only_for_everyone(file_path):
    # 清除所有现有的权限
    subprocess.run(['icacls', file_path, '/reset'], capture_output=True, text=True)
    
    # 使用 icacls 命令设置权限为所有用户只读
    result = subprocess.run(['icacls', file_path, '/grant:r', 'Everyone:R'], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return True

def is_file_read_only_for_everyone(file_path):
    # 使用 icacls 命令获取文件权限
    result = subprocess.run(['icacls', file_path], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    
    # 解析 icacls 的输出
    permissions = result.stdout
    
    # 检查 Everyone 组是否有只读（R）权限
    if 'Everyone:(R)' in permissions:
        return True
    return False

def check_file_usage(file_path):
    try:
        # 尝试以独占方式打开文件
        with open(file_path, 'x'):
            pass
        # 如果上述代码块没有抛出异常，则文件未被使用
        return False
    except FileExistsError:
        # 如果文件已存在，说明文件正在被使用
        return True
    
def kill_thread_after(timeout, thread):
    """设置定时器在指定时间后终止线程"""
    thread.start()
    timer = threading.Timer(timeout, thread._stop)
    timer.start()
    # thread.join()
    # timer.join()  # 等待定时器线程结束

import shutil
def delete_files_with_char(folder_path, char_to_search):
    """删除指定文件夹中文件名包含特定字符的文件"""
    for filename in os.listdir(folder_path):
        if char_to_search in filename:
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                shutil.rmtree(file_path)
 


if __name__ == '__main__':
    _BASE = os.path.dirname(os.path.abspath(__file__))
    _AGENT_BASE = os.path.dirname(_BASE)
    path = os.path.join(_AGENT_BASE, 'file')
    path_des = os.path.join(_AGENT_BASE, 'file_log') 
    agents = Agents()
    # record = []
    st_time = time.time()
    while 1: 
        print('主进程检查任务中')
        time.sleep(2)     
        max_time = 2
        try:   
            files, times = get_time_file(path)
            # i是任务名
            # print('dada', time.time() - st_time, (time.time() - st_time)/5)
            for i in files:
                # print(i,i[:-6],i[0:10]+'.'+mowei[10:])
                if 'execution_sequence' in i or 'task_send_number' in i:
                    continue
                mowei = i[:-6]
                if (time.time() - float(i[0:10]+'.'+mowei[10:]) ) > 320:
                    print('任务已超时')
                    # 使用示例
                    char_to_search = i[:-6]  # 替换为要搜索的特定字符
                    delete_files_with_char(path, char_to_search)
                    filename = os.path.join(_BASE, 'drop_task.txt')
                    # 要追加的内容
                    text_to_append = i[:-4] + '\n'
                    # 打开文件以追加，如果文件不存在则创建
                    with open(filename, 'a+') as file:
                        file.write(text_to_append)
                
            for i in files:

                try:
                    # 检查文件权限
                    # is_read_only = is_file_read_only_for_everyone(path + '/' +i)
                    # # 设置文件权限为所有用户只读
                    # file_stats = os.stat(path + '/' +i)

                    # # print('oct(file_stats.st_mode)', oct(file_stats.st_mode),path + '/' +i)
                    # if 'execution_sequence' in i and is_read_only == False:
                    file_stats = os.stat(path + '/' +i)
                    # print('oct(file_stats.st_mode)', oct(file_stats.st_mode))
                    if 'execution_sequence' in i and oct(file_stats.st_mode) != '0o100755' and 'sub' not in i:
                        print('file_stats', file_stats.st_mode, oct(file_stats.st_mode))
                        thread = threading.Thread(target=task_processing, args=([path,path_des,i]))
                        kill_thread_after(320, thread)  # 2秒后杀死线程
                except:
                    print('任务已被完成')

        except:
            print('文件不存在')


