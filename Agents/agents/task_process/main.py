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

client = MilvusClient(
    uri='http://192.168.2.7:19530',
    token='root:Milvus'
)
connections.connect("default", host="192.168.2.7", port="19530")
# oasst1_val = Collection("crewai_agents1")
# oasst1_val = Collection("crewai_agents1_algorithm1")
oasst1_val = Collection("crewai_agents1_algorithm1_subvec")
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
        max_time = 2
        time.sleep(1)     
        try:
            files, times = get_time_file(path)
            # i是任务名
            # print('dada', time.time() - st_time, (time.time() - st_time)/5)

            for i in files:
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
                # 检查文件是否被使用
                try:
                    file_stats = os.stat(path + '/' +i)
                    # print('oct(file_stats.st_mode)', oct(file_stats.st_mode))
                    if 'execution_sequence' in i and oct(file_stats.st_mode) != '0o100755' and 'sub' not in i:
                    # if 'execution_sequence' in i:
                        # thread = threading.Thread(target=task_processing, args=([path,path_des,i,oasst1_val]))
                        # thread.daemon = True
                        # thread.start()
                        # thread.join(max_time)
                        # 创建一个线程，并设置超时时间
                        thread = threading.Thread(target=task_processing, args=([path,path_des,i,oasst1_val]))
                        kill_thread_after(320, thread)  # 2秒后杀死线程
                except:
                    print('任务已被完成')
        except:
            print('文件不存在')