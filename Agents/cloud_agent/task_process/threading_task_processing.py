from transform_without_result import task_get
from agents import Agents
from tasks import get_tasks, get_tasks_conclusion
# from tasks_local import get_tasks_local, get_tasks_conclusion_local
import time
from crewai import Crew, Process
from insert_embedding import insert_result
import os
from node_send import send_file
from node_send_host_new import send_host_file
from node_send_answer import send_host_answer

from node_send_host_new_time import send_host_file_time
from node_send_answer_time import send_host_answer_time
from read_task import read_task
from read_sequence import read_sequence
from mv_file import mvfile_filename
from time_order import get_time_file
from gpu_size import monitor_resources
# 创建类型id
from utils import get_type_id
from llm_api import llm_solve_problem

import threading
import subprocess
import glob
from pymilvus import  MilvusClient
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from insert_keywords_embedding import insert_keywords_result
from search_milvus import search_data
os.environ["OTEL_SDK_DISABLED"] = "true"

_BASE = os.path.dirname(os.path.abspath(__file__))
_AGENT_BASE = os.path.dirname(_BASE)
_FILE_RESULT = os.path.join(_AGENT_BASE, 'file_result')
_SUBTASK_TIME = os.path.join(_BASE, 'subtask_result', 'task_time')
_SUBTASK_RESULT = os.path.join(_BASE, 'subtask_result', 'task_result')

def count_files_with_specific_content(directory, content):
    # 获取指定目录下的所有文件
    files = glob.glob(os.path.join(directory, '*'))
    
    # 初始化计数器
    count = 0

    # 遍历所有文件
    for file in files:
        if os.path.isfile(file):  # 确保是文件而不是文件夹
            filename = os.path.basename(file)  # 获取文件名
            if content in filename and 'execution_sequence' not in filename and 'send_number' not in filename and 'content_time' not in filename:  # 检查文件名是否包含指定字符串且不包含'x'
                count += 1

    return count


def find_files_containing_name(directory_path, name_part):
    try:
        # 获取目录中的所有文件和文件夹
        items = os.listdir(directory_path)
        
        # 过滤出文件并检查名称是否包含指定部分
        matching_files = [item for item in items if os.path.isfile(os.path.join(directory_path, item)) and name_part in item]
        
        return matching_files
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

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


def insert_milvus_record(node):
    i = node
    if i == 0:
        connections.disconnect("default")
        connections.connect("default", host="192.168.2.26", port="19530")
        from pymilvus import MilvusClient
        client = MilvusClient(
            uri='http://192.168.2.26:19530'
        )
        collection_name = "crewai_agents1_algorithm1_insert"
        collection = Collection("crewai_agents1_algorithm1_insert")
        collection_keywords_name = "crewai_agents1_keywords_algorithm1_insert"
        collection_keywords = Collection("crewai_agents1_keywords_algorithm1_insert")

    elif i == 1:
        connections.disconnect("default")
        connections.connect("default", host="192.168.2.6", port="19530")
        from pymilvus import MilvusClient
        client = MilvusClient(
            uri='http://192.168.2.6:19530'
        )
        collection_name = "crewai_agents1_algorithm1_insert"
        collection = Collection("crewai_agents1_algorithm1_insert")
        collection_keywords_name = "crewai_agents1_keywords_algorithm1_insert"
        collection_keywords = Collection("crewai_agents1_keywords_algorithm1_insert")
    else:
        connections.disconnect("default")
        connections.connect("default", host="192.168.2.7", port="19530")
        from pymilvus import MilvusClient
        client = MilvusClient(
            uri='http://192.168.2.7:19530'
        )
        collection_name = "crewai_agents1_algorithm1_insert"
        collection = Collection("crewai_agents1_algorithm1_insert")
        collection_keywords_name = "crewai_agents1_keywords_algorithm1_insert"
        collection_keywords = Collection("crewai_agents1_keywords_algorithm1_insert")
    
    return collection, collection_keywords

def task_processing(path,path_des,i):
    print('iiiiii:', i)
    program_time = time.time()
    agents = Agents()
    execution_sequence = read_sequence(path + '/' + i)
    tasks_list = execution_sequence.split('\n')
    print(type(tasks_list))
    task_ex = tasks_list[:-1]

    # 
    sub_execution_files = find_files_containing_name(path, 'sub_execution_sequence_' + i[19:-4])
    if len(sub_execution_files) != 0:
        for sub_execution_file in sub_execution_files:
            # 读取内容
            out = read_sequence(path + '/' + sub_execution_file)
            ppp_tasks_list = out.split('\n')
            for content in ppp_tasks_list[:-1]:
                task_ex.append(content)
    print('task_ex', task_ex)
    # print(task_ex)
    print('task_list',tasks_list)


    # 检查任务
    index = 0
    task_final = 0
    last_file_name = ''
    tasks_pre = []
    for z in tasks_list[:-1]:
        # print(z)
        file_task, ip = z.split(' ')
        # 先保存一下任务执行列表的内容
        tasks_pre.append(file_task)
        if index + 1 == len(tasks_list[:-1]):
            # 判断最后一个任务在本节点上, 则不发送
            last_file_name = file_task
            sendip = ip
            if ip == '192.168.2.26':
                task_final = 1
        index = index + 1
    print('last_file_name', last_file_name)
    # 总任务长度（包括了最终任务）
    tasks_len = index
    # 检查任务列表中的内容是否还有未完成的
    sub_tasks_pre = []
    sub_execution_files = find_files_containing_name(path, 'sub_execution_sequence_' + file_task[:-6])
    if len(sub_execution_files) != 0:
        for sub_execution_file in sub_execution_files:
            # 读取内容
            out = read_sequence(path + '/' + sub_execution_file)
            ppp_tasks_list = out.split('\n')
            for content in ppp_tasks_list[:-1]:
                if content.split(' ')[1] == '192.168.2.26':
                    sub_tasks_pre.append(content.split(' ')[0])

    
    while 1:
        print('qidong:', i)
        if (time.time() - program_time)/300 > 1:
            print('超时仍未完成任务')

            try:
                mvfile_filename(path, path_des, 'execution_sequence_' + file_task[:-6]+'.txt')
                if len(sub_execution_files) != 0:
                    for sub_execution_file in sub_execution_files:
                        # 读取内容
                        mvfile_filename(path, path_des, sub_execution_file)
            except:
                pass
            
            import sys
            sys.exit(0)
        permission_number = 0o755

        if 'sub_' in i:
            try:
                os.chmod(path + '/' + i, permission_number)
                files, times = get_time_file(path)
                time.sleep(5)
            except:
                print('跳出任务执行，已完成')
                break
        else:
            try:
                os.chmod(path + '/' + i[:-4]+'.txt', permission_number)
                files, times = get_time_file(path)
                time.sleep(5)
            except:
                print('ex跳出任务执行，已完成')
                break

        file_count_number = count_files_with_specific_content(path, file_task[:-6])
        if file_count_number == 0:
            try:
                mvfile_filename(path, path_des, 'execution_sequence_' + file_task[:-6]+'.txt')
                if len(sub_execution_files) != 0:
                    for sub_execution_file in sub_execution_files:
                        # 读取内容
                        mvfile_filename(path, path_des, sub_execution_file)
                import sys
                sys.exit(0)
            except:
                pass
        # if len(set(files) & set(tasks_pre)) == 0 and len(set(files) & set(sub_tasks_pre)) == 0:
        #     print('set1', set(files)) 
        #     print('set2', set(tasks_pre[:-1]))
        #     # mvfile_filename(path, path_des, 'execution_sequence_' + last_file_name[:-6]+'.txt')
        #     # import sys
        #     # sys.exit(0)
        # else:
        for z in task_ex:
            # print(z)
            file_task, ip = z.split(' ')
            # print('file_task:',file_task)
            # 看任务是否还在本地中
            if file_task in files:
                # print('last_file_namelast_file_namelast_file_name',last_file_name)
                # 普通任务
                print('file_task != last_file_name', file_task, last_file_name, file_task != last_file_name)
                if  file_task != last_file_name:
                    # 判断是否还存在前置任务：
                    # if file_task 'sub_execution_sequence_' + file_task[] 17188547010770657_0'
                    sub_execution_files = find_files_containing_name(path,'sub_execution_sequence_' + file_task[:-6])
                    print('file_task[:-6]file_task[:-6]', file_task[:-6])

                    if len(sub_execution_files) == 0:
                        # 普通任务
                        print('无子任务 file:', file_task)
                        task = read_task(path + '/' + file_task)
                        if len(task)>1000:
                            task = task[:1000]
                        print('tasks:\n', task)

                        # connections.disconnect("default")
                        # client = MilvusClient(
                        #     uri='http://192.168.2.7:19530',
                        #     token='root:Milvus'
                        # )
                        # connections.connect("default", host="192.168.2.7", port="19530")
                        # oasst1_val = Collection("crewai_agents1_algorithm1_subvec")
                        # text_id, text_sim, text, text_filed, text_emotion = search_data(oasst1_val, task)

                        # task = task + 'The reference information for the task is as follows:\n' + text[0]
                        
                        tasks = get_tasks(task)

                        # result = llm_solve_problem(tasks)
                        crew = Crew(
                            agents=[agents.Comprehensive()],
                            tasks=tasks,
                            verbose=2,  # You can set it to 1 or 2 to different logging levels
                            full_output=False
                        )
                    
                        # try:
                        #     result = crew.kickoff()
                        #     # break
                        # except:
                        #     crew = Crew(
                        #     agents=[agents.Comprehensive_without()],
                        #     tasks=tasks,
                        #     verbose=2,  # You can set it to 1 or 2 to different logging levels
                        #     full_output=False
                        # )
                        result = crew.kickoff()
                        print("######################")
                        print(result)

                        type_id = get_type_id()
                        # for sert_i in range(3):
                        #     oasst1_val, oasst1_keywords = insert_milvus_record(sert_i)
                        #     q_id, a_id = insert_result(oasst1_val, task, result, type_id)
                        #     insert_keywords_result(oasst1_keywords, task, result, type_id, q_id, a_id)

                        s_t = time.time()
                        print('insert_time', time.time()-s_t)
                        with open(_FILE_RESULT + '/' + file_task , 'a', encoding='utf-8') as file:
                            file.write(str(result))
                        file.close()

                        if sendip != '192.168.2.26':
                            print('需要转发任务' + file_task + ', 任务目标地:' + sendip)
                            if sendip == '192.168.2.8':
                                send_file(_FILE_RESULT + '/' + file_task, '192.168.2.26')  
                            else:  
                                send_file(_FILE_RESULT + '/' + file_task, sendip)

                        with open(_SUBTASK_TIME + '/' + file_task, 'a') as file:
                            file.write(str(float(time.time())))
                        file.close()

                        # result save
                        with open(_SUBTASK_RESULT + '/' + file_task , 'a', encoding='utf-8') as file:
                            file.write(str(result))
                        file.close()

                        send_host_file(_FILE_RESULT + '/' + file_task, '192.168.2.26')
                        send_host_file_time(_SUBTASK_TIME + '/' + file_task, '192.168.2.26')
                        mvfile_filename(path, path_des, file_task)
                    else:

                        for sub_execution_file in sub_execution_files:
                            # 读取内容
                            out = read_sequence(path + '/' + sub_execution_file)
                            print('outoutout', out)
                            tasks_list = out.split('\n')
                            sub_lists = []
                            sub_index_lists = []
                            for content in tasks_list[:-1]:
                                # print(content.split(' ')[0])
                                sub_lists.append(content.split(' ')[0])
                                sub_index_lists.append(content.split(' ')[1])
                            local_pre = os.listdir(_FILE_RESULT + '/')
                            # print(sub_lists)
                            # 本身属于子任务组里的小任务
                            # print('------------------------------------------------------------')
                            print(file_task[:-6] + '_' + str(int(file_task[-5])-1) + '.txt')
                            if file_task in sub_lists:
                                
                                sub_index = sub_lists.index(file_task)
                                # print(sub_index)
                                if sub_index != len(sub_lists)-1:
                                    sub_send_ip = sub_index_lists[sub_index + 1]
                                else:
                                    sub_send_ip = '192.168.2.26'


                                if sub_index == 0:
                                    print('首任务，正常执行')
                                    print('file:',file_task)
                                    task = read_task(path + '/' + file_task)
                                    if len(task)>1000:
                                        task = task[:1000]
                                    print('tasks:\n', task)


                                    # connections.disconnect("default")
                                    # client = MilvusClient(
                                    #     uri='http://192.168.2.7:19530',
                                    #     token='root:Milvus'
                                    # )
                                    # connections.connect("default", host="192.168.2.7", port="19530")
                                    # oasst1_val = Collection("crewai_agents1_algorithm1_subvec")
                                    # text_id, text_sim, text, text_filed, text_emotion = search_data(oasst1_val, task)

                                    # task = task + 'The reference information for the task is as follows:\n' + text[0]
                                    tasks = get_tasks(task)

                                else:
                                    if len(set(sub_lists[:sub_index]) & set(local_pre)) != len(sub_lists[:sub_index]):
                                        continue
                                    else:
                                        print('后续任务')
                                        tasks_content = []
                                        print('file:', file_task)
                                        task = read_task(path + '/' + file_task)
                                        print('tasks:\n', task)
                                        for n in local_pre:
                                            if n in sub_lists[:sub_index]:
                                                tasks_content.append(read_sequence(_FILE_RESULT + '/' + n))

                                        task = task + '\n' + 'The relevant information for the task is as follows:'
                                        for con in tasks_content:
                                            task = task + '\n' + con 
                                        if len(task)>1000:
                                            task = task[:1000]
                                        # tasks = get_tasks(task)


                                
                                tasks = get_tasks(task)

                                crew = Crew(
                                    agents=[agents.Comprehensive()],
                                    tasks=tasks,
                                    verbose=2,  # You can set it to 1 or 2 to different logging levels
                                    full_output=False
                                )
                                # while 1:
                                # try:
                                #     result = crew.kickoff()
                                    # break
                                # except:
                                #     crew = Crew(
                                #     agents=[agents.Comprehensive_without()],
                                #     tasks=tasks,
                                #     verbose=2,  # You can set it to 1 or 2 to different logging levels
                                #     full_output=False
                                # )
                                result = crew.kickoff()
                                # result = crew.kickoff()
                                print("######################")
                                print(result)
                                type_id = get_type_id()
                                # for sert_i in range(3):
                                #     oasst1_val, oasst1_keywords = insert_milvus_record(sert_i)
                                #     q_id, a_id = insert_result(oasst1_val, task, result, type_id)
                                #     insert_keywords_result(oasst1_keywords, task, result, type_id, q_id, a_id)
                                s_t = time.time()
                                print('insert_time', time.time()-s_t)
                                with open(_FILE_RESULT + '/' + file_task , 'a', encoding='utf-8') as file:
                                    file.write(str(result))
                                file.close()

                                if sub_send_ip != '192.168.2.26':
                                    print('需要转发任务' + file_task + ', 任务目标地:' + sub_send_ip)
                                    # send_file(_FILE_RESULT + '/' + file_task, sub_send_ip)
                                    if sub_send_ip == '192.168.2.8':
                                        send_file(_FILE_RESULT + '/' + file_task, '192.168.2.26')  
                                    else:  
                                        send_file(_FILE_RESULT + '/' + file_task, sub_send_ip)
                                # 补充转发任务
                                send_file(_FILE_RESULT + '/' + file_task, '192.168.2.6')
                                send_file(_FILE_RESULT + '/' + file_task, '192.168.2.5')
                                send_file(_FILE_RESULT + '/' + file_task, '192.168.2.7')
                                send_file(_FILE_RESULT + '/' + file_task, '192.168.2.26')

                                with open(_SUBTASK_TIME + '/' + file_task, 'a') as file:
                                    file.write(str(float(time.time())))
                                file.close()

                                # result save
                                with open(_SUBTASK_RESULT + '/' + file_task , 'a', encoding='utf-8') as file:
                                    file.write(str(result))
                                file.close()

                                send_host_file(_FILE_RESULT + '/' + file_task, '192.168.2.26')
                                send_host_file_time(_SUBTASK_TIME + '/' + file_task, '192.168.2.26')
                                mvfile_filename(path, path_des, file_task)



                            elif file_task[:-6] + '_' + str(int(file_task[-5])-1) + '.txt' in sub_lists:
                                print('jicheng file:', file_task, file_task[:-6] + '_' + str(int(file_task[-5])-1) + '.txt', sub_lists)
                                # 子任务组里的最后一个
                                if len(set(sub_lists) & set(local_pre)) != len(sub_lists):
                                    print('last_task', set(sub_lists), set(local_pre))
                                    continue
                                else:
                                    print('小子任务组里的最后一个')
                                    print('file:', file_task)
                                    tasks_content = []
                                    task = read_task(path + '/' + file_task)
                                    print('tasks:\n', task)
                                    for n in local_pre:
                                        if n in sub_lists:
                                            tasks_content.append(read_sequence(_FILE_RESULT + '/' + n))

                                    task = task + '\n' + 'The relevant information for the task is as follows:'
                                    for con in tasks_content:
                                        task = task + '\n' + con 
                                    if len(task)>1000:
                                        task = task[:1000]
                                        
                                    tasks = get_tasks(task)
                                    crew = Crew(
                                        agents=[agents.Comprehensive()],
                                        tasks=tasks,
                                        verbose=2,  # You can set it to 1 or 2 to different logging levels
                                        full_output=False
                                    )
                                    # while 1:
                                    # try:
                                    #     result = crew.kickoff()
                                    #     # break
                                    # except:
                                    #     crew = Crew(
                                    #     agents=[agents.Comprehensive()],
                                    #     tasks=tasks,
                                    #     verbose=2,  # You can set it to 1 or 2 to different logging levels
                                    #     full_output=False
                                    # )
                                    result = crew.kickoff()
                                    print("######################")
                                    print(result)
                                    type_id = get_type_id()
                                    # for sert_i in range(3):
                                    #     oasst1_val, oasst1_keywords = insert_milvus_record(sert_i)
                                    #     q_id, a_id = insert_result(oasst1_val, task, result, type_id)
                                    #     insert_keywords_result(oasst1_keywords, task, result, type_id, q_id, a_id)

                                    s_t = time.time()
                                    print('insert_time', time.time()-s_t)
                                    with open(_FILE_RESULT + '/' + file_task , 'a', encoding='utf-8') as file:
                                        file.write(str(result))
                                    file.close()

                                    if sendip != '192.168.2.26':
                                        print('需要转发任务' + file_task + ', 任务目标地:' + sendip)
                                        # send_file(_FILE_RESULT + '/' + file_task, sendip)
                                        if sendip == '192.168.2.8':
                                            send_file(_FILE_RESULT + '/' + file_task, '192.168.2.26')  
                                        else:  
                                            send_file(_FILE_RESULT + '/' + file_task, sendip)

                                    with open(_SUBTASK_TIME + '/' + file_task, 'a') as file:
                                        file.write(str(float(time.time())))
                                    file.close()

                                    # result save
                                    with open(_SUBTASK_RESULT + '/' + file_task , 'a', encoding='utf-8') as file:
                                        file.write(str(result))
                                    file.close()

                                    send_host_file(_FILE_RESULT + '/' + file_task, '192.168.2.26')
                                    send_host_file_time(_SUBTASK_TIME + '/' + file_task, '192.168.2.26')
                                    mvfile_filename(path, path_des, file_task)

                            else:
                                # 普通任务
                                print('normal file:', file_task, file_task[:-6] + '_' + str(int(file_task[-5])-1) + '.txt', sub_lists)
                                task = read_task(path + '/' + file_task)
                                if len(task)>1000:
                                    task = task[:1000]
                                print('tasks:\n', task)

                                # connections.disconnect("default")
                                # client = MilvusClient(
                                #     uri='http://192.168.2.7:19530',
                                #     token='root:Milvus'
                                # )
                                # connections.connect("default", host="192.168.2.7", port="19530")
                                # oasst1_val = Collection("crewai_agents1_algorithm1_subvec")
                                # text_id, text_sim, text, text_filed, text_emotion = search_data(oasst1_val, task)

                                # task = task + 'The reference information for the task is as follows:\n' + text[0] 


                                tasks = get_tasks(task)
                                crew = Crew(
                                    agents=[agents.Comprehensive()],
                                    tasks=tasks,
                                    verbose=2,  # You can set it to 1 or 2 to different logging levels
                                    full_output=False
                                )
                                # while 1:
                                # try:
                                #     result = crew.kickoff()
                                    
                                # except:
                                #     crew = Crew(
                                #     agents=[agents.Comprehensive_without()],
                                #     tasks=tasks,
                                #     verbose=2,  # You can set it to 1 or 2 to different logging levels
                                #     full_output=False
                                # )
                                result = crew.kickoff()
                                print("######################")
                                print(result)
                                type_id = get_type_id()
                                # for sert_i in range(3):
                                #     oasst1_val, oasst1_keywords = insert_milvus_record(sert_i)
                                #     q_id, a_id = insert_result(oasst1_val, task, result, type_id)
                                #     insert_keywords_result(oasst1_keywords, task, result, type_id, q_id, a_id)

                                s_t = time.time()
                                print('insert_time', time.time()-s_t)
                                with open(_FILE_RESULT + '/' + file_task , 'a', encoding='utf-8') as file:
                                    file.write(str(result))
                                file.close()

                                if sendip != '192.168.2.26':
                                    print('需要转发任务' + file_task + ', 任务目标地:' + sendip)
                                    # send_file(_FILE_RESULT + '/' + file_task, sendip)
                                    if sendip == '192.168.2.8':
                                        send_file(_FILE_RESULT + '/' + file_task, '192.168.2.26')  
                                    else:  
                                        send_file(_FILE_RESULT + '/' + file_task, sendip)
                                    

                                with open(_SUBTASK_TIME + '/' + file_task, 'a') as file:
                                    file.write(str(float(time.time())))
                                file.close()

                                # result save
                                with open(_SUBTASK_RESULT + '/' + file_task , 'a', encoding='utf-8') as file:
                                    file.write(str(result))
                                file.close()

                                send_host_file(_FILE_RESULT + '/' + file_task, '192.168.2.26')
                                send_host_file_time(_SUBTASK_TIME + '/' + file_task, '192.168.2.26')
                                mvfile_filename(path, path_des, file_task)
                
                # 在本节点处理最后一个任务
                elif file_task == last_file_name:
                    # 检查已经完成任务的数量是否满足执行任务列表规定的
                    print('last')
                    tasks_content = []
                    # num = 0
                    # 检查前置任务数量是否足够
                    local_pre = os.listdir(_FILE_RESULT + '/')
                    if len(set(tasks_pre[:-1]) & set(local_pre)) != len(tasks_pre[:-1]):
                        print('任务对比', set(tasks_pre[:-1]), set(local_pre))
                        continue
                    # for m in tasks_pre[:-1]:
                    #     if m in local_pre:
                    #         num = num + 1
                    # print(num, tasks_len -1)
                    # if num == tasks_len -1:
                    #     break
                    
                    for n in local_pre:
                        if file_task[:-6] in n:
                            tasks_content.append(read_sequence(_FILE_RESULT + '/' + n))

                    task = read_task(path + '/' + last_file_name)
                    task = task + '\n' + 'The relevant information for the task is as follows:'
                    for con in tasks_content:
                        task = task + '\n' + con
                    if len(task)>1000:
                        task = task[:1000]
                    print('tasks:\n', task)

                    # connections.disconnect("default")
                    # client = MilvusClient(
                    #     uri='http://192.168.2.7:19530',
                    #     token='root:Milvus'
                    # )
                    # connections.connect("default", host="192.168.2.7", port="19530")
                    # oasst1_val = Collection("crewai_agents1_algorithm1_subvec")
                    # text_id, text_sim, text, text_filed, text_emotion = search_data(oasst1_val, task)

                    # task = task + 'The reference information for the task is as follows:\n' + text[0]

                    tasks = get_tasks(task)
                    print('wanzhengtasks:\n', tasks)

                    crew = Crew(
                        agents=[agents.Comprehensive()],
                        tasks=tasks,
                        verbose=2,  # You can set it to 1 or 2 to different logging levels
                        full_output=False
                    )

                    # while 1:
                    # try:
                    #     result = crew.kickoff()
                    #     # break
                    # except:
                    #     crew = Crew(
                    #     agents=[agents.Comprehensive_without()],
                    #     tasks=tasks,
                    #     verbose=2,  # You can set it to 1 or 2 to different logging levels
                    #     full_output=False
                    # )
                    result = crew.kickoff()
                    print('result:', result)
                    # while True:
                    #     time.sleep(0.5)
                    #     try:
                    #         result = crew.kickoff()
                    #         break
                    #     except:
                    #         print('Error: TypeError: object of type NoneType has no len()')

                    type_id = get_type_id()
                    # for sert_i in range(3):
                    #     oasst1_val, oasst1_keywords = insert_milvus_record(sert_i)
                    #     q_id, a_id = insert_result(oasst1_val, task, result, type_id)
                    #     insert_keywords_result(oasst1_keywords, task, result, type_id, q_id, a_id)
                    # q_id, a_id = insert_result(oasst1_val, task, result, type_id)
                    # insert_keywords_result(task, result, type_id, q_id, a_id)

                    with open(_FILE_RESULT + '/' + file_task , 'a', encoding='utf-8') as file:
                        file.write(str(result))
                        file.close()

                    for root, dirs, files in os.walk(path):
                        # i是任务名
                        for i in files:
                            # 按照执行顺序执行任务
                            if 'task_send_number_' + file_task[:-6] in i:
                                st_time = read_sequence(path + '/' + 'task_send_number_' + file_task[:-6] + '.txt')
                    with open(_SUBTASK_TIME + '/' + file_task, 'a') as file:
                    # 写入内容
                        file.write(str(float(time.time())))
                        file.close()
                    # result save
                    with open(_SUBTASK_RESULT + '/' + file_task , 'a', encoding='utf-8') as file:
                        file.write(str(result))
                    
                    
                    if file_task not in sub_tasks_pre:      
                        send_host_answer(_FILE_RESULT + '/' + file_task, '192.168.2.26')
                        send_host_answer_time(_SUBTASK_TIME + '/' + file_task, '192.168.2.26')
                    mvfile_filename(path, path_des, file_task)
                    mvfile_filename(path, path_des, 'execution_sequence_' + last_file_name[:-6]+'.txt')
                    mvfile_filename(path, path_des, 'task_send_number_' + last_file_name[:-6]+'.txt')