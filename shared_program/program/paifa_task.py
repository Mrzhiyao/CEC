from program.trans_result import task_get
from program.transform_qa_task import task_get_qa
import time
import os
from program.llm_send import send_file_llm
from program.node_send import send_file

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

def read_sequence(path):
    file = open(path, 'r')
    content = file.read()  # 所有内容保存在content中
    file.close()
    return content

def dis_task(task_number, sorted_task_list, action_carry, task_time, state_relation, task_content_list):
    jsonl_file = open('lumos_complex_qa_ground_onetime.jsonl', 'r')
    # 逐行读取
    n = 0
    for line in jsonl_file:
        # print(n, '第个任务')
        if n != task_number:
            n = n + 1
        else:
            task_list = task_get(line)
            f_s = 0
            task_list_new = []
            nt = 0

            task_list_new = []
            task_list = task_get_qa(line)
            nt = 0
            task_time = str(time.time())
            sub_pre_tasks = []
            for x in range(len(task_content_list)):
                tasks = task_content_list[x]
                task_list_new.append(tasks)
                file_name = task_time.replace('.', '') + '_' + str(nt)
                # 写入子任务到本地，以转发到边缘并行执行
                file = open('./subtasks/' + file_name + '.txt', 'a', encoding='utf-8')
                # 写入内容
                file.write(str(tasks))
                file.write('\n')
                # 关闭文件
                file.close()

                # 把连串任务的中最后的任务去掉，因为要统计到任务列表里
                judge_number = 0
                for sub_relation in state_relation:
                    if nt in sub_relation:
                        try:
                            sub_pre_tasks[judge_number].append(file_name)
                        except:
                            sub_pre_tasks.append([])
                            sub_pre_tasks[judge_number].append(file_name)
                        continue
                    judge_number = judge_number + 1
                nt = nt + 1

            path = './subtasks/'
            f_send = 0
            task_file = ''
            send_endpoint = ''
            # 判断是否往节点上发送了任务
            tasks_receive = [0, 0, 0, 0]
            start_time = time.time()
            for root, dirs, files in os.walk(path):
                for i in files:
                    if task_time.replace('.', '') in i:

                        # 有未合并子任务存在时才走这条路
                        if len(sub_pre_tasks) != 0:
                            for index_sub_pre_tasks in range(len(sub_pre_tasks)):
                                if i.replace('.txt','') in sub_pre_tasks[index_sub_pre_tasks] and i.replace('.txt','') != sub_pre_tasks[index_sub_pre_tasks][-1]:    
                                    task_file = i
                                    # 转发位
                                    content = path + i

                                    st = time.time()
                                    if action_carry[f_send] == 0:
                                        tasks_receive[0] = 1
                                        send_endpoint = '192.168.2.5'
                                        send_file(content, ip='192.168.2.5')
                                        file = open('./subtasks/sub_execution_sequence_' + str(task_file[:-6]) + '_' + str(index_sub_pre_tasks) + '.txt', 'a+')
                                        file.write(str(i))
                                        file.write(' 192.168.2.5')
                                        file.write('\n')
                                    elif action_carry[f_send] == 1:
                                        tasks_receive[1] = 1
                                        send_endpoint = '192.168.2.6'
                                        send_file(content, ip='192.168.2.6')
                                        file = open('./subtasks/sub_execution_sequence_' + str(task_file[:-6]) + '_' + str(index_sub_pre_tasks) + '.txt', 'a+')
                                        file.write(str(i))
                                        file.write(' 192.168.2.6')
                                        file.write('\n')
                                    elif action_carry[f_send] == 2:
                                        action_last = 2
                                        tasks_receive[2] = 1
                                        send_endpoint = '192.168.2.7'
                                        send_file(content, ip='192.168.2.7')
                                        file = open('./subtasks/sub_execution_sequence_' + str(task_file[:-6]) + '_' + str(index_sub_pre_tasks) + '.txt', 'a+')
                                        file.write(str(i))
                                        file.write(' 192.168.2.7')
                                        file.write('\n')
                                    elif action_carry[f_send] == 3:
                                        action_last = 3
                                        tasks_receive[3] = 1
                                        send_endpoint = '192.168.2.7'
                                        send_file_llm(content, ip='192.168.2.7')
                                        file = open('./subtasks/sub_execution_sequence_' + str(task_file[:-6]) + '_' + str(index_sub_pre_tasks) + '.txt', 'a+')
                                        file.write(str(i))
                                        file.write(' 192.168.2.26')
                                        file.write('\n')

                                    f_send = f_send + 1
                                else:
                                    task_file = i
                                    # 转发位
                                    content = path + i

                                    st = time.time()
                                    if action_carry[f_send] == 0:
                                        tasks_receive[0] = 1
                                        send_endpoint = '192.168.2.5'
                                        send_file(content, ip='192.168.2.5')
                                        file = open('./subtasks/execution_sequence_' + str(task_file[:-6]) + '.txt', 'a+')
                                        file.write(str(i))
                                        file.write(' 192.168.2.5')
                                        file.write('\n')
                                    elif action_carry[f_send] == 1:
                                        tasks_receive[1] = 1
                                        send_endpoint = '192.168.2.6'
                                        send_file(content, ip='192.168.2.6')
                                        file = open('./subtasks/execution_sequence_' + str(task_file[:-6]) + '.txt', 'a+')
                                        file.write(str(i))
                                        file.write(' 192.168.2.6')
                                        file.write('\n')
                                    elif action_carry[f_send] == 2:
                                        action_last = 2
                                        tasks_receive[2] = 1
                                        send_endpoint = '192.168.2.7'
                                        send_file(content, ip='192.168.2.7')
                                        file = open('./subtasks/execution_sequence_' + str(task_file[:-6]) + '.txt', 'a+')
                                        file.write(str(i))
                                        file.write(' 192.168.2.7')
                                        file.write('\n')
                                    elif action_carry[f_send] == 3:
                                        action_last = 3
                                        tasks_receive[3] = 1
                                        send_endpoint = '192.168.2.7'
                                        send_file_llm(content, ip='192.168.2.7')
                                        file = open('./subtasks/execution_sequence_' + str(task_file[:-6]) + '.txt', 'a+')
                                        file.write(str(i))
                                        file.write(' 192.168.2.26')
                                        file.write('\n')

                                    f_send = f_send + 1
                        else:
                            task_file = i
                            # 转发位
                            content = path + i

                            st = time.time()
                            if action_carry[f_send] == 0:
                                tasks_receive[0] = 1
                                send_endpoint = '192.168.2.5'
                                send_file(content, ip='192.168.2.5')
                                file = open('./subtasks/execution_sequence_' + str(task_file[:-6]) + '.txt', 'a+')
                                file.write(str(i))
                                file.write(' 192.168.2.5')
                                file.write('\n')
                            elif action_carry[f_send] == 1:
                                tasks_receive[1] = 1
                                send_endpoint = '192.168.2.6'
                                send_file(content, ip='192.168.2.6')
                                file = open('./subtasks/execution_sequence_' + str(task_file[:-6]) + '.txt', 'a+')
                                file.write(str(i))
                                file.write(' 192.168.2.6')
                                file.write('\n')
                            elif action_carry[f_send] == 2:
                                action_last = 2
                                tasks_receive[2] = 1
                                send_endpoint = '192.168.2.7'
                                send_file(content, ip='192.168.2.7')
                                file = open('./subtasks/execution_sequence_' + str(task_file[:-6]) + '.txt', 'a+')
                                file.write(str(i))
                                file.write(' 192.168.2.7')
                                file.write('\n')
                            elif action_carry[f_send] == 3:
                                action_last = 3
                                tasks_receive[3] = 1
                                send_endpoint = '192.168.2.7'
                                send_file_llm(content, ip='192.168.2.7')
                                file = open('./subtasks/execution_sequence_' + str(task_file[:-6]) + '.txt', 'a+')
                                file.write(str(i))
                                file.write(' 192.168.2.26')
                                file.write('\n')

                            f_send = f_send + 1
                
                if task_file != '':
                    # print('task_file',task_file)
                    file = open('./subtasks/task_send_number_' + str(task_file[:-6]) + '.txt', 'a')
                    file.write(str(start_time))
                    # 关闭文件
                    file.close()

                # 发送总任务数到执行末尾任务的节点上
                if send_endpoint != '':
                    send_file('./subtasks/task_send_number_' + str(task_file[:-6]) + '.txt', ip=send_endpoint)
                
                sub_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'subtasks')

                try:
                    content =  'sub_execution_sequence_' + str(task_file[:-6])
                    matching_files = find_files_containing_name(sub_directory, content)
                    if len(matching_files) != 0:
                        for sub_file in matching_files:
                            out = read_sequence(path + '/' + sub_file)
                            ppp_tasks_list = out.split('\n')
                            for content in ppp_tasks_list[:-1]:
                                if content.split(' ')[1] == '192.168.2.5':
                                    send_file('./subtasks/' +  sub_file, ip='192.168.2.5')
                                elif content.split(' ')[1] == '192.168.2.6':  
                                    send_file('./subtasks/' +  sub_file, ip='192.168.2.6')
                                elif content.split(' ')[1] == '192.168.2.7':  
                                    send_file('./subtasks/' +  sub_file, ip='192.168.2.7')
                                elif content.split(' ')[1] == '192.168.2.26':  
                                    send_file_llm('./subtasks/' +  sub_file, ip='192.168.2.7')

                    else:
                        pass
                except:
                    pass


                for k in range(4):
                    if tasks_receive[k] == 1 and k == 0:
                        try:
                            send_file('./subtasks/execution_sequence_' + str(task_file[:-6]) + '.txt', ip='192.168.2.5')
                            # time.sleep(2)
                        except:
                            pass


                    elif tasks_receive[k] == 1 and k == 1:
                        try:
                            send_file('./subtasks/execution_sequence_' + str(task_file[:-6]) + '.txt', ip='192.168.2.6')
                        except:
                            pass


                    elif tasks_receive[k] == 1 and k == 2:
                        try:
                            send_file('./subtasks/execution_sequence_' + str(task_file[:-6]) + '.txt', ip='192.168.2.7')
                        except:
                            pass


                    elif tasks_receive[k] == 1 and k == 3:
                        try:
                            send_file_llm('./subtasks/execution_sequence_' + str(task_file[:-6]) + '.txt', ip='192.168.2.7')
                        except:
                            pass


                if task_file != '':

                    file = open('./subtasks/task_send_content_time_' + str(task_file[:-6]) + '.txt', 'a', encoding='utf-8')
                    file.write(str(tasks))
                    file.write('\n')
                    file.write(str(task_file[:-6]))
                    file.write('\n')
                    file.close()
            time.sleep(50)

            jsonl_file.close()
            break
    return task_file
