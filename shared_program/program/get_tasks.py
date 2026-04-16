from program.transform_without_result import task_get
from program.transform_qa_task import task_get_qa
from program.agents import Agents
from program.tasks import get_tasks
import time
from program.insert_embedding import insert_result
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
from program.node_send import send_file

from program.vector_database.search_milvus import search_data
from program.vector_database.get_filed import get_filed
from program.vector_database.get_emotion import get_emotion
from program.vector_database.get_keywords import get_keywords
from program.get_source import get_source
from sklearn.preprocessing import StandardScaler
import ast
import numpy as np
from program.vector_database.search_keywords import search_keywords
import re
import csv

def cross_entropy(p, q):
    # 添加一个小的常数来防止log(0)
    epsilon = 1e-12
    q = np.clip(q, epsilon, 1. - epsilon)
    return -np.sum(p * np.log(q))

def ip_vec(a, b):
    norm = np.linalg.norm(a)
    if norm == 0:
        a_normalized =  a
    else:
        a_normalized = a / norm

    norm = np.linalg.norm(b)
    if norm == 0:
        b_normalized =  b
    else:
        b_normalized =  b / norm

    return np.dot(a_normalized, b_normalized)


def get_task_reset():
    jsonl_file = open('lumos_complex_qa_ground_onetime.jsonl', 'r')
    # 逐行读取
    n = 0
    agents = Agents()
    # 测试环境
    for line in jsonl_file:
        if n >= 1:
            break
        n = n + 1
        task_list = task_get(line)
        sub_number = 0
        states = []
        task_arrays = []


        # qa task
        task_opt = []
        f_s = 0
        task_list_new = []
        task_list = task_get_qa(line)
        for x in range(len(task_list)):
            if 'QA' in task_list[x] or 'Calculator' in task_list[x] or 'QA' in task_list[x]:
                task_mid = []
                tasks = 'This is a progressive task:\n'
                for y in range(f_s, x+1):
                    tasks = tasks + task_list[y]
                    task_mid.append(task_list[y])
                f_s = x + 1
                task_list_new.append(tasks)
                task_opt.append(task_mid)

        # 优化规划
        _res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'node_receive_resource')
        path1 = os.path.join(_res_dir, 'node1', 'resource.txt')
        path2 = os.path.join(_res_dir, 'node2', 'resource.txt')
        path3 = os.path.join(_res_dir, 'node3', 'resource.txt')

        o_number = 0
        while 1:
            try:
                x1 = get_source(path1)
                x2 = get_source(path2)
                x3 = get_source(path3)
                break
            except:
                pass
        if float(x1[0])<70:
            o_number = o_number + 1
        if float(x2[0])<70:
            o_number = o_number + 1
        if float(x3[0])<70:
            o_number = o_number + 1
            
        len_task = len(task_opt) - 1
        
        task_index_list = []
        if o_number-len_task>0:
            task_number = 0
            max_opt_task_length = o_number-len_task
            indexx = 0
            for sub_task_list in task_opt[:-1]:
                # 判断检查并保持任务的流程数有没有达到上限
                if task_number >= max_opt_task_length:
                    break
                # 主机数
                sim_num = 0
                if len(sub_task_list) != 1:
                    for i in range(3):
                        if i == 0:
                            connections.disconnect("default")
                            client = MilvusClient(
                                uri='http://192.168.2.5:19530',
                                token='root:Milvus'
                            )
                            connections.connect("default", host="192.168.2.5", port="19530")
                            oasst1_val = Collection("crewai_agents1_algorithm1_subvec")
                        elif i == 1:
                            connections.disconnect("default")
                            client = MilvusClient(
                                uri='http://192.168.2.6:19530',
                                token='root:Milvus'
                            )
                            connections.connect("default", host="192.168.2.6", port="19530")
                            oasst1_val = Collection("crewai_agents1_algorithm1_subvec")
                        else:
                            connections.disconnect("default")
                            client = MilvusClient(
                                uri='http://192.168.2.7:19530',
                                token='root:Milvus'
                            )
                            connections.connect("default", host="192.168.2.7", port="19530")
                            oasst1_val = Collection("crewai_agents1_algorithm1_subvec")

                        sub_task_kw = sub_task_list[0].replace(" = ParagraphRetrieve", "")
                        sub_task_kw = sub_task_kw.replace(" = QA", "")
                        sub_task_kw = sub_task_kw.replace(" = Calculator", "")
                        sub_task_kw = sub_task_kw.replace(" = Code", "")
                        for i in range(15):
                            cont = 'R' + str(i)
                            sub_task_kw = sub_task_kw.replace(cont, "")
                        sub_task_kw = sub_task_kw.replace("Query", "")
                        sub_task_kw = sub_task_kw.replace("Question", "")
                        sub_task_kw = sub_task_kw.replace(": ", "")
                        sub_task_kw = sub_task_kw.replace(": ", "")
                        sub_task_kw = sub_task_kw.replace("[],", "")
                        sub_task_kw = sub_task_kw.replace("(,", "")
                        sub_task_kw = sub_task_kw.replace("([,", "")
                        sub_task_kw = sub_task_kw.replace(")", "")
                        sub_task_kw = sub_task_kw.replace("(", "")
                        sub_task_kw = sub_task_kw.replace("[", "")
                        sub_task_kw = sub_task_kw.replace("]", "")
                        sub_task_kw = sub_task_kw.replace("],", "")
                        sub_task_kw = sub_task_kw.replace("[,", "")
                        sub_task = sub_task_kw

                        text_id, text_sim, text, text_filed, text_emotion = search_data(oasst1_val, sub_task)
                        for i in range(1):
                            if text_sim[i] < 0.6:
                                sim_num = sim_num + 1
                        
                        if sim_num == 3:
                            task_number = task_number + 1
                            task_index_list.append(indexx)

                indexx = indexx + 1
        task_out_opt = []
        record_list = []
        for insert_i in range(len(task_opt)):
            
            if insert_i in task_index_list:
                sub_record = []
                for cont in task_opt[insert_i]:
                    task_out_opt.append(cont)
                    # 多-1是为了不包括这个最后的任务
                    sub_record.append(len(task_out_opt)-1)
                record_list.append(sub_record)
            else:
                if len(task_opt[insert_i]) == 1:
                    task_out_opt.append(task_opt[insert_i][0])
                else:
                    insert_tasks = 'This is a progressive task:\n'
                    for cont in task_opt[insert_i]:
                        insert_tasks = insert_tasks + cont
                    task_out_opt.append(insert_tasks)




        task_shunxu = 0
        for sub_task in task_out_opt:
            # 当前任务类型
            state = []

            # 获取资源
            path1 = os.path.join(_res_dir, 'node1', 'resource.txt')
            path2 = os.path.join(_res_dir, 'node2', 'resource.txt')
            path3 = os.path.join(_res_dir, 'node3', 'resource.txt')
            while 1:
                try:
                    for x in get_source(path1):
                        state.append(x)
                    for x in get_source(path2):
                        state.append(x)
                    for x in get_source(path3):
                        state.append(x)
                    break
                except:
                    pass
                    state = []
            task_type = [0,0,0,0,0]
            # 任务属性
            if "KnowledgeQuery" in sub_task:
                task_type[0] = 1
            if "ParagraphRetrieve" in sub_task:
                task_type[1] = 1
            if "QA" in sub_task:
                task_type[2] = 1
            if "Calculator" in sub_task:
                task_type[3] = 1
            if "Code(pseudo_code)" in sub_task:
                task_type[4] = 1
            
            for one_task_type in task_type:
                state.append(one_task_type)

            # 任务长度
            state.append(len(task_list_new))
            # 当前任务所处的位置
            state.append(task_shunxu)
            # 首、中间、末尾0 1 2
            if task_shunxu == 0:
                state.append(0)
            elif task_shunxu == len(task_list_new) -1:
                state.append(2)
            else:
                state.append(1)
            # 最近前置任务的调度位置, 默认为-1 
            state.append(-1)

            # 主机数
            for i in range(3):
                if i == 0:
                    connections.disconnect("default")
                    client = MilvusClient(
                        uri='http://192.168.2.5:19530',
                        token='root:Milvus'
                    )
                    connections.connect("default", host="192.168.2.5", port="19530")
                    oasst1_val = Collection("crewai_agents1_algorithm1_subvec")
                    oasst1_keywords = Collection("crewai_agents1_keywords_algorithm1_subvec")
                elif i == 1:
                    connections.disconnect("default")
                    client = MilvusClient(
                        uri='http://192.168.2.6:19530',
                        token='root:Milvus'
                    )
                    connections.connect("default", host="192.168.2.6", port="19530")
                    oasst1_val = Collection("crewai_agents1_algorithm1_subvec")
                    oasst1_keywords = Collection("crewai_agents1_keywords_algorithm1_subvec")
                else:
                    connections.disconnect("default")
                    client = MilvusClient(
                        uri='http://192.168.2.7:19530',
                        token='root:Milvus'
                    )
                    connections.connect("default", host="192.168.2.7", port="19530")
                    oasst1_val = Collection("crewai_agents1_algorithm1_subvec")
                    oasst1_keywords = Collection("crewai_agents1_keywords_algorithm1_subvec")

                sub_task_kw = sub_task.replace(" = ParagraphRetrieve", "")
                sub_task_kw = sub_task_kw.replace(" = QA", "")
                sub_task_kw = sub_task_kw.replace(" = Calculator", "")
                sub_task_kw = sub_task_kw.replace(" = Code", "")
                for i in range(15):
                    cont = 'R' + str(i)
                    sub_task_kw = sub_task_kw.replace(cont, "")
                sub_task_kw = sub_task_kw.replace("Query", "")
                sub_task_kw = sub_task_kw.replace("Question", "")
                sub_task_kw = sub_task_kw.replace(": ", "")
                sub_task_kw = sub_task_kw.replace(": ", "")
                sub_task_kw = sub_task_kw.replace("[],", "")
                sub_task_kw = sub_task_kw.replace("(,", "")
                sub_task_kw = sub_task_kw.replace("([,", "")
                sub_task_kw = sub_task_kw.replace(")", "")
                sub_task_kw = sub_task_kw.replace("(", "")
                sub_task_kw = sub_task_kw.replace("[", "")
                sub_task_kw = sub_task_kw.replace("]", "")
                sub_task_kw = sub_task_kw.replace("],", "")
                sub_task_kw = sub_task_kw.replace("[,", "")
                sub_task = sub_task_kw
                ques_filed = get_filed(sub_task)
                ques_emotion = get_emotion(sub_task)
                ques_keywords_ori = get_keywords(sub_task)
                # find keywords
                ques_keywords = []
                ques_keywords_score = []
                for score, key_text in ques_keywords_ori:
                    if key_text != 'knowledge' and key_text != 'Reference documents' and key_text != 'information' and key_text != 'r1'and key_text != 'r2'and key_text != 'r3'and key_text != 'r4'and key_text != 'r5'and key_text != 'r6'and key_text != 'r7'and key_text != 'r8'and key_text != 'r8'and key_text != 'r9'and key_text != 'r10'and key_text != 'r11'and key_text != 'r12'and key_text != 'r13'and key_text != 'r14' and key_text != ' = 'and key_text != '=':
                        ques_keywords_score.append(score)
                        ques_keywords.append(key_text)
                # 最多只取前三个
                if len(ques_keywords) > 3:
                    ques_keywords = ques_keywords[0:3]
                    ques_keywords_score = ques_keywords_score[0:3]
                # knowledge
                text_id, text_sim, text, text_filed, text_emotion = search_data(oasst1_val, sub_task)
                for i in range(5):
                    state.append(text_sim[i])
                    np.array(ast.literal_eval(text_filed[i]))
                    sim_filed = ip_vec(ques_filed, np.array(ast.literal_eval(text_filed[i])))
                    state.append(sim_filed)
                    sim_emotion = ip_vec(ques_emotion, np.array(ast.literal_eval(text_emotion[i])))
                    state.append(sim_emotion)
                    # keyword:
                    state_keywords_score = 0
                    for j in range(len(ques_keywords_score)):
                        texts_sim, texts_score = search_keywords(text_id[i], client, ques_keywords[j])
                        midd = 0
                        for k in range(len(texts_sim)):
                            midd = midd + ques_keywords_score[j]*texts_sim[k]*float(texts_score[k])
                        if len(texts_sim) == 0:
                            midd = 0
                        else:
                            midd = midd / len(texts_sim)
                        state_keywords_score = state_keywords_score + midd
                    state_keywords_score = state_keywords_score / len(ques_keywords_score)
                    state.append(state_keywords_score)


            state1 = state[:9]
            state2 = state[9:18]
            state3 = state[18:]
            scaler_1 = StandardScaler()
            scaler_2 = StandardScaler()
            scaler_3 = StandardScaler()
            standardized_state1 = np.array([state1]).T
            standardized_state2 = np.array([state2]).T
            standardized_state3 = np.array([state3]).T
            sub_number = sub_number + 1
            new_state = []
            for s in standardized_state1:
                new_state.append(s)
            for s in standardized_state2:
                new_state.append(s)
            for s in standardized_state3:
                new_state.append(s)
            states.append(new_state)

            task_shunxu = task_shunxu + 1

        jsonl_file.close()
        return states, record_list, task_out_opt

def get_task(input_number):
    jsonl_file = open('lumos_complex_qa_ground_onetime.jsonl', 'r')
    # 逐行读取
    n = 0
    agents = Agents()
    # 测试环境
    for line in jsonl_file:
        if n != input_number:
            n = n + 1
            continue

        task_list = task_get(line)
        sub_number = 0
        states = []
        task_arrays = []

        # qa task
        task_opt = []
        f_s = 0
        task_list_new = []
        task_list = task_get_qa(line)
        for x in range(len(task_list)):
            if 'QA' in task_list[x] or 'Calculator' in task_list[x] or 'QA' in task_list[x]:
                task_mid = []
                tasks = 'This is a progressive task:\n'
                for y in range(f_s, x+1):
                    tasks = tasks + task_list[y]
                    task_mid.append(task_list[y])
                f_s = x + 1
                task_list_new.append(tasks)
                task_opt.append(task_mid)

        # 优化规划
        _res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'node_receive_resource')
        path1 = os.path.join(_res_dir, 'node1', 'resource.txt')
        path2 = os.path.join(_res_dir, 'node2', 'resource.txt')
        path3 = os.path.join(_res_dir, 'node3', 'resource.txt')

        o_number = 0
        while 1:
            try:
                x1 = get_source(path1)
                x2 = get_source(path2)
                x3 = get_source(path3)
                break
            except:
                pass
        if float(x1[0])<70:
            o_number = o_number + 1
        if float(x2[0])<70:
            o_number = o_number + 1
        if float(x3[0])<70:
            o_number = o_number + 1
            
        len_task = len(task_opt) - 1
        
        task_index_list = []
        if o_number-len_task>0:
            task_number = 0
            max_opt_task_length = o_number-len_task
            indexx = 0
            for sub_task_list in task_opt[:-1]:
                # 判断检查并保持任务的流程数有没有达到上限
                if task_number >= max_opt_task_length:
                    break
                # 主机数
                sim_num = 0
                if len(sub_task_list) != 1:
                    for i in range(3):
                        if i == 0:
                            connections.disconnect("default")
                            client = MilvusClient(
                                uri='http://192.168.2.5:19530',
                                token='root:Milvus'
                            )
                            connections.connect("default", host="192.168.2.5", port="19530")
                            oasst1_val = Collection("crewai_agents1_algorithm1_subvec")
                        elif i == 1:
                            connections.disconnect("default")
                            client = MilvusClient(
                                uri='http://192.168.2.6:19530',
                                token='root:Milvus'
                            )
                            connections.connect("default", host="192.168.2.6", port="19530")
                            oasst1_val = Collection("crewai_agents1_algorithm1_subvec")
                        else:
                            connections.disconnect("default")
                            client = MilvusClient(
                                uri='http://192.168.2.7:19530',
                                token='root:Milvus'
                            )
                            connections.connect("default", host="192.168.2.7", port="19530")
                            oasst1_val = Collection("crewai_agents1_algorithm1_subvec")

                        sub_task_kw = sub_task_list[0].replace(" = ParagraphRetrieve", "")
                        sub_task_kw = sub_task_kw.replace(" = QA", "")
                        sub_task_kw = sub_task_kw.replace(" = Calculator", "")
                        sub_task_kw = sub_task_kw.replace(" = Code", "")
                        for i in range(15):
                            cont = 'R' + str(i)
                            sub_task_kw = sub_task_kw.replace(cont, "")
                        sub_task_kw = sub_task_kw.replace("Query", "")
                        sub_task_kw = sub_task_kw.replace("Question", "")
                        sub_task_kw = sub_task_kw.replace(": ", "")
                        sub_task_kw = sub_task_kw.replace(": ", "")
                        sub_task_kw = sub_task_kw.replace("[],", "")
                        sub_task_kw = sub_task_kw.replace("(,", "")
                        sub_task_kw = sub_task_kw.replace("([,", "")
                        sub_task_kw = sub_task_kw.replace(")", "")
                        sub_task_kw = sub_task_kw.replace("(", "")
                        sub_task_kw = sub_task_kw.replace("[", "")
                        sub_task_kw = sub_task_kw.replace("]", "")
                        sub_task_kw = sub_task_kw.replace("],", "")
                        sub_task_kw = sub_task_kw.replace("[,", "")
                        sub_task = sub_task_kw

                        text_id, text_sim, text, text_filed, text_emotion = search_data(oasst1_val, sub_task)
                        for i in range(1):
                            if text_sim[i] < 0.6:
                                sim_num = sim_num + 1
                        
                        if sim_num == 3:
                            task_number = task_number + 1
                            task_index_list.append(indexx)

                indexx = indexx + 1
        task_out_opt = []
        record_list = []
        for insert_i in range(len(task_opt)):
            
            if insert_i in task_index_list:
                sub_record = []
                sub_length = 0
                for cont in task_opt[insert_i]:
                    task_out_opt.append(cont)
                    sub_record.append(len(task_out_opt)-1)
                record_list.append(sub_record)
            else:
                if len(task_opt[insert_i]) == 1:
                    task_out_opt.append(task_opt[insert_i][0])
                else:
                    insert_tasks = 'This is a progressive task:\n'
                    for cont in task_opt[insert_i]:
                        insert_tasks = insert_tasks + cont
                    task_out_opt.append(insert_tasks)




        task_shunxu = 0
        for sub_task in task_out_opt:
            # 当前任务类型
            state = []

            # 获取资源
            path1 = os.path.join(_res_dir, 'node1', 'resource.txt')
            path2 = os.path.join(_res_dir, 'node2', 'resource.txt')
            path3 = os.path.join(_res_dir, 'node3', 'resource.txt')
            while 1:
                try:
                    for x in get_source(path1):
                        state.append(x)
                    for x in get_source(path2):
                        state.append(x)
                    for x in get_source(path3):
                        state.append(x)
                    break
                except:
                    pass
                    state = []



            task_type = [0,0,0,0,0]
            # 任务属性
            if "KnowledgeQuery" in sub_task:
                task_type[0] = 1
            if "ParagraphRetrieve" in sub_task:
                task_type[1] = 1
            if "QA" in sub_task:
                task_type[2] = 1
            if "Calculator" in sub_task:
                task_type[3] = 1
            if "Code(pseudo_code)" in sub_task:
                task_type[4] = 1
            
            for one_task_type in task_type:
                state.append(one_task_type)

            # 任务长度
            state.append(len(task_list_new))
            # 当前任务所处的位置
            state.append(task_shunxu)
            # 首、中间、末尾0 1 2
            if task_shunxu == 0:
                state.append(0)
            elif task_shunxu == len(task_list_new) -1:
                state.append(2)
            else:
                state.append(1)
            # 最近前置任务的调度位置, 默认为-1 
            state.append(-1)


            # 主机数
            for i in range(3):
                if i == 0:
                    connections.disconnect("default")
                    client = MilvusClient(
                        uri='http://192.168.2.5:19530',
                        token='root:Milvus'
                    )
                    connections.connect("default", host="192.168.2.5", port="19530")
                    oasst1_val = Collection("crewai_agents1_algorithm1_subvec")
                    oasst1_keywords = Collection("crewai_agents1_keywords_algorithm1_subvec")
                elif i == 1:
                    connections.disconnect("default")
                    client = MilvusClient(
                        uri='http://192.168.2.6:19530',
                        token='root:Milvus'
                    )
                    connections.connect("default", host="192.168.2.6", port="19530")
                    oasst1_val = Collection("crewai_agents1_algorithm1_subvec")
                    oasst1_keywords = Collection("crewai_agents1_keywords_algorithm1_subvec")
                else:
                    connections.disconnect("default")
                    client = MilvusClient(
                        uri='http://192.168.2.7:19530',
                        token='root:Milvus'
                    )
                    connections.connect("default", host="192.168.2.7", port="19530")
                    oasst1_val = Collection("crewai_agents1_algorithm1_subvec")
                    oasst1_keywords = Collection("crewai_agents1_keywords_algorithm1_subvec")

                sub_task_kw = sub_task.replace(" = ParagraphRetrieve", "")
                sub_task_kw = sub_task_kw.replace(" = QA", "")
                sub_task_kw = sub_task_kw.replace(" = Calculator", "")
                sub_task_kw = sub_task_kw.replace(" = Code", "")
                for i in range(15):
                    cont = 'R' + str(i)
                    sub_task_kw = sub_task_kw.replace(cont, "")
                sub_task_kw = sub_task_kw.replace("Query", "")
                sub_task_kw = sub_task_kw.replace("Question", "")
                sub_task_kw = sub_task_kw.replace(": ", "")
                sub_task_kw = sub_task_kw.replace(": ", "")
                sub_task_kw = sub_task_kw.replace("[],", "")
                sub_task_kw = sub_task_kw.replace("(,", "")
                sub_task_kw = sub_task_kw.replace("([,", "")
                sub_task_kw = sub_task_kw.replace(")", "")
                sub_task_kw = sub_task_kw.replace("(", "")
                sub_task_kw = sub_task_kw.replace("[", "")
                sub_task_kw = sub_task_kw.replace("]", "")
                sub_task_kw = sub_task_kw.replace("],", "")
                sub_task_kw = sub_task_kw.replace("[,", "")
                sub_task = sub_task_kw
                ques_filed = get_filed(sub_task)
                ques_emotion = get_emotion(sub_task)
                ques_keywords_ori = get_keywords(sub_task)
                # find keywords
                ques_keywords = []
                ques_keywords_score = []
                for score, key_text in ques_keywords_ori:
                    if key_text != 'knowledge' and key_text != 'Reference documents' and key_text != 'information' and key_text != 'r1'and key_text != 'r2'and key_text != 'r3'and key_text != 'r4'and key_text != 'r5'and key_text != 'r6'and key_text != 'r7'and key_text != 'r8'and key_text != 'r8'and key_text != 'r9'and key_text != 'r10'and key_text != 'r11'and key_text != 'r12'and key_text != 'r13'and key_text != 'r14' and key_text != ' = 'and key_text != '=':
                        ques_keywords_score.append(score)
                        ques_keywords.append(key_text)
                # 最多只取前三个
                if len(ques_keywords) > 3:
                    ques_keywords = ques_keywords[0:3]
                    ques_keywords_score = ques_keywords_score[0:3]
                # knowledge
                text_id, text_sim, text, text_filed, text_emotion = search_data(oasst1_val, sub_task)
                for i in range(5):
                    state.append(text_sim[i])
                    np.array(ast.literal_eval(text_filed[i]))
                    sim_filed = ip_vec(ques_filed, np.array(ast.literal_eval(text_filed[i])))
                    state.append(sim_filed)
                    sim_emotion = ip_vec(ques_emotion, np.array(ast.literal_eval(text_emotion[i])))
                    state.append(sim_emotion)
                    # keyword:
                    state_keywords_score = 0
                    for j in range(len(ques_keywords_score)):
                        texts_sim, texts_score = search_keywords(text_id[i], client, ques_keywords[j])
                        midd = 0
                        for k in range(len(texts_sim)):
                            midd = midd + ques_keywords_score[j]*texts_sim[k]*float(texts_score[k])
                        if len(texts_sim) == 0:
                            midd = 0
                        else:
                            midd = midd / len(texts_sim)
                        state_keywords_score = state_keywords_score + midd
                    if len(ques_keywords_score) == 0:
                        state_keywords_score = 0.5
                    else:
                        state_keywords_score = state_keywords_score / len(ques_keywords_score)
                    state.append(state_keywords_score)


            state1 = state[:9]
            state2 = state[9:18]
            state3 = state[18:]
            scaler_1 = StandardScaler()
            scaler_2 = StandardScaler()
            scaler_3 = StandardScaler()
            standardized_state1 = np.array([state1]).T
            standardized_state2 = np.array([state2]).T
            standardized_state3 = np.array([state3]).T
            sub_number = sub_number + 1
            new_state = []
            for s in standardized_state1:
                new_state.append(s)
            for s in standardized_state2:
                new_state.append(s)
            for s in standardized_state3:
                new_state.append(s)
            states.append(new_state)

            task_shunxu =  task_shunxu + 1
        jsonl_file.close()

        return states, record_list, task_out_opt


