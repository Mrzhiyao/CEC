import os
import numpy as np
np.random.seed(101)
# 生成包含0到2999的数组
numbers = np.arange(500)

# 打乱数组顺序
np.random.shuffle(numbers)

# 将打乱的数组分成3份
split1, split2, split3 = np.array_split(numbers, 3)

#
from embedding import embedding
from get_filed import get_filed
from get_emotion import get_emotion
import numpy
import time
from utils import get_uuid, get_current_time
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from read_task import read_task
from correct import get_newcontent
from insert_keywords_embedding import insert_keywords_result
from utils import get_type_id
from get_qa import get_qa_ques
from get_qa import get_qa_results


connections.disconnect("default")
connections.connect("default", host="YOUR_EDGE_NODE3_IP", port="19530")
# oasst1_val = Collection("crewai_agents1")
# oasst1_val = Collection("crewai_agents1_qw14b")
oasst1_val = Collection("crewai_agents1_algorithm1_subvec")
jsonl_file = open('os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared_program', 'program', 'lumos_complex_qa_ground_onetime.jsonl')', 'r')
# 逐行读取
n = 0
ques = []

ques_all = get_qa_ques()
answer_all = get_qa_results()

import random
random_numbers = []
for num in range(500):
    # print(num)
    if len(ques_all[num]) >= len(answer_all[num]):
        len_list = len(answer_all[num])
    else:
        len_list = len(ques_all[num]) 
    ques = ques_all[num]
    content = answer_all[num]
        
    random_list = [random.choice([0, 1, 2]) for _ in range(len_list)]
    # print(random_list)

    random_numbers.append(random_list)

for num in range(500):
    print(num)
    if len(ques_all[num]) >= len(answer_all[num]):
        len_list = len(answer_all[num])
    else:
        len_list = len(ques_all[num]) 
    ques = ques_all[num]
    content = answer_all[num]
        
    for i in range(len_list):
        if random_numbers[num][i] != 2:
            continue
        type_id = get_type_id()
        # print(i)
        # print(i,':','ques:\n', ques[i],'ans:\n', content[i])
        vector_q = embedding(ques[i])
        vector_a = embedding(content[i])
        record_created_date = get_current_time()
        qa_new_message_id = get_uuid()
        if len(content[i]) > 9000:
            record_answer = content[i][:9000]
        else:
            record_answer = content[i]
        entities = [
            [qa_new_message_id],  # 信息id
            [record_created_date],  # field created_date
            [record_answer],  # 文本内容
            [vector_a],  # pro/assistant
            ['assistant'],
            [str(get_filed(record_answer))],
            [str(get_emotion(record_answer))],
            ['无'],
            [' '],
            [type_id]
        ]
        # insert_result =
        oasst1_val.insert(entities)
        if len(ques[i]) > 9000:
            record_question = ques[i][:9000]
        else:
            record_question = ques[i]
        record_message_id = get_uuid()
        entities = [
            [record_message_id],  # 信息id
            [record_created_date],  # field created_date
            [record_question],  # 文本内容
            [vector_q],  # pro/assistant
            ['prompter'],
            [str(get_filed(record_question))],
            [str(get_emotion(record_question))],
            [record_answer],
            [qa_new_message_id],
            [type_id]
        ]
        # insert_result =
        oasst1_val.insert(entities)

        q_id = record_message_id
        a_id = qa_new_message_id

        insert_keywords_result(record_question, record_answer, type_id, q_id, a_id)