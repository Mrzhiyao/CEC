import os
import numpy as np
np.random.seed(101)
# 生成包含0到2999的数组
numbers = np.arange(500)

# 打乱数组顺序
np.random.shuffle(numbers)

# 将打乱的数组分成3份
split1, split2, split3 = np.array_split(numbers, 3)

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

connections.disconnect("default")
connections.connect("default", host="YOUR_EDGE_NODE2_IP", port="19530")
# oasst1_val = Collection("crewai_agents1")
# oasst1_val = Collection("crewai_agents1_qw14b")
oasst1_val = Collection("crewai_agents1_algorithm1_insert")
oasst1_val2 = Collection("crewai_agents1_keywords_algorithm1_insert")
jsonl_file = open('os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared_program', 'program', 'lumos_complex_qa_ground_onetime.jsonl')', 'r')
# 逐行读取
n = 0
ques = []
for line in jsonl_file:
    position_s = line.find('Task:')
    position_e = line.find('Subgoals')
    # print(line[position_s+6:position_e-2])
    ques.append(line[position_s+6:position_e-2])
    # print(n, '第个任务')

content = get_newcontent()
content[-1] = 'Based on the available information, there is no mention of any siblings for Alexander Korda in the given context. To determine if he had any siblings or not, one would need to consult other reliable sources, such as biographies, family records, or official documents that provide detailed genealogical information.'
# print(len(content))

# e1 = []
# e2 = []
# e3 = []
# e4 = []
# e5 = []
# e6 = []
# e7 = []
# e8 = []
import random
a = 0  # 下限
b = 500  # 上限
random.seed(2)
s2 = [random.randint(a, b) for _ in range(10)]

# for i in range(1000):
for i in s2:
    type_id = get_type_id()
    print(i)
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

    insert_keywords_result(oasst1_val2, record_question, record_answer, type_id, q_id, a_id)