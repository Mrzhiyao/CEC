import json
import os
from embedding import embedding
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
from pymilvus import  MilvusClient
def insert_data(oasst1_val, question, answer,type_id):
    vector_q = embedding(question)
    vector_a = embedding(answer)
    record_created_date = get_current_time()
    qa_new_message_id = get_uuid()
    if len(answer) > 9000:
        record_answer = answer[:9000]
    else:
        record_answer = answer
    entities = [
        [qa_new_message_id],  # 信息id
        [record_created_date],  # field created_date
        [record_answer],  # 文本内容
        [vector_a],  # pro/assistant
        ['assistant'],
        [0],
        [str(0)],
        ['无'],
        [' '],
        [type_id]
    ]
    # insert_result =
    oasst1_val.insert(entities)
    if len(question) > 9000:
        record_question = question[:9000]
    else:
        record_question = question
    record_message_id = get_uuid()
    entities = [
        [record_message_id],  # 信息id
        [record_created_date],  # field created_date
        [record_question],  # 文本内容
        [vector_q],  # pro/assistant
        ['prompter'],
        [0],
        [str(0)],
        [record_answer],
        [qa_new_message_id],
        [type_id]
    ]
    # insert_result =
    oasst1_val.insert(entities)


if __name__ == '__main__':
    client = MilvusClient(
        uri='http://192.168.2.4:19530',
        token='root:Milvus'
    )
    connections.connect("default", host="192.168.2.4", port="19530")
    oasst1_val = Collection("lumos_100")

    with open('task_result.txt', 'r') as file:
        lines = file.readlines()
        result_list = []
        for line in lines:
            result_list.append(line.replace('\n', ''))
        print(len(result_list))
    # 逐行读取
    n = 0
    jsonl_file = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared_program', 'program', 'lumos_complex_qa_ground_onetime.jsonl'), 'r')
    # 逐行读取
    for line in jsonl_file:
        if n >= 100:
            break
        n = n + 1
        # 解码JSON
        json_obj = json.loads(line)
        print(n)
        final_task = ' '
        content = 'Task reference information and execution process:\n'
        if len(json_obj['messages']) > 1:
            for i in range(len(json_obj['messages'])):
                content = content + json_obj['messages'][i]['role'] + '\n' + json_obj['messages'][i]['content']
                if 'Task: ' in json_obj['messages'][i]['content']:
                    end = json_obj['messages'][i]['content'].find('Subgoal to be grounded: ')
                    final_task = json_obj['messages'][i]['content'][697:end]
                    print(final_task[1:].replace('\n', ''))
                    insert_data(oasst1_val, final_task[1:].replace('\n', ''), result_list[n-1], str(n-1))

# from pymilvus import  MilvusClient
#
# client = MilvusClient(
#     uri='http://10.20.1.15:30532',
#     token='root:Milvus'
# )
# connections.connect("default", host="10.20.1.15", port="30532")
