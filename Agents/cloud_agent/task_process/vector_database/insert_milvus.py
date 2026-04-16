#
from embedding import embedding
from text_character.get_filed import get_filed
from text_character.get_emotion import get_emotion
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
# from pymilvus import  MilvusClient
#
# client = MilvusClient(
#     uri='http://10.20.1.15:30532',
#     token='root:Milvus'
# )
# connections.connect("default", host="10.20.1.15", port="30532")
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
        [str(get_filed(record_answer))],
        [str(get_emotion(record_answer))],
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
    return q_id, a_id
