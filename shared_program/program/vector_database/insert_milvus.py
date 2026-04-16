#
from program.embedding import embedding
import numpy
import time
from program.utils import get_uuid, get_current_time
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

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
