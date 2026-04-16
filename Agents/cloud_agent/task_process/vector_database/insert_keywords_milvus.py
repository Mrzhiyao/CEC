#
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

def insert_keywords_data(oasst1_val, words, words_rank, source, source_id, type_id):
    words = words.replace("(","")
    words = words.replace(")","")
    words = words.replace("（","")
    words = words.replace("）","")
    words = words.replace(",","")
    words = words.replace(";","")
    words = words.replace(".","")
    words = words.replace("?","")
    # print('end',words)
    vector_s = embedding(words)
    # vector_a = embedding(answer)
#     id message_id created_date text vector from_text from_text_id class_id
    record_created_date = get_current_time()
    if len(source) > 9000:
        source = source[:9000]

    entities = [
        [get_uuid()],  # 信息id
        [record_created_date],  # field created_date
        [words],  # 文本内容
        [vector_s],  # pro/assistant
        [str(words_rank)],  # releated ranked
        [source],
        [source_id],
        [type_id]
    ]
    # insert_result =
    # print(words, str(words_rank), source,source_id )
    oasst1_val.insert(entities)

