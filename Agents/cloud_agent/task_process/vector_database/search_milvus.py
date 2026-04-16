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
from embedding import embedding
# from pymilvus import  MilvusClient
#
# client = MilvusClient(
#     uri='http://10.20.1.15:30532',
#     token='root:Milvus'
# )
# connections.connect("default", host="10.20.1.15", port="30532")
def search_data(oasst1_val, question):
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }
    search_params = {"metric_type": "IP", "limit": 3,
                     "params": {"ef": 250}
                     }

    result = oasst1_val.search([embedding(question)], "vector", search_params, limit=1,
                               output_fields=["id", "text", "role", "text_answer", "text_answerid"])
    # result = oasst1_val.search(search_vec, "title_vector", search_params, limit=1,
    #                            aaaaaaaaaaaaaaaaoutput_fields=["title_vector"])
    ids = []
    texts = []
    roles = []
    review_counts = []
    ranks = []
    text_answers = []
    text_answerids = []
    distances = []
    text_list = []
    for hits in result:
        for hit in hits:
            distance = hit.distance
            # print(distance,'distance')
            distances.append(distance)
            text = hit.entity.get('text')
            text_answer = hit.entity.get('text_answer')
            text_answerid = hit.entity.get('text_answerid')
            if hit.entity.get('role') == 'prompter':
                # print('text_answer',text_answer)
                return text_answer
            else:
                # print('text',text)
                return text