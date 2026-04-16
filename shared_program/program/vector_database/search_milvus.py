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

def search_data(oasst1_val, question):
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }
    search_params = {"metric_type": "IP", "limit": 3,
                     "params": {"ef": 250}
                     }

    result = oasst1_val.search([embedding(question)], "vector", search_params, limit=5,
                               output_fields=["message_id", "text", "role", "text_answer", "filed", "emotion", "text_answerid"])
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
    texts = []
    texts_sim = []
    texts_filed = []
    texts_emotion = []
    texts_id = []
    for hits in result:
        for hit in hits:
            distance = hit.distance
            # print(distance,'distance')
            distances.append(distance)
            text = hit.entity.get('text')
            text_id = hit.entity.get('message_id')
            text_answer = hit.entity.get('text_answer')
            text_filed = hit.entity.get('filed')
            text_emotion = hit.entity.get('emotion')
            text_answerid = hit.entity.get('text_answerid')

            texts_id.append(text_id)
            texts_filed.append(text_filed)
            texts_emotion.append(text_emotion)
            texts_sim.append(distance)
            if hit.entity.get('role') == 'prompter':
                # print('text_answer',text_answer)
                texts.append(text_answer)
                # return text_answer, text_filed, text_emotion
            else:
                # print('text',text)
                texts.append(text)
                # return text, text_filed, text_emotion
    return texts_id, texts_sim, texts, texts_filed, texts_emotion