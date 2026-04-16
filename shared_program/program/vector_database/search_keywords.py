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


def search_keywords(message_id, client, question):
    collection_name = 'crewai_agents1_keywords_algorithm1_subvec'
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }
    search_params = {"metric_type": "IP", "limit": 3,
                     "params": {"ef": 250}
                     }
    filter_content = "from_text_id==" + "\'" + str(message_id) + "\'"
    result_sim = client.search(data=[embedding(question)], collection_name=collection_name,
                               filter=filter_content, output_fields=["id", "text", "vector", "rank"],
                               search_params=search_params,
                               limit=3)

    texts_score = []
    texts_id = []
    texts_sim = []
    for hits in result_sim:
        for hit in hits:
            # print(hit)
            distance = hit['distance']
            text_score = hit['entity']['rank']
            texts_sim.append(distance)
            texts_score.append(text_score)
    return texts_sim, texts_score
