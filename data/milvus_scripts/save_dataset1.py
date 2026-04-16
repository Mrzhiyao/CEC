# 445812018542381317

from pymilvus import MilvusClient
import time
import numpy as np

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
connections.connect("default", host="192.168.2.26", port="19530")
oasst1_val = Collection("crewai_agents1_algorithm1_insert")
client = MilvusClient(
    uri='http://192.168.2.26:19530',
    token='root:Milvus'
)
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10},
}
# 447307393318022778
result = oasst1_val.query("id>1", limit=10000, output_fields=["id", "message_id", "created_date", "text", "vector", "role", "filed","emotion", "text_answer", "text_answerid","class_id"])
record = []
linshi = []
linshi_id = []
linshi_message_id = []
linshi_created_date = []
linshi_text = []
linshi_vec = []
linshi_role = []
linshi_filed = []
linshi_emotion = []
linshi_text_answer = []
linshi_text_answerid = []
linshi_class_id = []


num = 0
for i in result:
    linshi_id.append(i['id'])
    linshi_message_id.append(i['message_id'])
    linshi_created_date.append(i['created_date'])
    linshi_text.append(i['text'])
    linshi_vec.append(i['vector'])
    linshi_role.append(i['role'])
    linshi_filed.append(i['filed'])
    linshi_emotion.append(i['emotion'])
    linshi_text_answer.append(i['text_answer'])
    linshi_text_answerid.append(i['text_answerid'])
    linshi_class_id.append(i['class_id'])
    print(i['id'])
    num = num +1
    print(num)
    # record.append(linshi)

np.savez('lumos_id10000.npz', list1_name=linshi_id)
np.savez('lumos_message_id10000.npz', list1_name=linshi_message_id)
np.savez('lumos_created_date10000.npz', list1_name=linshi_created_date)
np.savez('lumos_text10000.npz', list1_name=linshi_text)
np.savez('lumos_vec10000.npz', list1_name=linshi_vec)

np.savez('lumos_role10000.npz', list1_name=linshi_role)
np.savez('lumos_filed10000.npz', list1_name=linshi_filed)
np.savez('lumos_emotion10000.npz', list1_name=linshi_emotion)
np.savez('lumos_text_answer10000.npz', list1_name=linshi_text_answer)
np.savez('lumos_text_answerid10000.npz', list1_name=linshi_text_answerid)
np.savez('lumos_class_id10000.npz', list1_name=linshi_class_id)

