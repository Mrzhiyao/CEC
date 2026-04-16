from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import time
# 使用默认数据库 ‘default’,也可以自己建数据库
connections.connect("default", host="192.168.2.5", port="19530")
from pymilvus import MilvusClient
client = MilvusClient(
    uri='http://192.168.2.5:19530'
)
# oasst1_val = Collection("two_input11")
# oasst1_val.release()
# oasst1_val = Collection("two_input_nodes11")
# oasst1_val.release()
client.drop_collection("crewai_agents1")
client.drop_collection("crewai_agents1_keywords")
