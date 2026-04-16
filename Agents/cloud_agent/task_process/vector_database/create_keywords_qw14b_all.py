import numpy


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
connections.disconnect("default")
connections.connect("default", host="192.168.2.5", port="19530")
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="message_id", dtype=DataType.VARCHAR, max_length=65000),
    FieldSchema(name="created_date", dtype=DataType.VARCHAR, max_length=65000),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65000),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="rank", dtype=DataType.VARCHAR, max_length=65000),
    FieldSchema(name="from_text", dtype=DataType.VARCHAR, max_length=65000),
    FieldSchema(name="from_text_id", dtype=DataType.VARCHAR, max_length=65000),
    FieldSchema(name="class_id", dtype=DataType.VARCHAR, max_length=65000),
]
schema = CollectionSchema(fields, "crewai_agents1_keywords_qw14b_all")
oasst1_val = Collection("crewai_agents1_keywords_qw14b_all", schema)
# [[1,2,3,4]]
import random

oasst1_val.flush()

index = {
    "index_type": "HNSW",
    "metric_type": "IP",
	"M": 48,
	"efConstruction": 500,
}
oasst1_val.create_index("vector", index)

oasst1_val.load()
#vectors_to_search = list_vec[0]
# search_params = {"metric_type": "IP","limit": 3,
#                  "params": {"ef":32,"range_filter":1.0,"radius":2.0}
# }

#result = oasst1_val.search(vectors_to_search, "vec", search_params, limit=3, output_fields=["title"])
#
#for hits in result:
#    for hit in hits:
#        print(f"hit: {hit}, random field: {hit.entity.get('text')}")

# oasst1_nodes
# oasst1_nodes
# oasst1_nodes