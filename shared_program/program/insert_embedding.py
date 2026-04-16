import json
# KnowledgeQuery(query): Capture the relevant webpages according to the query;
# ParagraphRetrieve(context, query): Given a query, retrieve the most relevant paragraphs from the given context;
# QA(context, query): Given context, answer the given query;
# Calculator(formula): Calculate the input formula;
# Code(pseudo_code): Generate a Python function that corresponds to the pseudo code.
# 假设jsonl_file是一个包含JSONL数据的文件对象
from langchain.agents import Tool
from langchain.utilities import GoogleSerperAPIWrapper
from program.vector_database.insert_milvus import insert_data
from program.utils import get_type_id

def insert_result(oasst1_val, content):
    type_id = get_type_id()
    out = str(content).split('TaskOutput')
    # 打印文件内容
    for i in out:
        # print(i)
        if 'final_output' in i:
            a = i[17:]
            position_e = out[-1].find(', summary')
            # print(position_s, position_e)
            q = '\'' + out[-1][48:position_e]
            # print('q:',q)
            # print('a:',a)
            insert_data(oasst1_val, q, a, type_id)
            # print('--------------')
        else:
            position_e = i.find(', summary')
            # print(position_s, position_e)
            q = i[13:position_e]
            position_s = i.find('exported_output=')
            a = i[position_s+16:]
            # print('q:',q)
            # print('a:',a)
            insert_data(oasst1_val, q, a, type_id)
            # print('--------------')

# 有知识的话直接重组一次任务
