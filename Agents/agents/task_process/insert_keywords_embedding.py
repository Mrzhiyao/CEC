import json
# KnowledgeQuery(query): Capture the relevant webpages according to the query;
# ParagraphRetrieve(context, query): Given a query, retrieve the most relevant paragraphs from the given context;
# QA(context, query): Given context, answer the given query;
# Calculator(formula): Calculate the input formula;
# Code(pseudo_code): Generate a Python function that corresponds to the pseudo code.
# 假设jsonl_file是一个包含JSONL数据的文件对象
from langchain.agents import Tool
from langchain.utilities import GoogleSerperAPIWrapper
from vector_database.insert_milvus import insert_data
from utils import get_type_id

from vector_database.insert_keywords_milvus import insert_keywords_data
from text_character.get_keywords import get_keywords

def insert_keywords_result(q, content, type_id, q_id, a_id):

    # q = q.replace(" = KnowledgeQuery","")
    q = q.replace(" = ParagraphRetrieve","")
    q = q.replace(" = QA","")
    q = q.replace(" = Calculator","")
    q = q.replace(" = Code","")

    for i in range(15):
        cont = 'R' + str(i)
        q = q.replace(cont,"")
        # q = q.replace("(","")
        # q = q.replace(")","")
    q = q.replace("Query","")
    q = q.replace("Question","")
    q = q.replace(": ","")
    q = q.replace(": ","")
    q = q.replace("[],","")
    q = q.replace("(,", "")
    q = q.replace("([,", "")
    q = q.replace(")", "")
    q = q.replace("(", "")
    q = q.replace("[", "")
    q = q.replace("]", "")
    q = q.replace("],", "")
    q = q.replace("[,", "")
    q_words = get_keywords(q)
    a_words = get_keywords(content)

    number_q = 0
    number_a = 0
    for words_rank, words in q_words:
        if number_q >=5:
            break
        if words_rank >1.0:
            # print('start')
            # print(words_rank, words)
            insert_keywords_data(words, words_rank, q, q_id, type_id)
            number_q = number_q + 1
    for words_rank, words in a_words:
        if number_a >=5:
            break
        if words_rank >1.0:
            # print('start')
            # print(words_rank, words)
            insert_keywords_data(words, words_rank, content, a_id, type_id)
            number_a = number_a + 1


