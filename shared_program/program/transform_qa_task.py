import json
# KnowledgeQuery(query): Capture the relevant webpages according to the query;
# ParagraphRetrieve(context, query): Given a query, retrieve the most relevant paragraphs from the given context;
# QA(context, query): Given context, answer the given query;
# Calculator(formula): Calculate the input formula;
# Code(pseudo_code): Generate a Python function that corresponds to the pseudo code.
# 假设jsonl_file是一个包含JSONL数据的文件对象

def task_get_qa(line):
    # 解码JSON
    global task_list
    json_obj = json.loads(line)
    # 处理json_obj
    # print(len(json_obj['messages']))

    for i in range(len(json_obj['messages'])):
        # print(json_obj['messages'][i]['role'])
        # print(json_obj['messages'][i]['content'])
        if 'assistant' in json_obj['messages'][i]['role']:
            str_content = json_obj['messages'][i]['content']
            task_list = str_content.split(';')
            # print(task_list)

    # print('----------------------')
    return task_list

# 有知识的话直接重组一次任务
