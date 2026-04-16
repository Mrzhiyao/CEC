import json

def task_get(line):
    # 解码JSON
    global task_list
    json_obj = json.loads(line)
    for i in range(len(json_obj['messages'])):
        if 'assistant' in json_obj['messages'][i]['role']:
            str_content = json_obj['messages'][i]['content']
            task_list = str_content.split(';')

    return task_list

