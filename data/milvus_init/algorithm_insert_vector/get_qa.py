import os
import json

def task_get(line):
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


def get_qa_results():
    with open('os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task_result_500.txt')', 'r', encoding='utf-8') as file:
        number = 0
        results = []
        for line in file:
            result = []
            # print(line)
            # if number > 10:
            #     break
            # print('line', line.split('TaskOutput'))
            content = line.split('description=')

            for lines in range(len(content)):
                # print(lines, content[lines])
                if lines == 0:
                    continue
                # ques = content[lines].split(', summary=')[0]
                ques = content[lines].split('exported_output=')[1]
                result.append(ques.split('), TaskOutput(')[0])

            results.append(result)
            number = number + 1

        # print(len(results))
    
    return results

def get_qa_ques():

    jsonl_file = open('os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared_program', 'program', 'lumos_complex_qa_ground_onetime.jsonl')', 'r')
    # 逐行读取
    n = 0
    results_ques = []
    for line in jsonl_file:
        # result_ques = []
        if n > 500:
            break
        task_list = task_get(line)
        results_ques.append(task_list)
        n = n + 1
        # tasks = get_tasks(task_list)

    jsonl_file.close()
    # print(len(results_ques))

    return results_ques