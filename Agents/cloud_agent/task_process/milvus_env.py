import random
import numpy as np
from towhee_embedding import towhee_embedding
from llm_use import use_llm
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import time


def search(search_vec, collection):
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }
    oasst1_val = collection
    result = oasst1_val.search(search_vec, "vec", search_params, limit=10,
                               output_fields=["id", "text", "role", "review_count", "rank", "text_answer",
                                              "text_answerid"])
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
    for hits in result:
        for hit in hits:
            print(hit)
            distance = hit.distance
            # print(distance,'distance')
            distances.append(distance)
            # print(hit.distance, hit.entity.get('review_count'))
            id = hit.entity.get('id')
            ids.append(id)
            # print(id)

            text = hit.entity.get('text')

            texts.append(text)
            role = hit.entity.get('role')
            roles.append(role)
            review_count = hit.entity.get('review_count')
            review_counts.append(review_count)
            rank = hit.entity.get('rank')
            if rank == 'nan':
                rank = 0
            ranks.append(rank)
            text_answer = hit.entity.get('text_answer')
            text_list.append(text_answer)
            text_answers.append(text_answer)
            text_answerid = hit.entity.get('text_answerid')
            text_answerids.append(text_answerid)
    return distances, ids, texts, roles, review_counts, ranks, text_answers, text_answerids, text_list


def search_neighbor(symbol, neirong):
    # print(vectors_to_search, len(vectors_to_search))

    sorted_rank_value = []
    rank_value = []
    id = []
    vec = []
    text = []
    send_node = []
    similarity = []

    if symbol == 1:
        connections.disconnect("default")
        connections.connect("default", host="10.10.1.230", port="19530")
        from pymilvus import MilvusClient
        client = MilvusClient(
            uri='http://10.10.1.230:19530'
        )
        collect_base = Collection("knowledge3")
    #
    # elif symbol == 2:
    #     connections.disconnect("default")
    #     connections.connect("default", host="192.168.140.132", port="19530")
    #     from pymilvus import MilvusClient
    #     client = MilvusClient(
    #         uri='http://192.168.140.132:19530'
    #     )
    #     collect_base = Collection("two_input_nodes21")
    #
    # elif symbol == 3:
    #     connections.disconnect("default")
    #     connections.connect("default", host="192.168.140.133", port="19530")
    #     from pymilvus import MilvusClient
    #     client = MilvusClient(
    #         uri='http://192.168.140.133:19530'
    #     )
    #     collect_base = Collection("two_input_nodes31")

    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }
    # que = oasst1_val.query(expr="id==4407", output_fields=["id","text","vec"])
    result = collect_base.search(neirong, "vec", search_params, limit=10,
                                 output_fields=["id", "text", "vec", "send_node", "rank"])
    num = 0
    try:
        for hits in result:
            for hit in hits:
                num = num + 1
                # sorted_rank_value
                similarity.append(hit.distance)
                rank_value.append(hit.entity.get('rank'))
                id.append(hit.entity.get('id'))
                vec.append(hit.entity.get('vec'))
                text.append(hit.entity.get('text'))
                send_node.append(hit.entity.get('send_node'))
    except:
        print('send记录为空')
    if len(rank_value) < 10:
        #  如果现在发送记录不足10条，则padding补充
        for i in range(10 - len(rank_value)):
            similarity.append(1000)
            rank_value.append(-30)
            id.append(111 + i)
            vec.append([[random.randint(1, 1) for _ in range(768)]])
            text.append('补充的padding内容')
            # 起点?
            if symbol == 1:
                send_node.append(random.randint(2, 3))
            elif symbol == 2:
                midd = random.randint(2, 3)
                if midd == 2:
                    midd = 1
                send_node.append(midd)
            else:
                send_node.append(random.randint(1, 2))
    #   从大到小进行排序
    rank_sorted = rank_value
    # sorted_index = np.argsort(rank_value)
    sorted_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    return similarity, rank_value, id, vec, text, send_node, rank_sorted, sorted_index


def reset(node, task_status, ques, answer):
    i = node
    state = np.random.randn(30)
    task = []
    task_array = []
    answer_array = []
    task_embeding = []
    state_id = []
    search_delay = []
    send_nodes = []
    task_status = 1
    ques_list = ques
    answer_list = answer
    task_array_embedding = []
    task_array.insert(0, ques_list)
    answer_array.insert(0, answer_list)
    # task.append(ques_list[i])
    # print(task_array[0],'task_array[0]')
    task_array_embedding.append(towhee_embedding(task_array[0][0]))
    # print(task_array_embedding)
    quotidian = 0
    if i == 0:
        connections.disconnect("default")
        connections.connect("default", host="10.10.1.230", port="19530")
        from pymilvus import MilvusClient
        client = MilvusClient(
            uri='http://10.10.1.230:19530'
        )
        collection_name1 = "knowledge3"
        collection = Collection("knowledge3")

        collection_send1 = Collection("knowledge3")
        # collection_step = collection1
        collection_name = collection_name1
        collection_send = collection_send1
    # elif i == 1:
    #     connections.disconnect("default")
    #     connections.connect("default", host="192.168.140.132", port="19530")
    #     from pymilvus import MilvusClient
    #     client = MilvusClient(
    #         uri='http://192.168.140.132:19530'
    #     )
    #     collection_name2 = "two_input21"
    #     collection = Collection("two_input21")
    #     collection_send2 = Collection("two_input_nodes21")
    #     # collection_step = collection2
    #     collection_name = collection_name2
    #     collection_send = collection_send2
    # else:
    #     connections.disconnect("default")
    #     connections.connect("default", host="192.168.140.133", port="19530")
    #     from pymilvus import MilvusClient
    #     client = MilvusClient(
    #         uri='http://192.168.140.133:19530'
    #     )
    #     collection_name3 = "two_input31"
    #     collection = Collection("two_input31")
    #     collection_send3 = Collection("two_input_nodes31")
    #     # collection_step = collection3
    #     collection_name = collection_name3
    #     collection_send = collection_send3

    # print(self.task_array[i],'self.task_array[i]')
    if task_array != []:

        start_time = time.time()
        # print(self.task_array_embedding[i], collection,'self.task_array_embedding[i], collection')
        distances, ids, texts, roles, review_counts, ranks, text_answers, text_answerids, text_list = search(
            task_array_embedding, collection)
        # 如果向量数据库中无内容？
        for j in range(10):
            state[quotidian] = distances[j]
            quotidian = quotidian + 1
            if roles[j] == 'prompter':
                state[quotidian] = 0
            else:
                state[quotidian] = 1
            quotidian = quotidian + 1
            state[quotidian] = review_counts[j]
            quotidian = quotidian + 1
            state_id.append(ids[j])
        # similarity, rank_value, id, vec, text, send_node, rank_sorted, sorted_index = search_neighbor(
        #     symbol=i + 1, neirong=task_array_embedding)
        # 补充边缘节点处理信息到state里
        # for k in range(10):
        #     state[quotidian] = similarity[sorted_index[k]]
        #     quotidian = quotidian + 1
        #     send_nodes.append(send_node[sorted_index[k]])
        time_consume = time.time() - start_time
        search_delay.append(time_consume)
    # else:  # 该节点现在无任务
    #     start_time = time.time()
    #     for j in range(10):
    #         state[quotidian] = 10
    #         quotidian = quotidian + 1
    #         state[quotidian] = 2  # 无内容
    #         quotidian = quotidian + 1
    #         state[quotidian] = 0
    #         quotidian = quotidian + 1
    #
    #     similarity, rank_value, id, vec, text, send_node, rank_sorted, sorted_index = search_neighbor(
    #         symbol=i + 1,
    #         neirong=[[random.randint(1, 1) for _ in range(768)]])
    #     # 补充边缘节点处理信息到state里
    #     # print('sds')
    #     for k in range(10):
    #         state[quotidian] = 0
    #         quotidian = quotidian + 1
    #         send_nodes.append(send_node[sorted_index[k]])
    #     time_consume = time.time() - start_time
    #     search_delay.append(time_consume)
    # 通过动作，更新大模型需要处理的LLM序列中的任务数
    # 起始为0
    # self.state[-1] = 0
    # print(len(self.state))
    # print(len(self.task_array[0]),len(self.task_array[1]),len(self.task_array[2]))
    # state[-1] = 0
    # state[-2] = 0
    # state[-3] = 0
    return state, task_array, send_nodes, state_id, text_list, texts


def step(texts, wenben_list, node, task_status, ques, answer, state_for_common, action, state_id):
    i = node
    qiwang_task = 0
    actual_task = 0

    dt = np.zeros(3)
    dloss = np.zeros(3)
    dcost = np.zeros(3)

    reward_list = np.zeros(3 + 1)  # reward_list has to be a horizontial numpy vector
    sim_list = np.zeros(3)
    delay_list = np.zeros(3)
    llm_use_yn = np.zeros(3)

    if i == 0:
        connections.disconnect("default")
        connections.connect("default", host="10.10.1.230", port="19530")
        from pymilvus import MilvusClient
        client = MilvusClient(
            uri='http://10.10.1.230:19530'
        )
        collection_name1 = "knowledge3"
        collection1 = Collection("knowledge3")

        collection_send1 = Collection("knowledge3")
        collection_step = collection1
        collection_name = collection_name1
        collection_send = collection_send1
    print(len(state_for_common))
    state_agent = state_for_common[0:60]

    if action == 0:
        value = -1000
        suoyin = 0
        action_select = 0

        if state_agent[3 * action_select + 1] == 0.0 and state_agent[3 * action_select] < 0.15:
            response = wenben_list[action_select]
            result = response
            print('直接使用query结果')
        else:
            if state_agent[3 * action_select + 1] == 0.0:
                print('通过知识库增强prompt后请求LLM')
                print('参考文档：' + wenben_list[action_select] + '\n' + ques[0])
                response, consume = use_llm(
                    '在回答问题前，请你记住你是北京师范大学超算中心助手，你会回答用户关于超算中心的问题。注意：1.如果提示信息与用户的问题关联很大，请使用提示信息作答，不要超出提示信息的范围；2.如果提示信息与用户问题无关自行回答。3.如果用户输入的问题过于简单，表述不清，可以要求用户补充描述问题。以下为提示信息' + '\n' +
                    wenben_list[action_select] + '\n' + '用户问题:' + '\n' + ques[0])
                result = response
            else:
                print('通过知识库增强prompt后请求LLM')
                print('参考文档：' + texts[action_select] + '\n' + ques[0])
                response, consume = use_llm(
                    '在回答问题前，请你记住你是北京师范大学超算中心助手，你会回答用户关于超算中心的问题。注意：1.如果提示信息与用户的问题关联很大，请使用提示信息作答，不要超出提示信息的范围；2.如果提示信息与用户问题无关自行回答。3.如果用户输入的问题过于简单，表述不清，可以要求用户补充描述问题。以下为提示信息' + '\n' +
                    wenben_list[action_select] + '\n' + '用户问题:' + '\n' + ques[0])
                result = response


    else:
        print('直接请求LLM')
        llm_use_yn[i] = 1
        response, consume = use_llm(
            '在回答问题前，请你记住你是北京师范大学超算中心助手，你会回答用户关于超算中心的问题。注意：1.如果你并不清楚请如实回答；2.如果用户输入的问题过于简单，表述不清，可以要求用户补充描述问题' + '\n' +
            ques[0])
        result = response

    return result
