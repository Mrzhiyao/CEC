import re

# 示例数据
Rcontent = ['R1 =', 'R2 =', 'R3 =', 'R4 =', 'R5 =', 'R6 =', 'R7 =', 'R8 =', 'R9 =', 'R10 =']
cankao = {
    'KnowledgeQuery': 0, 
    'ParagraphRetrieve': 1,
    'QA': 2,
    'Calculater': 3,
    'Code_generate': 4,
    'Comprehensive': 5
}

Rcontent_check = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10']

def analyze_tasks(task_content_list):
    # 任务1：统计每个任务中出现的Rcontent数量
    count_list = []
    # 任务2：每个任务最后一个R对应的cankao值
    cankao_list = []
    # 任务3：是否任务列表的最后一项
    last_flag_list = [1 if i == len(task_content_list)-1 else 0 
                      for i in range(len(task_content_list))]
    
    # 预处理Rcontent，构建匹配字典 {数字: 完整字符串}
    r_content_map = {}
    for r in Rcontent:
        # 提取R后面的数字
        match = re.search(r'R(\d+)', r)
        if match:
            num = int(match.group(1))
            r_content_map[num] = r
    
    for task_str in task_content_list:
        # 任务1：统计Rcontent出现次数
        count = 0
        found_r_nums = set()
        
        # 检查每个可能的R编号
        for r_num, r_str in r_content_map.items():
            pattern = re.escape(r_str).replace(r'\ ', r'\s*')  # 允许等号前后有空格
            if re.search(pattern, task_str):
                count += 1
                found_r_nums.add(r_num)
        
        count_list.append(count)
        
        # 任务2：查找最后一个R对应的内容及其cankao值
        last_r_value = None
        
        if found_r_nums:
            # 找出数字最大的R（最后一个）
            last_r_num = max(found_r_nums)
            last_r_str = r_content_map[last_r_num]
            
            # 提取最后一个R的完整定义
            pattern = re.escape(last_r_str).replace(r'\ ', r'\s*') + r'(.*?)(?=R\d|$)'
            match = re.search(pattern, task_str, re.DOTALL)
            
            if match:
                r_content = match.group(1).strip()
                
                # 查找r_content中出现的cankao关键词
                for key, value in cankao.items():
                    if re.search(r'\b' + re.escape(key) + r'\b', r_content):
                        last_r_value = value
                        break  # 只取第一个匹配到的值
            
        cankao_list.append(last_r_value)
    
    return count_list, cankao_list, last_flag_list



# task_content_list = [
#     'This is a progressive task:\nR1 = KnowledgeQuery(Boston) R2 = ParagraphRetrieve(R1, Query: Boston became one of the wealthiest international ports after what war?) R3 = QA([R2], Question: Boston became one of the wealthiest international ports after what war?)',
#     'This is a progressive task:\n R4 = KnowledgeQuery(Museum of the American Revolution) R5 = ParagraphRetrieve(R4, Query: When did the museum of the American Revolution open?) R6 = QA([R5], Question: When did the museum of the American Revolution open?)',
#     ' R7 = QA([R3, R6], Question: What was the opening date of the museum dedicated to the war that, after it occurred, Boston became one of the wealthiest international ports?)'
# ]

# # 执行分析
# task1, task2, task3 = analyze_tasks(task_content_list, Rcontent, cankao)

# # 打印结果
# print(f"任务1结果（每个任务中Rcontent出现次数）: {task1}")
# print(f"任务2结果（每个任务最后一个R的cankao值）: {task2}")
# print(f"任务3结果（是否最后一个任务）: {task3}")

# 3, 2, 1

def find_shared_references(task_content_list):
    """
    找出不同任务间共享的R引用，并返回共享的任务索引对
    
    Args:
        task_content_list (list): 包含多个任务描述的列表
        Rcontent_check (list): 需要检查的R变量列表（如['R1', 'R2', ...]）
        
    Returns:
        list: 包含共享R变量的任务索引对，格式为[[smaller_index, larger_index]]
    """
    # 存储每个任务中出现的R变量
    task_refs = []
    
    # 遍历所有任务，提取出现的R变量
    for task_str in task_content_list:
        task_refs.append([])
        for r in Rcontent_check:
            if r in task_str:
                task_refs[-1].append(r)
    
    # 找出共享R变量的任务对
    shared_pairs = []
    num_tasks = len(task_content_list)
    
    # 遍历所有任务对 (i, j)，其中i < j
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            # 找出两个任务共享的R变量
            shared_refs = set(task_refs[i]) & set(task_refs[j])
            
            # 如果有共享变量，则添加到结果列表
            if shared_refs:
                shared_pairs.append([i, j])
    
    return shared_pairs

# # 示例数据
# task_content_list = [
#     'This is a progressive task:\nR1 = KnowledgeQuery(Boston) R2 = ParagraphRetrieve(R1, Query: Boston became one of the wealthiest international ports after what war?) R3 = QA([R2], Question: Boston became one of the wealthiest international ports after what war?)',
#     'This is a progressive task:\n R4 = KnowledgeQuery(Museum of the American Revolution) R5 = ParagraphRetrieve(R4, Query: When did the museum of the American Revolution open?) R6 = QA([R5], Question: When did the museum of the American Revolution open?)',
#     ' R7 = QA([R3, R6], Question: What was the opening date of the museum dedicated to the war that, after it occurred, Boston became one of the wealthiest international ports?)'
# ]

# Rcontent_check = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10']

# # 找出共享R变量的任务对
# result = find_shared_references(task_content_list, Rcontent_check)

# print(f"共享R变量的任务对: {result}")