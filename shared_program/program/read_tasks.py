def read_task(path):
    global output
    f = open(path,encoding='utf-8',errors='ignore')             # 返回一个文件对象
    line = f.readline()             # 调用文件的 readline()方法
    while line:
        output = line
        # print(line)               # 后面跟 ',' 将忽略换行符
        # print(line, end = '')　　　# 在 Python 3中使用
        line = f.readline()
        # print(line)
    f.close()

    return output
