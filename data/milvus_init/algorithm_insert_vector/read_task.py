def read_task(path):
    out = []
    f = open(path)             # 返回一个文件对象  
    line = f.readline()             # 调用文件的 readline()方法  
    while line:  
        output = line
        out.append(output)
        # print(line)               # 后面跟 ',' 将忽略换行符  
        # print(line, end = '')　　　# 在 Python 3中使用  
        line = f.readline()  
    f.close()
    return out
