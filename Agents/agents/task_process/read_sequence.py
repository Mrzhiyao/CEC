def read_sequence(path):
    file = open(path, 'r')
    content = file.read()  # 所有内容保存在content中
    file.close()
    return content
