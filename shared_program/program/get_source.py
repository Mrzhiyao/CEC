import time


def get_source(path):
    results = []
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line != '\n':
                results.append(line.replace('\n', ''))
    # print(results)
    return results[0], results[1], results[2]

# print(get_source())



