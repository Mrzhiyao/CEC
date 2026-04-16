import os
from pathlib import Path
 
def get_files_by_ctime(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            ctime = os.path.getctime(file_path)
            yield file_path, ctime, file
 
# 示例使用

def get_time_file(path):
    files = []
    times = []
    for file_path, ctime, file in sorted(get_files_by_ctime(path), key=lambda x: x[1]):
        files.append(file)
        times.append(ctime)
    return files, times
