from gpu_size import monitor_resources
import time
import os
from send_resource import send_file

_RESOURCE_DIR = os.path.dirname(os.path.abspath(__file__))

while 1:
    gpu_utilization, memory_utilization, memory_used_val = monitor_resources()
    file_name = 'resource.txt'
    file_path = os.path.join(_RESOURCE_DIR, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(str(gpu_utilization))
        file.write('\n')
        file.write(str(memory_utilization))
        file.write('\n')
        file.write(str(memory_used_val))

    send_file(file_path, '192.168.2.5')
    time.sleep(5)
