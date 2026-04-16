#encoding=utf-8
import socket
import os
import time
import json

def print_bar1(percent):
    bar = '\r' + '*' * int((percent * 100)) + ' %3.0f%%|' % (percent * 100) + '100%'
    print(bar, end='', flush=True)

def send_file(content, ip):
    client = socket.socket()
    client.connect((ip, 10081))
    file_path = content
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    dic = {"option": "1", "name": file_name, "file_size": file_size}
    client.send(json.dumps(dic).encode())
    file_seek = int(client.recv(100).decode())
    if file_seek == file_size:
        print('文件已经存在服务端，退出此次传输...')
    else:
        new_size = file_size - file_seek
        begin_size = new_size
        with open(file_path, 'rb') as f:
            f.seek(file_seek)
            while new_size:
                content = f.read(1024)
                client.send(content)
                new_size -= len(content)
                print_bar1(round((begin_size - new_size) / begin_size, 2))
                time.sleep(0.2)
            print('')
    client.close()
