import os
from search_milvus import search_data
from read_task import read_task

def read_file():
    with open('os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task_result.txt')', 'r', encoding='utf-8') as f1:
        list = []
        for num, line in enumerate(f1.readlines()):
            # print(line)
            pattern = line.replace('\t','\n')  # 用换行符替换所有缩进符
            list.append(pattern)
    return list

# print(len(read_file()))

#以 utf-8 的编码格式打开指定文件
f = open('os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task_result.txt')',encoding = "utf-8")
#输出读取到的数据
# print(f.read())
out = f.read()
f.close()

x1 = out.split('\n')



error_record = read_task('os.path.join(os.path.dirname(os.path.abspath(__file__)), 'error.txt')')
# print(error_record)
new_record= []
for i in error_record:
    new_record.append(i.replace("\n",""))


# print(x1[77])
x_out = []
wanzheng_content = ''
for n in range(1800):
    # if n==85:
    #     print(x1[n])
    if n>=7 and n<13:
        wanzheng_content = wanzheng_content + '\n' + x1[n]
        continue
    if n == 13:
        x_out.append(wanzheng_content)
        wanzheng_content = ''
        continue

    if n>=77 and n<85:
        wanzheng_content = wanzheng_content + '\n' + x1[n]
        continue
    if n == 85:
        x_out.append(wanzheng_content)
        wanzheng_content = ''
        continue

    if n>=217 and n<229:
        wanzheng_content = wanzheng_content + '\n' + x1[n]
        continue
    if n == 229:
        x_out.append(wanzheng_content)
        wanzheng_content = ''
        continue

    if n>=242 and n<249:
        wanzheng_content = wanzheng_content + '\n' + x1[n]
        continue
    if n == 249:
        x_out.append(wanzheng_content)
        wanzheng_content = ''
        continue

    if n>=315 and n<338:
        wanzheng_content = wanzheng_content + '\n' + x1[n]
        continue
    if n == 338:
        x_out.append(wanzheng_content)
        wanzheng_content = ''
        continue

    if n>=387 and n<422:
        wanzheng_content = wanzheng_content + '\n' + x1[n]
        continue
    if n == 422:
        x_out.append(wanzheng_content)
        wanzheng_content = ''
        continue

    if n>=572 and n<613:
        wanzheng_content = wanzheng_content + '\n' + x1[n]
        continue
    if n == 613:
        x_out.append(wanzheng_content)
        wanzheng_content = ''
        continue


    # if n == 653:
    #     print('dddddd',x1[n])
    if n>=667 and n<673:
        wanzheng_content = wanzheng_content + '\n' + x1[n]
        continue
    if n == 673:
        x_out.append(wanzheng_content)
        wanzheng_content = ''
        continue

    if n>=724 and n<737:
        wanzheng_content = wanzheng_content + '\n' + x1[n]
        continue
    if n == 737:
        x_out.append(wanzheng_content)
        wanzheng_content = ''
        continue

    if n>=758 and n<766:
        wanzheng_content = wanzheng_content + '\n' + x1[n]
        continue
    if n == 766:
        print(wanzheng_content)
        x_out.append(wanzheng_content)
        wanzheng_content = ''
        continue

    if n>=804 and n<836:
        wanzheng_content = wanzheng_content + '\n' + x1[n]
        continue
    if n == 836:
        x_out.append(wanzheng_content)
        wanzheng_content = ''
        continue

    if n>=901 and n<918:
        wanzheng_content = wanzheng_content + '\n' + x1[n]
        continue
    if n == 918:
        x_out.append(wanzheng_content)
        wanzheng_content = ''
        continue

    if n>=1202 and n<1254:
        wanzheng_content = wanzheng_content + '\n' + x1[n]
        continue
    if n == 1254:
        x_out.append(wanzheng_content)
        wanzheng_content = ''
        continue

    if n>=1278 and n<1315:
        wanzheng_content = wanzheng_content + '\n' + x1[n]
        continue
    if n == 1315:
        x_out.append(wanzheng_content)
        wanzheng_content = ''
        continue

    if n>=1392 and n<1406:
        wanzheng_content = wanzheng_content + '\n' + x1[n]
        continue
    if n == 1406:
        x_out.append(wanzheng_content)
        wanzheng_content = ''
        continue

    if n>=1413 and n<1461:
        wanzheng_content = wanzheng_content + '\n' + x1[n]
        continue
    if n == 1461:
        x_out.append(wanzheng_content)
        wanzheng_content = ''
        continue

    if n>=1630 and n<1636:
        wanzheng_content = wanzheng_content + '\n' + x1[n]
        continue
    if n == 1636:
        x_out.append(wanzheng_content)
        wanzheng_content = ''
        continue

    if n>=1690 and n<1717:
        # print(x1[n])
        wanzheng_content = wanzheng_content + '\n' + x1[n]
        continue
    if n == 1717:
        x_out.append(wanzheng_content)
        wanzheng_content = ''
        continue


    x_out.append(x1[n])
print(len(x_out))
# for i in range(2000):
#     if 1000 <= i <= 1008:
#         print(x_out[i])
#         print('\n')

# for i in new_record:
#     print(i)
#     print(x1[int(i)])
# print(x1[1276:1314])

def get_newcontent():
    f = open('os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task_result.txt')',encoding = "utf-8")
    #输出读取到的数据
    # print(f.read())
    out = f.read()
    f.close()

    x1 = out.split('\n')


    error_record = read_task('os.path.join(os.path.dirname(os.path.abspath(__file__)), 'error.txt')')
    # print(error_record)
    new_record= []
    for i in error_record:
        new_record.append(i.replace("\n",""))


    # print(x1[77])
    x_out = []
    wanzheng_content = ''
    for n in range(5000):
        if n>=7 and n<13:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 13:
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=77 and n<85:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 85:
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=217 and n<229:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 229:
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=242 and n<249:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 249:
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=315 and n<338:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 338:
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=387 and n<422:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 422:
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=572 and n<613:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 613:
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        # if n>=652 and n<653:
        #     wanzheng_content = wanzheng_content + '\n' + x1[n]
        #     continue
        # if n == 653:
        #     x_out.append(wanzheng_content)
        #     wanzheng_content = ''
        #     continue

        if n>=667 and n<673:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 673:
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=724 and n<737:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 737:
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue


        if n>=758 and n<765:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 765:
            # print(wanzheng_content)
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=778 and n<780:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 780:
            # print(wanzheng_content)
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue


        if n>=802 and n<834:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 834:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=901 and n<918:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 918:
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=942 and n<959:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 959:
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=1202 and n<1254:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 1254:
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=1276 and n<1313:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 1313:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=1392 and n<1406:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 1406:
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=1413 and n<1461:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 1461:
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=1630 and n<1636:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 1636:
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=1690 and n<1717:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 1717:
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=1731 and n<1733:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 1733:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=2003 and n<2039:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 2039:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=2053 and n<2086:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 2086:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=2132 and n<2157:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 2057:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=2277 and n<2279:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 2279:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue


        if n>=2331 and n<2350:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 2350:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=2391 and n<2415:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 2415:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=2431 and n<2479:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 2479:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=2492 and n<2494:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 2494:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=2497 and n<2499:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 2499:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=2521 and n<2527:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 2527:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=2620 and n<2622:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 2622:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=2683 and n<2685:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 2685:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=2703 and n<2705:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 2705:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=2741 and n<2767:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 2767:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=2801 and n<2806:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 2806:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=2839 and n<2843:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 2843:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=2927 and n<2929:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 2929:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=2954 and n<2999:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 2999:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=3023 and n<3042:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 3042:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=3064 and n<3068:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 3068:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=3070 and n<3092:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 3092:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=3135 and n<3183:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 3183:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=3187 and n<3227:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 3227:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=3269 and n<3291:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 3291:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=3381 and n<3382:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 3382:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=3399 and n<3401:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 3401:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=3434 and n<3479:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 3479:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=3605 and n<3607:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 3607:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=3632 and n<3639:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 3639:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=3663 and n<3719:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 3719:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=3734 and n<3766:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 3766:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=3809 and n<3868:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 3868:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=3956 and n<3997:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 3997:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue

        if n>=4080 and n<4083:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 4083:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue
    
        if n>=4097 and n<4101:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            continue
        if n == 4101:
            wanzheng_content = wanzheng_content + '\n' + x1[n]
            x_out.append(wanzheng_content)
            wanzheng_content = ''
            continue
        x_out.append(x1[n])
    return x_out

# listt = get_newcontent()
# for i in listt:
#     print(i)