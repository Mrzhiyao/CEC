import spacy

# 加载英文语言模型
nlp = spacy.load("en_core_web_sm")

# # 定义待处理的文本
# text = "Elon Musk is the CEO of SpaceX, which is headquartered in Hawthorne, California. " \
#        "Tesla, Inc. is an American electric vehicle and clean energy company founded by Elon Musk."
# text2 = 'Mike like you'
# text3 = 'Beijing is beautiful'
# # 处理文本
# doc = nlp(text2)

# # 初始化领域字典
# domain_counter = {
#     "Technology": 0,
#     "Automotive": 0,
#     "Finance": 0,
#     "Other": 0
# }
# print(doc.ents)
# '是否需要一个other'
# # 获取实体并统计各个领域的实体数量
# for ent in doc.ents:
#     print(ent.text)
#     print(ent.label_)
#     if ent.label_ == "PERSON":
#         domain_counter["Technology"] += 1
#     elif ent.label_ == "ORG":
#         if "Tesla" in ent.text:
#             domain_counter["Automotive"] += 1
#         elif "SpaceX" in ent.text:
#             domain_counter["Technology"] += 1
#         else:
#             domain_counter["Other"] += 1
#     elif ent.label_ == "GPE":
#         domain_counter["Other"] += 1

# # 根据实体数量推断文本所属的领域
# max_domain = max(domain_counter, key=domain_counter.get)
# print("文本所属的领域是:", max_domain)

# spaCy 支持以下实体类型：
# PERSON、NORP（国籍、宗教和政治团体）、FAC（建筑物、机场等）、
# ORG（组织）、GPE（国家、城市等）、LOC（山脉、水体等）、PRODUCT（产品） 、
# EVENT（事件名称）、WORK_OF_ART（书籍、歌曲名称）、LAW（法律文件名称）、
# LANGUAGE（命名语言）、DATE、TIME、PERCENT、MONEY、QUANTITY、ORDINAL 和 CARDINAL。
def  get_filed(text):
    lists = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            lists[0] = lists[0] + 1
        elif ent.label_ == "NORP":
            lists[1] = lists[1] + 1
        elif ent.label_ == "FAC":
            lists[2] = lists[2] + 1
        elif ent.label_ == "ORG":
            lists[3] = lists[3] + 1
        elif ent.label_ == "GPE":
            lists[4] = lists[4] + 1
        elif ent.label_ == "LOC":
            lists[5] = lists[5] + 1
        elif ent.label_ == "PRODUCT":
            lists[6] = lists[6] + 1
        elif ent.label_ == "EVENT":
            lists[7] = lists[7] + 1
        elif ent.label_ == "WORK_OF_ART":
            lists[8] = lists[8] + 1
        elif ent.label_ == "LAW":
            lists[9] = lists[9] + 1
        elif ent.label_ == "LANGUAGE":
            lists[10] = lists[10] + 1
        elif ent.label_ == "DATE":
            lists[11] = lists[11] + 1
        elif ent.label_ == "TIME":
            lists[12] = lists[12] + 1
        elif ent.label_ == "PERCENT":
            lists[13] = lists[13] + 1
        elif ent.label_ == "MONEY":
            lists[14] = lists[14] + 1
        elif ent.label_ == "QUANTITY":
            lists[15] = lists[15] + 1
        elif ent.label_ == "ORDINAL":
            lists[16] = lists[16] + 1
        elif ent.label_ == "CARDINAL":
            lists[17] = lists[17] + 1

    return lists

# text = "Elon Musk is the CEO of SpaceX, which is headquartered in Hawthorne, California. " \
#        "Tesla, Inc. is an American electric vehicle and clean energy company founded by Elon Musk."

# print(get_filed(text))