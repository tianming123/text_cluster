import json
import pandas as pd
import numpy as np
import re


from collections import Counter

import tran


def counter(arr,n):
    return Counter(arr).most_common(len(Counter(arr))) # 返回出现频率最高的两个数

#使用正则表达式对作者分词
def auth_participle(auth_item):
    pattern = r'[(\d)]'
    if type(auth_item) != float:
        result = [name.replace(';','').strip(' ') for name in re.split(pattern,auth_item) if len(name)>2]
        return result
    return []

#使用正则表达式对机构分词
def aff_participle(aff_item):
    pattern = r'[(\d)]'
    if type(aff_item) != float:
        result = [name.replace(';','').strip(' ') for name in re.split(pattern,aff_item) if len(name)>2]
        return result
    return []

def parse_auth():
    df = pd.read_excel("allData.xls")
    auth_total = []
    origin_auth_list = np.array(df['Author'])
    for auth_item in origin_auth_list:
        #  正则表达式分词
        auth_total+=auth_participle(auth_item)
    result = []
    index = 1
    #  统计频次
    for auth in counter(auth_total,20):
        tmp_dict = {}
        tmp_dict["au_id"] = index
        tmp_dict["au_name"] = auth[0]
        tmp_dict["number_of_papers"] = auth[1]
        tmp_dict["rank_of_papers"] = index
        result.append(tmp_dict)
        index+=1

    # with open('10_top_authors.json', 'w') as f:
    #     json.dump(result, f)
    return result

#机构映射函数case by case
def aff_map(aff_total):
    for i in range(len(aff_total)):
        #  精准匹配
        if aff_total[i] in [ 'Microsoft Research',"Microsoft Research Asia","Microsoft Corporation","Microsoft Research India"]:
            aff_total[i] = 'Microsoft'
        if aff_total[i] == "MIT CSAIL" or aff_total[i] == "MIT":
            aff_total[i] = "Massachusetts Institute of Technology"
        if aff_total[i] in ["Hong Kong University of Science and Technology" ,"Chinese University of Hong Kong","University of Hong Kong"]:
            aff_total[i] = "Hong Kong University"
        if aff_total[i] == " University of California":
            aff_total[i] = "University of California"
        if aff_total[i] == " National University of Singapore":
            aff_total[i] = "National University of Singapore"
        if aff_total[i] == "Google Inc." or aff_total[i]=="Google Research":
            aff_total[i] = "Google"
        if aff_total[i] == "University of Illinois at Urbana-Champaign":
            aff_total[i] = "University of Illinois"
        if aff_total[i] in ["University of Texas at Austin","University of Texas at Arlington"]:
            aff_total[i] = "University of Texas"
        if aff_total[i] in ["Intel Corporation" ,"Intel Labs","Intel Labs."]:
            aff_total[i] = "Intel"
        if aff_total[i] == "HP Labs.":
            aff_total[i] = "HP Labs"
        if aff_total[i] in ["IBM Research","IBM Almaden Research Center"]:
            aff_total[i] = 'IBM'
        if aff_total[i] ==  "Tsinghua National Laboratory for Information Science and Technology":
            aff_total[i] = "Tsinghua University"
        #  模糊匹配
        if 'IBM' in aff_total[i]:
            aff_total[i] = 'IBM'
        if 'AMD' in aff_total[i]:
            aff_total[i] = 'AMD'
        if 'Google' in aff_total[i]:
            aff_total[i] = 'Google'
        if 'Intel' in aff_total[i]:
            aff_total[i] = 'Intel'
        if "Facebook" in aff_total[i]:
            aff_total[i] = "Facebook"
        if 'Peking' in aff_total[i]:
            aff_total[i] = 'Peking University'
        if 'Alibaba' in aff_total[i]:
            aff_total[i] = 'Alibaba'
def filter_department(aff_item):
    for index in range(len(aff_item)):
        if aff_item[index].split(',')[0] not in ["Department of Computer Science",
                "Department of Electrical and Computer Engineering",
                "Department of Computer Science and Engineering",
                'School of Computing',
                'Dept. of Computer Science','Qatar Computing Research Institute']:  # 剔除学院
            aff_item[index] = aff_item[index].split(',')[0]
        else:
            #print(aff_item[index])
            if len(aff_item[index].split(','))>1:
                aff_item[index] = aff_item[index].split(',')[1].strip(' ')
            else:
                aff_item[index] = aff_item[index].split(',')[0].strip(' ')



def parse_affiliation():
    df = pd.read_excel("allData.xls")
    origin_aff_list = np.array(df['Author affiliation'])
    aff_total = []
    for aff_item in origin_aff_list:
        #  分词
        tmp_list = aff_participle(aff_item)
        #  去除二级机构
        filter_department(tmp_list)
        #  映射
        aff_map(tmp_list)
        #  去重
        tmp_list = list(set(tmp_list))
        aff_total += tmp_list



    result = []
    top_aff_20 ={}
    index = 1
    #  统计频次
    for aff in counter(aff_total,21)[1:]:
        top_aff_20[aff[0]] = aff[1]
    for key in top_aff_20.keys():
        tmp_dict = {}
        tmp_dict["af_id"] = index
        tmp_dict["af_name"] = key
        tmp_dict["number_of_papers"] = top_aff_20.get(key)
        tmp_dict["rank_of_papers"] = index
        index+=1
        result.append(tmp_dict)
    # with open('10_top_affiliations.json', 'w') as f:
    #     json.dump(result, f)
    return result

def parse_country_distribution():
    df = pd.read_excel("allData.xls")
    origin_aff_list = np.array(df['Author affiliation'])
    contry_total= []
    for aff_item in origin_aff_list:
        #  分词
        tmp_list = aff_participle(aff_item)

        for index in range(len(tmp_list)):
            # 提取国家
            tmp_list[index] = tmp_list[index].split(',')[-1].strip(' ')
            if tmp_list[index] in ['Microsoft Research']:
                tmp_list[index] = 'United States'
            if tmp_list[index] in ['Hong Kong']:
                tmp_list[index] = 'China'
        # 去重
        tmp_list = list(set(tmp_list))
        contry_total += tmp_list
    result = []
    top_country_20 = {}
    index = 1
    #  统计频次
    for aff in counter(contry_total,100):
        if aff[0] == '':
            continue
        top_country_20[aff[0]] = aff[1]
    #  写入临时字典
    for key in top_country_20.keys():
        tmp_dict = {}
        tmp_dict["name"] = key
        tmp_dict["value"] = top_country_20.get(key)
        tmp_dict["rank_of_country"] = index
        index+=1
        result.append(tmp_dict)
    with open('10_country_distribution.json', 'w') as f:
        json.dump(result, f)

def parse_au_af():
    df = pd.read_excel("allData.xls")
    au_res = parse_auth()
    af_res = parse_affiliation()

    origin_auth_list = np.array(df['Author'])
    origin_aff_list = np.array(df['Author affiliation'])
    au_af_list = []

    for index in range(len(origin_aff_list)):
        #  处理nan情况
        if type(origin_auth_list[index]) != float:
            auth_tmp_list = [item.strip(' ') for item in origin_auth_list[index].split(';')]
        else:
            auth_tmp_list = []
        if type(origin_aff_list[index]) != float:
            aff_tmp_list = [item.strip(' ') for item in origin_aff_list[index].split('(')]
        else:
            aff_tmp_list = []
        #  通过数字建立作者和机构的对照关系
        if len(auth_tmp_list)>0 and len(aff_tmp_list)>0:
            for i in range(len(auth_tmp_list)):
                for j in range(len(aff_tmp_list)):
                    if len(auth_tmp_list[i])>2 and len(aff_tmp_list[j])>2 and  auth_tmp_list[i][-2] == aff_tmp_list[j][0]:
                        au_af_map= {}
                        au_af_map[auth_tmp_list[i]] = aff_tmp_list[j]
                        au_af_list.append(au_af_map)
    result = []
    for tmp_map in au_af_list:
        pattern = r'[(\d)]'
        auth = [key for key in tmp_map.keys()][0]
        #  对作者编号处理
        auth_split = re.split(pattern, auth)[0].strip(' ')
        tmp_map["au_name"] = auth_split
        for au_item in au_res:
            if auth_split == au_item["au_name"]:
                tmp_map["au_id"] = au_item["au_id"]
        #  对机构编号处理
        affiliation = []
        affiliation.append([value for value in tmp_map.values()][0][2:].strip(' '))
        filter_department(affiliation)
        aff_map(affiliation)
        tmp_map["af_name"] = affiliation[0]
        for aff_item in af_res:
            if affiliation[0] == aff_item["af_name"]:
                tmp_map["af_id"] = aff_item["af_id"]
        tmp_map["value"] = 1
        tmp_map.pop(auth)
        result.append(tmp_map)
    print(result)
    with open('10_au_af_relations.json', 'w') as f:
        json.dump(result, f)

def parse_keywords():
    df = pd.read_excel("Classification code.xls",sheet_name='Classification code筛选')
    origin_class_list = np.array(df['Classification'])
    result = []
    for kw in origin_class_list:
        if '.' not in kw and 'Systems Science' not in kw:
            continue
        print(kw)
        kw_split = re.split(r'\d+',kw)
        for item in kw_split:
            if item not in ['.','']:
                res = item.strip(' ')
                result.append(res)
    result.sort()
    with open('10_keywords.json', 'w') as f:
        json.dump(result, f)

def parse_top_keywords_list():
    df = pd.read_excel("Classification code.xls", sheet_name='Classification code筛选')
    origin_class_list = np.array(df['Classification'])
    result = []
    id = 0
    index = 0
    tmp_dict = {}
    tmp_kws = []
    while index <len(origin_class_list):
        #  遇到不含.的字段认为是大类，重新维护一个list
        if '.' not in origin_class_list[index]:
            result.append(tmp_dict)
            tmp_dict = {}
            tmp_kws = []
            tmp_dict["tp_id"] = id
            id+=1
            #  以数字规则进行正则表达式分词
            kw_split = re.split(r'\d+', origin_class_list[index])
            for item in kw_split:
                if item not in ['.', '']:
                    res = item.strip(' ')
                    print(res)
                    tmp_dict["tp_name"] = res
                    tmp_kws.append(res)
            tmp_dict["kw"] = tmp_kws
            index += 1
        else:
            kw_split = re.split(r'\d+', origin_class_list[index])
            for item in kw_split:
                if item not in ['.', '']:
                    res = item.strip(' ')
                    tmp_kws.append(res)
            index += 1
    result.pop(0)
    for item in result:
        item["kw"].remove(item["kw"][0])
    result.append(tmp_dict)
    with open('10_topic_keyword_list.json', 'w') as f:
        json.dump(result, f)

def parse_top_papers():
    df = pd.read_excel("allData.xls")
    au_res = parse_auth()
    af_res = parse_affiliation()
    #  遍历原始数据，填写进去所有的字段
    origin_title_list = np.array(df['Title'])
    origin_abstract_list = np.array(df['Abstract'])
    origin_year_list = np.array(df['Publication year'])
    origin_auth_list = np.array(df['Author'])
    origin_aff_list = np.array(df['Author affiliation'])
    origin_class_list = np.array(df['Classification code'])
    #  填补缺失值
    for index in range(len(origin_class_list)):
        if type(origin_class_list[index]) == float:
            origin_class_list[index] = ''
    result = []
    id = 0

    with open("5_topic_keyword_list(1).json", 'r') as file:
        load_list = json.load(file)
    for index in range(len(origin_title_list)):
        tmp_dict = {}
        tmp_dict["title"] = origin_title_list[index]
        tmp_dict["abstract"] = origin_abstract_list[index]
        tmp_dict["id"] = id
        id+=1
        tmp_dict["year"] = origin_year_list[index]
        authors = []
        if type(origin_auth_list[index]) != float:
            auth_tmp_list = [item.strip(' ') for item in origin_auth_list[index].split(';')]
        else:
            auth_tmp_list = []
        if type(origin_aff_list[index]) != float:
            aff_tmp_list = [item.strip(' ') for item in origin_aff_list[index].split('(')]
        else:
            aff_tmp_list = []
        if len(auth_tmp_list)>0 and len(aff_tmp_list)>0:
            for i in range(len(auth_tmp_list)):
                for j in range(len(aff_tmp_list)):
                    if len(auth_tmp_list[i])>2 and len(aff_tmp_list[j])>2 and  auth_tmp_list[i][-2] == aff_tmp_list[j][0]:
                        auth_list_dict = {}
                        for au_item in au_res:
                            if auth_tmp_list[i].split('(')[0].strip(' ') == au_item["au_name"]:
                                auth_list_dict["au_id"] = au_item["au_id"]
                                auth_list_dict["au_name"] = au_item["au_name"]

                        af_list = []
                        af_list_dict = {}
                        affiliation = []
                        af_list_dict["af_country"] = aff_tmp_list[j].split(',')[-1].strip(' ')
                        affiliation.append(aff_tmp_list[j][2:].strip(' '))
                        filter_department(affiliation)
                        aff_map(affiliation)
                        af_list_dict["af_name"] = affiliation[0]
                        for af_item in af_res:
                            if affiliation[0] == af_item["af_name"]:
                                af_list_dict["af_id"] = af_item["af_id"]
                        af_list.append(af_list_dict)
                        auth_list_dict["af_list"] = af_list
                        authors.append(auth_list_dict)
        #print(auth_list)
        tmp_dict["authors"] = authors
        keywords = []
        if type(origin_class_list[index]) != float:

            keywords = origin_class_list[index].split('-')
            
            for kw_index in range(len(keywords)):
                kw_split = re.split(r'\d+', keywords[kw_index])
                for item in kw_split:
                    if item not in ['.', '']:
                        keywords[kw_index] = item.strip(' ')
        tmp_dict["keywords"] = keywords
        topics = 0
        topics_list = []
        for kw in keywords:
            for topic_index in range(len(load_list)):
                if kw in load_list[topic_index]['kw']:
                    topics_list.append(topic_index)
            # for load_kw_item in load_list:
            #     if kw in load_kw_item['kw']:
            #         topics+=1
        # topics_list = [topics]
        topics_list = list(set(topics_list))
        tmp_dict["topics"] = topics_list
        #print(tmp_dict)
        result.append(tmp_dict)
    with open('10_top_papers.json', 'w',encoding='utf-8') as f:
        json.dump(result, f)
    tran.method('10_top_papers.json')

parse_top_papers()




