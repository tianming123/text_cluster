import copy
import json

def method(path):
    with open(path,'r') as file1:
        load_dict1 = json.load(file1)
    with open('6_top_papers.json','r') as file2:
        load_dict2 = json.load(file2)
    with open('6_top_papers.json','r') as file2:
        load_dict3 = json.load(file2)
    print(load_dict1[0]['authors'])
    print(len(load_dict1[0]['authors']))
    print(load_dict2[0]['authors'])

    for num in range(len(load_dict2)):
        load_dict2[num]['title'] = load_dict1[num]['title']
        load_dict2[num]['abstract'] = "   "
        load_dict2[num]['year'] = load_dict1[num]['year']
        load_dict2[num]['keywords'] = load_dict1[num]['keywords']
        load_dict2[num]['topics'] = load_dict1[num]['topics']
        au_list = []

        for index in range(len(load_dict1[num]['authors'])):
            tmp = copy.deepcopy(load_dict2[num]['authors'][0])
            tmp['au_id'] = load_dict1[num]['authors'][index]['au_id']
            tmp['au_name'] = load_dict1[num]['authors'][index]['au_name']
            tmp_list = [{'af_id': 0, 'af_name': 'Innovat Acad Seed Design CAS', 'af_country': 'China'}]
            if 'af_id' in load_dict1[num]['authors'][index]['af_list'][0].keys():
                tmp_list[0]['af_id'] = load_dict1[num]['authors'][index]['af_list'][0]['af_id']
            else:
                tmp_list[0]['af_id'] = -1
            tmp_list[0]['af_name'] = load_dict1[num]['authors'][index]['af_list'][0]['af_name']
            tmp_list[0]['af_country'] = load_dict1[num]['authors'][index]['af_list'][0]['af_country']
            tmp['af_list'] = tmp_list
            au_list.append(tmp)
        load_dict2[num]['authors'] = au_list
    print(len(load_dict1))
    print(len(load_dict2))
    print(len(load_dict3))

    for num in range(1570):
        num2 = num+1730
        load_dict3[num]["id"] = load_dict1[num2]["id"]
        load_dict3[num]['title'] = load_dict1[num2]['title']
        load_dict3[num]['abstract'] = "   "
        load_dict3[num]['year'] = load_dict1[num2]['year']
        load_dict3[num]['keywords'] = load_dict1[num2]['keywords']
        load_dict3[num]['topics'] = load_dict1[num2]['topics']
        au_list = []
        for index in range(len(load_dict1[num2]['authors'])):
            tmp = copy.deepcopy(load_dict3[num]['authors'][0])
            tmp['au_id'] = load_dict1[num2]['authors'][index]['au_id']
            tmp['au_name'] = load_dict1[num2]['authors'][index]['au_name']
            tmp_list = [{'af_id': 0, 'af_name': 'Innovat Acad Seed Design CAS', 'af_country': 'China'}]
            if 'af_id' in load_dict1[num2]['authors'][index]['af_list'][0].keys():

                tmp_list[0]['af_id'] = load_dict1[num2]['authors'][index]['af_list'][0]['af_id']
            else:
                tmp_list[0]['af_id'] = -1
            tmp_list[0]['af_name'] = load_dict1[num2]['authors'][index]['af_list'][0]['af_name']
            tmp_list[0]['af_country'] = load_dict1[num2]['authors'][index]['af_list'][0]['af_country']
            tmp['af_list'] = tmp_list
            au_list.append(tmp)
        load_dict3[num]['authors'] = au_list
    res = load_dict2+load_dict3[0:1571]
    with open("record.json","w") as dump_f:
        json.dump(res,dump_f)
