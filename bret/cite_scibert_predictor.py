
#import pandas as pd
#import numpy as np
import json,time,codecs
#from  tqdm import tqdm
from sklearn.metrics import accuracy_score,classification_report
import torch
import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader,RandomSampler,SequentialSampler
from transformers import BertConfig,AdamW,get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import jsonlines as jsonl
#参数
bert_path = '/gpfs1/home/nc/chaixq/cite/huggingface/untitled/scibert_input'   #预训练模型的位置
tokenizer = AutoTokenizer.from_pretrained(bert_path)   #初始化分词器
max_len = 60     #数据阻断长度
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10


#1.1处理数据成input_ids,token_type_ids,attention_mask,label
def dataSet(data_path):
    input_all = []
    labels = []
    with open(data_path,encoding='utf-8') as f:
        for item in jsonl.Reader(f):
            a = item["string"]
            a.replace("\r\n", " ")

            if item["label"] == "background":
                c = 0
            elif item["label"] == "method":
                c = 1
            else:
                c = 2
            #调用tokenizer转换成bert需要的数据格式
            encode_dict = tokenizer.encode(text=a.strip(),max_length=max_len,padding='max_length',
                                                return_tensors = 'pt',truncation=True)
            #分别获取三个值  目前值的类型为list
            input_all.append(encode_dict)
            #token_type_ids.append(encode_dict['token_type_ids'])
            #attention_mask.append(encode_dict['attention_mask'])

            labels.append(int(c))
            #print("labels:",type(labels))
    labels = torch.tensor(labels)
    #list转化成tensor格式
    #input_ids,token_type_ids,attention_mask = torch.tensor(input_ids),torch.tensor(token_type_ids),torch.tensor(attention_mask)
    input_all = torch.cat(input_all,dim=0)
    return input_all,labels

#1.2 dataloder批量处理
def dataLoader(input_all,labels):
    #tensor数据整合
    #labels = torch.tensor(labels)
    print("input_all:",input_all.shape)
    print("label:",labels.shape)
    data = TensorDataset(input_all,labels)
    loader = DataLoader(data,batch_size=BATCH_SIZE,shuffle=True)
    print("loader:",type(loader))#shuffle打乱每行数据的顺序
    return loader
#print("loader:",loader.shape)

#1.3实例化函数
#训练集带label
input_all_train,labels_train = dataSet('/gpfs1/home/nc/chaixq/cite/scicite-master/scicite-master/data/data/train.jsonl')
train_loader = dataLoader(input_all_train,labels_train)
#验证集带label
input_all_dev,labels_dev = dataSet('/gpfs1/home/nc/chaixq/cite/scicite-master/scicite-master/data/data/dev.jsonl')
dev_loader = dataLoader(input_all_dev,labels_dev)
#测试集 没有的话label放到dataloader
# input_ids_test,token_type_ids_test,attention_mask_test,labels_test = dataSet('data/test.txt')
# data = TensorDataset(input_ids_test,token_type_ids_test,attention_mask_test)
# sample = RandomSampler(data) #随机采样
# test_loader = DataLoader(data,sampler=sample,batch_size=BATCH_SIZE)
#测试集
input_all_test,labels_test = dataSet('/gpfs1/home/nc/chaixq/cite/scicite-master/scicite-master/data/data/test.jsonl')
test_loader = dataLoader(input_all_test,labels_test)
#得到后续用的数据为train_loader,dev_loader,test_loader


# class Bert_Model(nn.Module):
#     def __init__(self,bert_path,classes=10):
#         super(Bert_Model,self).__init__()
#         self.config = BertConfig.from_pretrained(bert_path)
#         self.bert = BertModel.from_pretrained(bert_path)
#         for param in self.bert.parameters():
#             param.requires_grad=True
#         self.fc = nn.Linear(self.config.hidden_size,classes)  #直接分类
#     def forward(self,input_ids,token_type_ids,attention_mask):
#         output = self.bert(input_ids,token_type_ids,attention_mask)[1]  #池化后的输出,是向量
#         logit = self.fc(output)    #全连接层,概率矩阵
#         return logit

#实例化bert模型
model = AutoModelForSequenceClassification.from_pretrained(bert_path,num_labels=3, output_attentions=False,
                                                      output_hidden_states=False).to(DEVICE)


#优化器
optimizer = AdamW(model.parameters(),lr=2e-5,weight_decay=1e-4)  #使用Adam优化器
#设置学习率
schedule = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=len(train_loader),num_training_steps=EPOCHS*len(test_loader))


# 在验证集上评估模型性能的函数
def evaluate(model, data_loader, device):
    model.eval()  # 防止模型训练改变权值
    val_true, val_pred = [], []
    with torch.no_grad():  # 计算的结构在计算图中,可以进行梯度反转等操作
        for i,batch in enumerate(data_loader):  # 得到的y要转换一下数据格式
            logits = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0] > 0).to(device),
                                 labels=batch[1].to(device))[1]  # logits 返回的是分类的数值，不是概率
            y_pred = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()  # 将概率矩阵转化成标签值
            val_pred.extend(y_pred)  # 将标签值放入列表
            label_ids = batch[1].to('cpu').numpy()
            val_true.extend(label_ids.tolist())  # 将真实标签转换成list放在列表中

    return accuracy_score(val_true, val_pred)


# 如果是比赛没有labels_test，那么这个函数for里面没有y，输出没有test_true，处理数据的时候没有labels_test放到dataloader里
def predict(model, data_loader, device):
    model.eval()
    test_pred, test_true = [], []
    with torch.no_grad():
        for i,batch in enumerate(data_loader):
            logits = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0] > 0).to(device),
                                 labels=batch[1].to(device))[1]#logits 返回的是分类的数值，不是概率
            y_pred = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()  # 将概率矩阵转化成标签值
            test_pred.extend(y_pred)
            label_ids = batch[1].to('cpu').numpy()
            test_true.extend(label_ids.tolist())
    return test_pred, test_true


# 训练函数
def train_and_eval(model, train_loader, valid_loader, optimizer, schedule, device, epoch):
    best_acc = 0.0
    patience = 0
    criterion = nn.CrossEntropyLoss()  # 损失函数
    for i in range(epoch):
        start = time.time()
        model.train()  # 开始训练
        print("***************我是狗Running training epoch{}************".format(i + 1))
        train_loss_sum = 0.0
        for step,batch in enumerate(train_loader):
            model.zero_grad()
            #print("batch:",batch.shape)
            #print("step:", step)
            #ids, tpe, att, y = ids.to(device), tpe.to(device), att.to(device), y.to(device)
            loss = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0] > 0).to(device),labels=batch[1].to(device))[0]
            #loss= model(ids, att, tpe,labels=3)[0]  # 加载模型获得概率矩阵
            #loss = criterion(y_pred, y)  # 计算损失
            optimizer.zero_grad()  # 梯度清零
            #print("loss:",type(loss))
            loss.backward()  # 反向传播
            optimizer.step()  # 更新优化参数
            schedule.step()  # 更新学习率
            train_loss_sum += loss.item()
            # 只打印五次结果
            if (step + 1) % (len(train_loader) // 5) == 0:
                print("Epoch {:04d} | Step {:04d}/{:04d} | Loss {:.4f} | Time {:.4f}".format(
                    i + 1, step + 1, len(train_loader), train_loss_sum / (step + 1), time.time() - start))
        # 每一次epoch输出一个准确率
        model.eval()
        acc = evaluate(model, valid_loader, device)  # 验证模型的性能
        f = codecs.open("/gpfs1/home/nc/chaixq/cite/huggingface/untitled/cite_scibert_predictor_out/acc.json", "w", "utf-8")
        f.write(json.dumps(acc, indent=4))
        f.close()
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_albert_model.pth")  # 保存最好的模型
        print("current acc is {:.4f},best acc is {:.4f}".format(acc, best_acc))
        print("time costed = {}s \n".format(round(time.time() - start, 5)))

train_and_eval(model, train_loader, dev_loader, optimizer, schedule, DEVICE, EPOCHS)

model.load_state_dict(torch.load("best_albert_model.pth"))
#得到预测标签和真实标签
test_pred,test_true= predict(model,test_loader,DEVICE)
#输出测试机的准确率
print("\n Test Accuracy = {} \n ".format(accuracy_score(test_true,test_pred)))
#打印各项验证指标
Fz=classification_report(test_true,test_pred,digits=4)
print(classification_report(test_true,test_pred,digits=4))
f = codecs.open("/gpfs1/home/nc/chaixq/cite/huggingface/untitled/cite_scibert_predictor_out/F.json", "w", "utf-8")
f.write(json.dumps(Fz, indent=4))
f.close()
print(test_pred[:10])
print('------------------')
print(test_true[:10])

from transformers import pipeline
# from datasets import load_dataset
#
# class Bert_Model(nn.Module):
#     def __init__(self,bert_path,classes=3):
#         super(Bert_Model,self).__init__()
#         self.config = BertConfig.from_pretrained(bert_path)
#         self.bert = BertModel.from_pretrained(bert_path)
#         for param in self.bert.parameters():
#             param.requires_grad=True
#         self.fc = nn.Linear(self.config.hidden_size,classes)  #直接分类
#     def forward(self,input_ids,token_type_ids,attention_mask):
#         output = self.bert(input_ids,token_type_ids,attention_mask)[1]  #池化后的输出,是向量
#         logit = self.fc(output)    #全连接层,概率矩阵
#         return logit
#
# #实例化bert模型
# bert_path = 'bert-base-uncased'   #预训练模型的位置
# tokenizer = BertTokenizer.from_pretrained(bert_path)   #初始化分词器
# model = Bert_Model(bert_path)
# model.load_state_dict(torch.load("E:\\untitled\\best_bert_model.pth"))

#test_dataset = test_dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

import xlsxwriter as xw
i=0
import re
#from datasets import load_dataset
#examples=load_dataset('D:\\linux\\scicite\\scicite\\test.jsonl',split="test[:]")
#print(examples)
workbook = xw.Workbook("/gpfs1/home/nc/chaixq/cite/huggingface/untitled/cite_scibert_predictor_out/test_data.xlsx")
worksheet1 = workbook.add_worksheet('Sheet1')
worksheet1.activate()
worksheet1.activate()
with open('/gpfs1/home/nc/chaixq/cite/scicite-master/scicite-master/data/data/test.jsonl', encoding='utf-8') as f:
    for item in jsonl.Reader(f):
        #numbers=item["citingPaperId"]
        #labels=item["labels"]
        line=item["string"]
        result=classifier(item["string"])
        result=str(result)
        
        s = re.sub('\(.*?\)', '', line)
        # s=re.split(r'[;, .]s*', s)
        count_num = len(re.split(r'[;, .]s*', s))
        # print(s)
        worksheet1.write(i, 0, item["citingPaperId"])
        worksheet1.write(i, 1, item["string"])
        worksheet1.write(i, 2, item["label"])
        worksheet1.write(i, 3, result)
        worksheet1.write(i, 4, count_num)
        i = i + 1
        #print(result)
    workbook.close()
