# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 16:53:03 2019

@author: tomson
"""

import json
import xlrd
import pickle
def report(es):
    print('total',len(es))
    unique={}
    nums=[0]*6
    for item in es:
        inputs=tuple(item['input'])
        if inputs not in unique:
            unique[inputs]=1
        else:
            unique[inputs]+=1
    print('actual post len',len(unique))
    for item in es:
        idx=item['emotion']
        nums[idx]+=1
    print(nums)
    assert sum(nums)==len(es)
maxlength=30
maxvocab=30000
data_path='./train_data.json'
data_path='./stc-3_emotion_train.json'
savename='v1'
savename='v2'
emotiondict=xlrd.open_workbook("emotiondict.xls")
print(emotiondict.sheet_names())
worksheet=emotiondict.sheet_by_index(0)
nrows=worksheet.nrows
tmp=[]
for i in range(1,nrows):
    tmp.append([worksheet.row_values(i)[1],worksheet.row_values(i)[2]])
file=open('projection.txt','r',encoding='utf-8').readlines()
tmps=[]
for line in file:
    tmps.append(line.strip().split(';')[1:])
tmps=tmps[1:]
emotiondict={}
def extract(indexs,tmp):
    tmp1=[]
    for item in tmp:
        if item[1] in indexs:
            tmp1.append(item[0])
    return tmp1
for i in range(1,6):
    emotiondict[i]=list(set(extract(tmps[i-1],tmp)))
emotiondict[0]=['<other>']
emotiondes={}
emotiondes[0]='other'
emotiondes[1]='like'
emotiondes[2]='sadness'
emotiondes[3]='disgust'
emotiondes[4]='anger'
emotiondes[5]='happiness'
files=json.load(open(data_path,'r',encoding='utf-8'))
examples=[]
for item in files:
    examples.append({'input':item[0][0].strip().split(),'emotion':int(str(item[1][1]).strip()),'output':item[1][0].strip().split()})
report(examples)
exampless=[]
for item in examples:
    if len(item['input'])<=maxlength and len(item['output'])<=maxlength and len(item['input'])>=5 and len(item['output'])>=5:
        exampless.append(item)
num=len(exampless)
splitnum1=int(num*0.8)
splitnum2=int(num*0.9)
trainset=exampless[:splitnum1]
evalset=exampless[splitnum1:splitnum2]
testset=exampless[splitnum2:]
allemotion=[]
emotiondicts={}
for key,value in  emotiondict.items():
    allemotion.extend(value)
for key,value in  emotiondict.items():
    emotiondicts[key]=dict([(word,idx) for idx,word in enumerate(value)])
print('train,eval,test num',len(trainset),len(evalset),len(testset))
vocab={}
for item in exampless:
    words=item['input']+item['output']
    for word in words:
        if word in vocab:
            vocab[word]+=1
        else:
            vocab[word]=0
newvocablist=['<pad>','<unk>','<sos>','<eos>']+[item[0] for item in sorted(vocab.items(),key=lambda x:x[1],reverse=True)][:maxvocab]
newvocablists=[item for item in newvocablist if item not in allemotion]
vocab=dict([(value,idx) for idx,value in enumerate(newvocablist)])
vocabnoemo=dict([(value,idx) for idx,value in enumerate(newvocablists)])
revocab=dict([(idx,value) for value,idx in vocab.items()])
result={}
result['train']=trainset
result['eval']=evalset
result['test']=testset
result['labeldes']=emotiondes
result['vocab']=vocab
result['emovocab']=emotiondicts
result['vocabnoemo']=vocabnoemo
with open('dataset/'+savename+'.pkl', 'wb') as f:
 pickle.dump(result,f)