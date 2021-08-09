# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 21:24:19 2019

@author: tomson
"""
from utils import *
from models import classmodel
import numpy
import pickle
import torch
import tqdm
import gc
class Config:
    batchsize=200
    useextra=True
    epoch=50
    lr=0.001
    l2=1e-4
    hidden_dim=256
    embedding_dim=200
    clip_grad=5
config=Config()
savename='v1'
#savename='v2'
with open('dataset/'+savename+'.pkl', 'rb') as f:
 result=pickle.load(f)
trainloader=Dataset(result['train'],result['vocab'],result['emovocab'],result['vocabnoemo'],result['labeldes'],config.batchsize,shuffle=False,useextra=False,maxlength=200000)
#evalloader=Dataset(result['eval'],result['vocab'],result['emovocab'],result['vocabnoemo'],result['labeldes'],config.batchsize,shuffle=False,useextra=False)
#testloader=Dataset(result['test'],result['vocab'],result['emovocab'],result['vocabnoemo'],result['labeldes'],config.batchsize,shuffle=False,useextra=False)
config.vocab_size=len(trainloader.vocabfull)
model=classmodel(config)
model.cuda()
optimizer=torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.l2)
for idx in range(config.epoch):
    loss1=0
    content=[]
    right=0
    print('haha')
    for batch in tqdm.tqdm(trainloader.getbatches()):
        model.train()
        optimizer.zero_grad()
        loss,att,predict=model(batch[3],batch[4],batch[5])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optimizer.step()
        loss1+=loss.cpu().tolist()
        for idj in range(len(batch[0])):
            content.append([batch[0][idj],att.cpu().data.numpy()[idj]])
            if predict.cpu().data.numpy()[idj]==int(batch[0][idj]['emotion']):
                right+=1
    print('loss:',loss1/trainloader.epochnum)
    print('acc:',right/len(trainloader.data))
    emotionlist=[dict([(key,[0,0.0000001]) for key,value in trainloader.vocabfull.items()]) for _ in range(5)]
    newdicts=[]
    emodicts={}
    emodicts[0]={'<other>':0}
    tmpss=[]
#    print('gh')
#    for item in content:
#        if '喜欢' in item[0]['output'] and item[0]['emotion']==3:
#            print(item[0]['output'])
#            print(item[1])
#            a=input('ghh')
    if(idx>7):
        for item in content:
            if item[0]['emotion']>0:
                for idxx,word in enumerate(item[0]['output']):
                    if word in emotionlist[item[0]['emotion']-1]:
                        if idxx>=1:
                            if item[0]['output'][idxx-1]!='不':
                                emotionlist[item[0]['emotion']-1][word][0]+=item[1][idxx]
                                emotionlist[item[0]['emotion']-1][word][1]+=1
                            else:
                                emotionlist[item[0]['emotion']-1][word][0]-=item[1][idxx]
                                emotionlist[item[0]['emotion']-1][word][1]+=1
                        else:
                                emotionlist[item[0]['emotion']-1][word][0]+=item[1][idxx]
                                emotionlist[item[0]['emotion']-1][word][1]+=1 
        for dicts in emotionlist:
            maxvalue=max([value[1] for key,value in dicts.items()])
            newdicts.append(dict([(key,value[0]/max([len(trainloader.data)/120,value[1]])) for key,value in dicts.items()])) 
        for idj,dicts in enumerate(newdicts):
            tmp=sorted(dicts.items(),key=lambda x:x[1],reverse=True)
            tmpss.append([item[0] for item in tmp])
            tmpq=[item for item in tmp if len(item[0])>=2]
            tps=[item[0] for item in tmp if len(item[0])>=1][:50]
            emodicts[idj+1]=dict([(word,idx) for idx,word in enumerate(tps)])
            print(idj,[item[0] for item in tmpq[:30]])
        for item in content[:20]:
            if item[0]['emotion']>0:
                print(item[0]['output'])
                print(item[1])
                print(item[0]['emotion'])
#        a=input('hahha')
        with open('dataset/'+'emodict'+'.pkl', 'wb') as f:
            pickle.dump(emodicts,f)
        del content
        break
#    else:
#        print('ds')
#        
#        content=[]
#        gc.collect()
#        print('hj')
#        pass
    