# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 21:24:19 2019

@author: tomson
"""
from utils import *
from models import classmodel,emomodel
import numpy
import pickle
import torch
import tqdm
import gc
from embedingmericx import evalss,evals
class Config:
    batchsize=200
    useextra=True
    epoch=200
    lr=0.001
    l2=1e-4
    hidden_dim=256
    embedding_dim=200
    clip_grad=5
    emotion_size =50
    emo_size=100
    mode='test'
config=Config()
savename='v1'
#savename='v2'
with open('dataset/'+savename+'.pkl', 'rb') as f:
 result=pickle.load(f)
with open('dataset/'+'emodict'+'.pkl', 'rb') as f:
            result['emovocab']=pickle.load(f)
            tmok=result['emovocab']
file=open('sgns.weibo.bigram-char','r',encoding='utf-8')
file1=file.readlines()[1:]
dicts={}
for line in file1:
    line=line.strip()
    line=line.split()
    dicts[line[0]]=[float(item) for item in line[1:]]
file.close()            
print(result['emovocab'])
trainloader=Dataset(result['train'],result['vocab'],result['emovocab'],result['vocabnoemo'],result['labeldes'],config.batchsize,shuffle=True,useextra=True,maxlength=100000)
evalloader=Dataset(result['eval'],result['vocab'],result['emovocab'],result['vocabnoemo'],result['labeldes'],config.batchsize,shuffle=False,useextra=True,maxlength=2000)
testloader=Dataset(result['test'],result['vocab'],result['emovocab'],result['vocabnoemo'],result['labeldes'],100,shuffle=False,useextra=True,maxlength=2000)
config.vocab_size=len(trainloader.vocabfull)
model=emomodel(config,trainloader.vocabfull,result['emovocab'])
model.cuda()
optimizer=torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.l2)
shows=[]
for batch in tqdm.tqdm(trainloader.getbatches()):
    shows.append([item if type(item)!=torch.Tensor else item.cpu().numpy() for item in batch ])
maxp=0 
maxper=0
bestr=[0]*6
if config.mode!='train':
    model.load_state_dict(torch.load(savename+'-model'+'.pt'))
if config.mode=='train':
    for idx in range(config.epoch):
        loss1=0
        content=[]
        right=0
        print('haha')
        print(idx)
        for batch in tqdm.tqdm(trainloader.getbatches()):
            model.train()
            optimizer.zero_grad()
            loss=model(*batch[1:])
            loss=loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            optimizer.step()
            loss1+=loss.cpu().tolist()
    #        for idj in range(len(batch[0])):
    #            content.append([batch[0][idj],att.cpu().data.numpy()[idj]])
    #            if predict.cpu().data.numpy()[idj]==int(batch[0][idj]['emotion']):
    #                right+=1
        print('loss:',loss1/trainloader.epochnum)
        sentences=[]
        for batch in tqdm.tqdm(testloader.getbatches()):
            model.eval()
            output=model.test(*batch[1:])
            for i in range(len(output)):
                sentences.append([batch[0][i]['input'],output[i],batch[0][i]['emotion'],batch[0][i]['output']])
        print(sentences[:5])
        rouge,bleu4,bleu2,epoch_average, epoch_extrema, epoch_greedy,rouges,bleu4s,bleu2s=evals([item[1] for item in sentences],[item[3] for item in sentences], dicts)
        if maxp<bleu4:
            maxp=bleu4
        cur=[rouge,bleu4,bleu2,epoch_average, epoch_extrema, epoch_greedy]    
        bestr=[max([item,item1]) for item,item1 in zip(bestr,cur)]
        if maxper<0.5*(rouge+bleu2):
            maxper=0.5*(rouge+bleu2)
            
            print('save model')
            torch.save(model.state_dict(),savename+'-model'+'.pt')
        print('best belu4', maxp)
        print('best result', bestr)
        print('cur result',rouge,bleu4,bleu2,epoch_average, epoch_extrema, epoch_greedy)
else:
        sentences=[]
        for batch in tqdm.tqdm(testloader.getbatches()):
            model.eval()
            output=model.test(*batch[1:])
            for i in range(len(output)):
                sentences.append([batch[0][i]['input'],output[i],batch[0][i]['emotion'],batch[0][i]['output']])
        print(sentences[:5])
        rouge,bleu4,bleu2,epoch_average, epoch_extrema, epoch_greedy,rouges,bleu4s,bleu2s=evals([item[1] for item in sentences],[item[3] for item in sentences], dicts)
        print('cur result',rouge,bleu4,bleu2,epoch_average, epoch_extrema, epoch_greedy)
        sentencess=[]
        for idx,item in enumerate(sentences):
            sentencess.append(item+[rouges[idx]])
        sentencess=sorted(sentencess,key=lambda x:x[4],reverse=True)
#    print('acc:',right/len(trainloader.data))
#    emotionlist=[dict([(key,[0,0.0000001]) for key,value in trainloader.vocabfull.items()]) for _ in range(5)]
#    newdicts=[]
#    tmpss=[]
##    print('gh')
#    for item in content:
#        if '喜欢' in item[0]['output'] and item[0]['emotion']==3:
#            print(item[0]['output'])
#            print(item[1])
#            a=input('ghh')
#    if(idx>8):
#        for item in content:
#            if item[0]['emotion']>0:
#                for idxx,word in enumerate(item[0]['output']):
#                    if word in emotionlist[item[0]['emotion']-1]:
#                        if idxx>=1:
#                            if item[0]['output'][idxx-1]!='不':
#                                emotionlist[item[0]['emotion']-1][word][0]+=item[1][idxx]
#                                emotionlist[item[0]['emotion']-1][word][1]+=1
#                            else:
#                                emotionlist[item[0]['emotion']-1][word][0]-=item[1][idxx]
#                                emotionlist[item[0]['emotion']-1][word][1]+=1
#                        else:
#                                emotionlist[item[0]['emotion']-1][word][0]+=item[1][idxx]
#                                emotionlist[item[0]['emotion']-1][word][1]+=1 
#        for dicts in emotionlist:
#            maxvalue=max([value[1] for key,value in dicts.items()])
#            newdicts.append(dict([(key,value[0]/max([len(trainloader.data)/120,value[1]])) for key,value in dicts.items()])) 
#        for idj,dicts in enumerate(newdicts):
#            tmp=sorted(dicts.items(),key=lambda x:x[1],reverse=True)
#            tmpss.append([item[0] for item in tmp])
#            tmp=[item for item in tmp if len(item[0])>=2]
#            print(idj,[item[0] for item in tmp[:30]])
#        for item in content[:20]:
#            if item[0]['emotion']>0:
#                print(item[0]['output'])
#                print(item[1])
#                print(item[0]['emotion'])
##        a=input('hahha')
#        del content
#        break
#    else:
#        print('ds')
#        
#        content=[]
#        gc.collect()
#        print('hj')
#        pass
    