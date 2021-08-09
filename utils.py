# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 20:29:01 2019

@author: tomson
"""
import random
import math
import torch
import copy
class Dataset:
    def __init__(self,dataset,vocabfull,emotionvocab,vocabnoemo,describ,batchsize,shuffle=True,useextra=False,maxlength=None):
        self.data=copy.deepcopy(dataset)[:maxlength]
        self.shuffle=shuffle
        self.useextra=useextra
        self.batchsize=batchsize
        self.vocabfull=vocabfull
        self.emotionvocab=emotionvocab
#        self.emotionvocab={0:{'<other>':0},1:{'<other>':0},2:{'<other>':0},3:{'<other>':0},4:{'<other>':0},5:{'<other>':0}}
        self.vocabnoemo=vocabnoemo
        self.describ=describ
        self.emo_max=max([len(value) for key,value in emotionvocab.items()])
    def getbatches(self):
        if self.shuffle:
            random.shuffle(self.data)
        self.epochnum=math.ceil(len(self.data)/self.batchsize)
        for i in range(self.epochnum):
            yield self.construct(self.data[i*self.batchsize:(i+1)*self.batchsize])
    def construct(self,examples):
        inputs=[item['input'] for item in examples]
        emotions=[item['emotion'] for item in examples]
        outputs=[item['output'] for item in examples]
        inputlength=[len(item)+1 for item in inputs]
        outputlength=[len(item)+1 for item in outputs]
        maxinputlen=max(inputlength)
        maxoutputlen=max(outputlength)
        realinputs=[]
        realoutputs=[]
        emotionlist=[]
        extravocablist=[]
        for item in emotions:
            emotionlist.append(self.emotionvocab[item])
        def complexfunc(word,ids,idss):
            if word in self.emotionvocab[ids]:
                return self.emotionvocab[ids][word]+len(self.vocabfull)
            if word in extravocablist[idss]:
                return extravocablist[idss][word]+len(self.vocabfull)+self.emo_max
            return self.vocabfull.get(word,1)
        def complexfunc1(word,ids):
            if word in extravocablist[ids]:
                return extravocablist[ids][word]+len(self.vocabfull)+self.emo_max
            return self.vocabfull.get(word,1)
        for item in inputs:
            extravocab={}
            for ite in item:
                if ite not in self.vocabfull and ite not in extravocab:
                    extravocab[ite]=len(extravocab)
            extravocablist.append(extravocab)
        if not self.useextra:
            for item in inputs:
                wordids=[self.vocabfull.get(word,1) for word in item]+[3]
                realinputs.append(wordids+(maxinputlen-len(wordids))*[0]) 
            for item in outputs:
                wordids=[2]+[self.vocabfull.get(word,1) for word in item]+[3]
                realoutputs.append(wordids+(maxoutputlen+1-len(wordids))*[0]) 
        else:
            for idx,item in enumerate(inputs):
                wordids=[complexfunc1(word,idx) for word in item]+[3]
                realinputs.append(wordids+(maxinputlen-len(wordids))*[0]) 
            for idx,item in enumerate(outputs):
                wordids=[2]+[complexfunc(word,emotions[idx],idx) for word in item]+[3]
                realoutputs.append(wordids+(maxoutputlen+1-len(wordids))*[0]) 
            pass
        realinputs=torch.Tensor(realinputs).long().cuda()
        realoutputs=torch.Tensor(realoutputs).long().cuda()
        inputlength=torch.Tensor(inputlength).long().cuda()
        outputlength=torch.Tensor(outputlength).long().cuda()
        realemotions=torch.Tensor(emotions).long().cuda()
        return examples,realinputs,inputlength,realoutputs,outputlength,realemotions,emotionlist,extravocablist