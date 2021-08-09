# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 21:36:44 2019

@author: tomson
"""
from encoder import ContextRNN,length2mask
import torch
from structure_generator.EmoDecoder import EmoDecoderRNN
import copy
class classmodel(torch.nn.Module):
    def __init__(self,config):
        """Base RNN Encoder Class"""
        super(classmodel, self).__init__()
        self.emb=torch.nn.Embedding(config.vocab_size,config.embedding_dim)
        self.drop=torch.nn.Dropout(0.3)
        self.inner=torch.nn.Linear(2*config.hidden_dim,2*config.hidden_dim,bias=False)
        self.encoder=ContextRNN(config.embedding_dim,config.hidden_dim)
        self.out=torch.nn.Linear(4*config.hidden_dim,6)
        self.loss=torch.nn.CrossEntropyLoss()
    def forward(self,outputs,length,emotions):
        outputs=self.drop(self.emb(outputs[:,1:]))
        outputs,hid=self.encoder(outputs,length)#b,s,2h
        hid=hid[0].transpose(0,1).reshape(outputs.size(0),-1)#b,2h
        mask=length2mask(length-1,outputs.size(1))
        logits=torch.matmul(self.inner(hid.unsqueeze(1)),outputs.transpose(1,2))#b,1,s
        logits=logits+(1-mask.unsqueeze(1))*-1e20
        att=torch.softmax(logits,-1)#b,1,s
#        att=torch.sigmoid(logits)#
        context=torch.matmul(att,outputs).squeeze(1)
        self.outlogits=self.out(torch.cat([context,hid],-1))
        loss=self.loss(self.outlogits,emotions)
        return loss,att.squeeze(1),self.outlogits.argmax(-1)
class emomodel(torch.nn.Module):
        def __init__(self,config,vocab,emotion_dict):
            super(emomodel, self).__init__()
            self.vocab=vocab
            self.emb=torch.nn.Embedding(config.vocab_size,config.embedding_dim)
            self.drop=torch.nn.Dropout(0.3)
            self.encoder=ContextRNN(config.embedding_dim,config.hidden_dim)
            self.decoder=EmoDecoderRNN(config.hidden_dim*2, config.emotion_size, config.emo_size, config.embedding_dim,vocab, emotion_dict)
        def forward(self,realinputs,inputlength,realoutputs,outputlength,realemotions,emotion_list,extravocablist):
            sourceids=copy.deepcopy(realinputs)
            realinputs[realinputs>=len(self.vocab)]=1
            outputs=self.drop(self.emb(realinputs))
            outputs,hid=self.encoder(outputs,inputlength)#b,s,2h
            hid=hid[0].transpose(0,1).reshape(outputs.size(0),-1)#b,2h
            return self.decoder(hid, realemotions, realoutputs,outputlength, emotion_list, outputs, sourceids, inputlength-1,extravocablist)
        def test(self,realinputs,inputlength,realoutputs,outputlength,realemotions,emotion_list,extravocablist):
            sourceids=copy.deepcopy(realinputs)
            realinputs[realinputs>=len(self.vocab)]=1
            outputs=self.drop(self.emb(realinputs))
            outputs,hid=self.encoder(outputs,inputlength)#b,s,2h
            hid=hid[0].transpose(0,1).reshape(outputs.size(0),-1)#b,2h
            return self.decoder.test(hid, realemotions, emotion_list, outputs, sourceids, inputlength-1,extravocablist)
        def testbeam(self,realinputs,inputlength,realoutputs,outputlength,realemotions,emotion_list,extravocablist):
            sourceids=copy.deepcopy(realinputs)
            realinputs[realinputs>=len(self.vocab)]=1
            outputs=self.drop(self.emb(realinputs))
            outputs,hid=self.encoder(outputs,inputlength)#b,s,2h
            hid=hid[0].transpose(0,1).reshape(outputs.size(0),-1)#b,2h
            return self.decoder.testbeam(hid, realemotions, emotion_list, outputs, sourceids, inputlength-1,extravocablist)
        