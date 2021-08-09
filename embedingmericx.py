# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 23:41:09 2019

@author: tomson
"""

import numpy as np
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.bleu.bleu import Bleu
from embedding_metric import embedding_metric
#file1=open('sgns.weibo.bigram-char','r',encoding='utf-8').readlines()[1:]
#file2=open('sgns.weibo.word','r',encoding='utf-8').readlines()[1:]
#dicts={}
#for line in file1:
#    line=line.strip()
#    line=line.split()
#    dicts[line[0]]=[float(item) for item in line[1:]]
def embedding_metrics(samples,ground_truth,word2vec):
        keys = word2vec
#        print([[s for s in sent.split() if s in keys] for sent in samples])
#        print([[s for s in sent.split() if s in keys] for sent in ground_truth])
        samples = [[word2vec[s] for s in sent.split() if s in keys] for sent in samples]
        ground_truth = [[word2vec[s] for s in sent.split() if s in keys] for sent in ground_truth]

        indices = [i for i, s, g in zip(range(len(samples)), samples, ground_truth) if s != [] and g != []]
        
        samples = [samples[i] for i in indices]
        ground_truth = [ground_truth[i] for i in indices]
#        n = len(samples)
#        n_sent += n
        if len(indices)==0:
            return 0,0,0
        metric_average = embedding_metric(samples, ground_truth, word2vec, 'average')
        metric_extrema = embedding_metric(samples, ground_truth, word2vec, 'extrema')
        metric_greedy = embedding_metric(samples, ground_truth, word2vec, 'greedy')
#        epoch_average, epoch_extrema, epoch_greedy=epoch_average, epoch_extrema, epoch_greedy
#        metric_average_history.append(metric_average)
#        metric_extrema_history.append(metric_extrema)
#        metric_greedy_history.append(metric_greedy)

        epoch_average = np.mean(metric_average, axis=0)
        epoch_extrema = np.mean(metric_extrema, axis=0)
        epoch_greedy = np.mean(metric_greedy, axis=0)

#        print('n_sentences:', n_sent)
#        print_str = f'Metrics - Average: {epoch_average:.3f}, Extrema: {epoch_extrema:.3f}, Greedy: {epoch_greedy:.3f}'
#        print(print_str)
#        print('\n')

        return epoch_average, epoch_extrema, epoch_greedy
def evals(predicts,goldens,dicts,path='sgns.weibo.bigram-char'):
    predicts=[item if len(item)>0 else ['是'] for item in predicts]
    goldens=[item if len(item)>0 else ['是'] for item in goldens]
    def process(wordlist):
        words=''.join(wordlist)
        tmp=''
        for word in words:
            tmp+=word+' '
        tmp=tmp[:-1]
        return tmp
    def process1(wordlist):
        words=' '.join(wordlist)
        return words
    predictdict=dict([[idx,[process(item)]] for idx,item in enumerate(predicts)])
    goldendict=dict([[idx,[process(item)]] for idx,item in enumerate(goldens)])
    predictnew=[process1(item) for idx,item in enumerate(predicts)]
    goldennew=[process1(item) for idx,item in enumerate(goldens)]
#    print(predictdict)
#    print(goldendict)
#    print(predictnew)
#    print(goldennew)
    rouge=Rouge()
    rouge,rouges=rouge.compute_score(goldendict,predictdict)
    bleu=Bleu()
    bleu4t,bleu4st=bleu.compute_score(goldendict,predictdict)
    bleu4=bleu4t[3]
    bleu4s=bleu4st[3]
    bleu2=bleu4t[1]
    bleu2s=bleu4st[1]
#    file=open('sgns.weibo.bigram-char','r',encoding='utf-8')
#    file1=file.readlines()[1:]
#    dicts={}
#    for line in file1:
#        line=line.strip()
#        line=line.split()
#        dicts[line[0]]=[float(item) for item in line[1:]]
#    file.close()
    epoch_average, epoch_extrema, epoch_greedy=embedding_metrics(predictnew,goldennew,dicts)
    return rouge,bleu4,bleu2,epoch_average, epoch_extrema, epoch_greedy,rouges,bleu4s,bleu2s
def evalss(predicts,goldens,path='sgns.weibo.bigram-char'):
    predicts=[item if len(item)>0 else ['是'] for item in predicts]
    goldens=[item if len(item)>0 else ['是'] for item in goldens]
    def process(wordlist):
        words=''.join(wordlist)
        tmp=''
        for word in words:
            tmp+=word+' '
        tmp=tmp[:-1]
        return tmp
    def process1(wordlist):
        words=' '.join(wordlist)
        return words
    predictdict=dict([[idx,[process(item)]] for idx,item in enumerate(predicts)])
    goldendict=dict([[idx,[process(item)]] for idx,item in enumerate(goldens)])
    predictnew=[process1(item) for idx,item in enumerate(predicts)]
    goldennew=[process1(item) for idx,item in enumerate(goldens)]
#    print(predictdict)
#    print(goldendict)
#    print(predictnew)
#    print(goldennew)
    rouge=Rouge()
    rouge,rouges=rouge.compute_score(goldendict,predictdict)
    bleu=Bleu()
    bleu4,bleu4s=bleu.compute_score(goldendict,predictdict)
    bleu4=bleu4[3]
    bleu4s=bleu4s[3]

    return rouge,bleu4,rouges,bleu4s
#predicts=[['我','喜欢','中国'],['你','书画','很','棒']]
#goldens=[['我','讨厌','中国'],['你','书画','很','棒']]
#print(evals(predicts,goldens))