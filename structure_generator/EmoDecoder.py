import torch.nn as nn
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import copy
from .baseRNN import BaseRNN
# self.word2idx['<UNK>'] = 1
def length2mask(length,maxlength):
    size=list(length.size())
    length=length.unsqueeze(-1).repeat(*([1]*len(size)+[maxlength])).long()
    ran=torch.arange(maxlength).cuda()
    ran=ran.expand_as(length)
    mask=ran<length
    return mask.float().cuda()
class EmoDecoderRNN(BaseRNN):
    def __init__(self, hid_size, emotion_size, emo_size, emb_size,vocab, emotion_dict):
        super(EmoDecoderRNN, self).__init__(len(vocab), hid_size, 0.3, 0.3, 1, 'gru')
        self.vocab=vocab
        self.revvocab=dict([(value,key) for key,value in vocab.items()])
        self.emotion_dict=emotion_dict
        self.revemodict={}
        self.emoemb=torch.nn.Embedding(6,emotion_size)
        self.emb=torch.nn.Embedding(len(vocab),emb_size)
        emo_max=max([len(value) for key,value in emotion_dict.items()])
        for key,_ in emotion_dict.items():
            self.revemodict[key]=dict([(value,key) for key,value in self.emotion_dict[key].items()])
        self.lmbda=1.5
        self.alpha=0.5
        self.dropout=torch.nn.Dropout(0.3)
        self.hid_size=hid_size
        self.emo_max=emo_max
        self.emo_size=emo_size
        self.emotion_size=emotion_size
        self.emb_size=emb_size
        self.vocab_size=len(self.vocab)
        self.module_lists=torch.nn.ModuleList([torch.nn.Embedding(self.emo_max,self.emo_size) for _ in range(6)])
        self.rnn=torch.nn.GRUCell(emb_size+emotion_size+emo_size+hid_size,hid_size)
        self.w0=torch.nn.Linear(emotion_size+hid_size,hid_size)
        self.w1=torch.nn.Linear(emo_size+hid_size+1,hid_size)
        self.w2=torch.nn.Linear(hid_size,1)
        self.w11=torch.nn.Linear(hid_size+hid_size+1,hid_size)
        self.w22=torch.nn.Linear(hid_size,1)
        self.w3=torch.nn.Linear(hid_size*2+emo_size,hid_size)
        self.w33=torch.nn.Linear(hid_size,3)
        self.w4=torch.nn.Linear(hid_size*2+emo_size,self.vocab_size)
        self.USE_CUDA=True
        self.classloss=torch.nn.CrossEntropyLoss(reduction='none')
        self.w5=torch.nn.Linear(hid_size,6)
    def step(self, hidden, context, context_ids, coverage, emo_mask, inputs, sourceids, inputlength, coverage1):
        input_mask=length2mask(inputlength,inputs.size(1))
        logits=torch.cat([hidden.unsqueeze(1).repeat(1,context.size(1),1),context,coverage.unsqueeze(-1)],-1)
        logits=self.w2(torch.tanh(self.w1(logits))).squeeze(-1)#b,s
        att=torch.softmax(logits+(1-emo_mask)*-1e20,-1)#b,s
        realatt=att
        fused1=torch.matmul(att.unsqueeze(1),context).squeeze(1)#b,h
        
        logits=torch.cat([hidden.unsqueeze(1).repeat(1,inputs.size(1),1),inputs,coverage1.unsqueeze(-1)],-1)
        logits=self.w22(torch.tanh(self.w11(logits))).squeeze(-1)#b,s
        att=torch.softmax(logits+(1-input_mask)*-1e20,-1)#b,s
        realatt1=att
        fused2=torch.matmul(att.unsqueeze(1),inputs).squeeze(1)#b,h
        
        fused=torch.cat([fused1,fused2,hidden],-1)
        generate=torch.softmax(self.w33(torch.tanh(self.w3(fused))),-1)#b,1
        vocab=generate[:,0].unsqueeze(-1)*torch.softmax(self.w4(fused),-1)
        copy=realatt*generate[:,1].unsqueeze(-1)
        copy1=realatt1*generate[:,2].unsqueeze(-1)
        ext_vocab = torch.zeros(hidden.size(0), self.emo_max+self.max_extra)
        if self.USE_CUDA:
                ext_vocab=ext_vocab.cuda()
        combined_vocab = torch.cat((vocab, ext_vocab), 1)
        combined_vocab = combined_vocab.scatter_add(1, context_ids+self.vocab_size, copy)
        combined_vocab = combined_vocab.scatter_add(1, sourceids, copy1)
        return combined_vocab, realatt,realatt1,[fused1,fused2]
    def forward(self, encoder_output, emotion_inputs, output, ouputlen, emodict_list, inputs,sourceids, inputlength, extravocablist): 
#        oov_max=max([len(item) for item in emodict_list])
        emotion_input=self.emoemb(emotion_inputs)
        encoder_output=self.dropout(torch.tanh(self.w0(torch.cat([encoder_output,emotion_input],-1))))
        emo_len=torch.Tensor([len(item) for item in emodict_list]).long().cuda()
        self.max_extra=max([len(dicts) for dicts in extravocablist])
        emotion=torch.zeros(encoder_output.size(0),self.emo_max,self.emo_size).cuda()
        for i in range(encoder_output.size(0)):
            emotion[i]=self.module_lists[emotion_inputs[i].data.cpu()].weight
        target=output[:,1:]
        out_mask=length2mask(ouputlen,target.size(1))
        emo_mask=length2mask(emo_len,self.emo_max)
        output=copy.deepcopy(output[:,:-1])
        output[output>=self.vocab_size]=1
        output=self.emb(output)
        hidden=encoder_output
        lm_loss=[]
        cov_loss=[]
        cov_loss1=[]
        out_hidden=[]
        batch_size=encoder_output.size(0)
        coverage = torch.zeros(batch_size, self.emo_max).cuda()
        coverage1 = torch.zeros(batch_size, inputs.size(1)).cuda()
        context_ids=torch.arange(self.emo_max).unsqueeze(0).repeat(encoder_output.size(0),1).cuda()
        fused=[torch.zeros(batch_size,self.emo_size).cuda(),torch.zeros(batch_size,inputs.size(2)).cuda()]
        for i in range(output.size(1)):
            fused=[torch.zeros(batch_size,self.emo_size).cuda(),torch.zeros(batch_size,inputs.size(2)).cuda()]
            emb=self.dropout(torch.cat([output[:,i,:],emotion_input]+fused,-1))
            currenttarget=target[:,i].unsqueeze(-1)
            target_mask_0 = out_mask[:,i]
            hidden=self.rnn(emb,hidden)
            combined_vocab, realatt, realatt1,fused=self.step(hidden, emotion, context_ids, coverage, emo_mask, inputs, sourceids, inputlength, coverage1)
#            target_mask_0 = target_id.ne(0).detach()
            outputs = combined_vocab.gather(1, currenttarget).add_(sys.float_info.epsilon).squeeze(-1)
            lm_loss.append(outputs.log().mul(-1) * target_mask_0.float())
            coverage = coverage + realatt
            coverage1 = coverage1 + realatt1

                # Coverage Loss
                # take minimum across both attn_scores and coverage
            _cov_loss, _ = torch.stack((coverage, realatt), 2).min(2)
            _cov_loss1, _ = torch.stack((coverage1, realatt1), 2).min(2)
            cov_loss.append(_cov_loss.sum(1)* target_mask_0.float())
            cov_loss1.append(_cov_loss1.sum(1)* target_mask_0.float())
            out_hidden.append(hidden)
        out_hidden =torch.stack(out_hidden,1)  #b,s,h
        eachlen=output.size(1)
        out_hidden=torch.index_select(out_hidden.reshape(-1,out_hidden.size(2)),dim=0,index=torch.arange(batch_size).cuda()*eachlen+ouputlen-1)#b,h
        out_loss=self.classloss(self.w5(out_hidden),emotion_inputs)
        total_masked_loss = torch.stack(lm_loss, 1).sum(1).div(ouputlen.float()) + self.lmbda * torch.stack(cov_loss, 1).sum(1).div(ouputlen.float())+self.lmbda * torch.stack(cov_loss1, 1).sum(1).div(ouputlen.float())
#+self.lmbda * \
#                torch.stack(cov_loss1, 1).sum(1).div(ouputlen.float())+self.alpha*out_loss
        return total_masked_loss
#            steploss=torch.gather(combined_vocab,1,currenttarget.unsqueeze(-1)).add_(sys.float_info.epsilon).squeeze(-1)
    def testbeam(self, encoder_output, emotion_inputs, emodict_list, inputs,sourceids, inputlength,extravocablist,beam=3): 
#        pass
        emotion_input=self.emoemb(emotion_inputs)
        emo_len=torch.Tensor([len(item) for item in emodict_list]).long().cuda()
        self.max_extra=max([len(dicts) for dicts in extravocablist])
        encoder_output=torch.tanh(self.w0(torch.cat([encoder_output,emotion_input],-1)))
        emotion=torch.zeros(encoder_output.size(0),self.emo_max,self.emo_size).cuda()
        for i in range(encoder_output.size(0)):
            emotion[i]=self.module_lists[emotion_inputs[i].data.cpu()].weight
#        target=output[:,1:]
#        out_mask=length2mask(ouputlen,target.size(1))
        emo_mask=length2mask(emo_len,self.emo_max)
#        output=copy.deepcopy(output[:,:-1])
#        output[output>=self.vocab_size]=1
#        output=self.emb(output)
        hidden=encoder_output
        lm_loss=[]
        conv_loss=[]
        out_tokens=[]
        batch_size=encoder_output.size(0)
        
        emotion=emotion.unsqueeze(1).repeat(1,beam,1,1).reshape(-1,emotion.size(1),emotion.size(2))
        emotion_input=emotion_input.unsqueeze(1).repeat(1,beam,1).reshape(-1,emotion_input.size(1))
        emo_mask=emo_mask.unsqueeze(1).repeat(1,beam,1).reshape(-1,emo_mask.size(1))
        hyps=[[] for _ in range(batch_size)]
        tokenss=torch.ones(batch_size,beam,1).long().cuda()*2
        score=torch.Tensor([-float('inf')]*batch_size*beam).float().cuda()
        for i in range(batch_size):
            score[i*beam]=0
        score=score.reshape(batch_size,beam)
        hidden=hidden.unsqueeze(1).repeat(1,beam,1).reshape(-1,hidden.size(1))
#        coverage = torch.zeros(batch_size, self.emo_max).cuda()
#        coverage1 = torch.zeros(batch_size, inputs.size(1)).cuda()
        coverage = torch.zeros(batch_size*beam, self.emo_max).cuda()
        coverage1 = torch.zeros(batch_size*beam, inputs.size(1)).cuda()
        context_ids=torch.arange(self.emo_max).unsqueeze(0).repeat(encoder_output.size(0),1).cuda().unsqueeze(1).repeat(1,beam,1).reshape(beam*batch_size,-1)
        inputs=inputs.unsqueeze(1).repeat(1,beam,1,1).reshape(-1,inputs.size(1),inputs.size(2))
        inputlength=inputlength.unsqueeze(1).repeat(1,beam).reshape(-1)
        sourceids=sourceids.unsqueeze(1).repeat(1,beam,1).reshape(-1,sourceids.size(1))
        batch_position = torch.arange(0, batch_size).long().cuda() * beam
        fused=[torch.zeros(batch_size* beam,self.emo_size).cuda(),torch.zeros(batch_size* beam,inputs.size(2)).cuda()]
        for ins in range(30):
            if min([len(items) for items in hyps])>=beam:
                break
            tokens=copy.deepcopy(tokenss[:,:,-1].reshape(-1))
            tokens[tokens>=self.vocab_size]=1
            emb=torch.cat([self.emb(tokens),emotion_input],-1)
#            currenttarget=target[:,i].unsqueeze(-1)
#            target_mask_0 = out_mask[:,i]
            hidden=self.rnn(emb,hidden)#b5,h
            combined_vocab, realatt, realatt1,fused=self.step(hidden, emotion, context_ids, coverage, emo_mask, inputs, sourceids, inputlength, coverage1)
            totallen=combined_vocab.size(1)
            combined_vocab=(combined_vocab+0.00000001).log()+score.reshape(-1).unsqueeze(-1)
            combined_vocab=combined_vocab.reshape(batch_size,-1)#b,beam*total
            score,idx=combined_vocab.topk(beam,-1)#b,5
            idx1=idx//totallen+batch_position.unsqueeze(-1)#b,5
            hidden=hidden.index_select(0,idx1.view(-1))
            tmptoken=tokenss.view(-1,tokenss.size(2)).index_select(0,idx1.view(-1)).reshape(batch_size,beam,-1)
            idx2=idx%totallen
            tokenss=torch.cat([tmptoken,idx2.unsqueeze(-1)],-1)
            if ins==29:
               for i in range(idx2.size(0)):
                    for j in range(beam):
                        if idx2[i,j].cpu().numpy()==3 or idx2[i,j].cpu().numpy()==0:
                            hyps[i].append([tokenss[i,j,1:-1].cpu().numpy(),score[i,j].cpu().data.tolist()/(ins+1)])
                        else:
                            hyps[i].append([tokenss[i,j,1:].cpu().numpy(),-300000])
#                            score[i,j]=-float('inf')
            else:
                for i in range(idx2.size(0)):
                    for j in range(beam):
                        if idx2[i,j].cpu().numpy()==3 or idx2[i,j].cpu().numpy()==0:
                            hyps[i].append([tokenss[i,j,1:-1].cpu().numpy(),score[i,j].cpu().data.tolist()/(ins+1)])
                            score[i,j]=-float('inf')
#            target_mask_0 = target_id.ne(0).detach()
#            output = combined_vocab.gather(1, currenttarget).add_(sys.float_info.epsilon).squeeze(-1)
#            tokens=torch.argmax(combined_vocab,-1)
#            tokenss=copy.deepcopy(tokenss)
#            tokenss[tokenss>=self.vocab_size]=1
#            lm_loss.append(output.log().mul(-1) * target_mask_0.float())
            coverage = coverage + realatt
            coverage1 = coverage1 + realatt1
            coverage=coverage.index_select(0,idx1.view(-1))
            coverage1=coverage1.index_select(0,idx1.view(-1))

                # Coverage Loss
                # take minimum across both attn_scores and coverage
#            _cov_loss, _ = torch.stack((coverage, realatt), 2).min(2)
#            cov_loss.append(_cov_loss.sum(1)* target_mask_0.float())
#            out_tokens.append(tokens)
#        outtokens=torch.stack(out_tokens,1).cpu().data.numpy()
        outtokens=[]
        for item in hyps:
            gh=sorted(item,key=lambda x:x[1],reverse=True)[0]
            outtokens.append(gh[0])
        realwords=[]
        self.revextradict=[dict([(value,key) for key,value in item.items()]) for item in extravocablist]
        for  i in range(len(outtokens)):
            tmp=[]
            for item in outtokens[i]:
                if item ==3 or 0:
                    break
                if item < self.vocab_size:
                    tmp.append(self.revvocab[item])
                elif item >= self.vocab_size+self.emo_max:
                    tmp.append(self.revextradict[i][item-self.vocab_size-self.emo_max])
                else:
                    tmp.append(self.revemodict[emotion_inputs[i].data.cpu().tolist()][item-self.vocab_size])
            realwords.append(tmp)
        return realwords
    def test(self, encoder_output, emotion_inputs, emodict_list, inputs,sourceids, inputlength,extravocablist): 
#        oov_max=max([len(item) for item in emodict_list])
        emotion_input=self.emoemb(emotion_inputs)
        emo_len=torch.Tensor([len(item) for item in emodict_list]).long().cuda()
        self.max_extra=max([len(dicts) for dicts in extravocablist])
        encoder_output=torch.tanh(self.w0(torch.cat([encoder_output,emotion_input],-1)))
        emotion=torch.zeros(encoder_output.size(0),self.emo_max,self.emo_size).cuda()
        for i in range(encoder_output.size(0)):
            emotion[i]=self.module_lists[emotion_inputs[i].data.cpu()].weight
#        target=output[:,1:]
#        out_mask=length2mask(ouputlen,target.size(1))
        emo_mask=length2mask(emo_len,self.emo_max)
#        output=copy.deepcopy(output[:,:-1])
#        output[output>=self.vocab_size]=1
#        output=self.emb(output)
        hidden=encoder_output
        lm_loss=[]
        conv_loss=[]
        out_tokens=[]
        batch_size=encoder_output.size(0)
        tokenss=torch.ones(batch_size).long().cuda()*2
        coverage = torch.zeros(batch_size, self.emo_max).cuda()
        coverage1 = torch.zeros(batch_size, inputs.size(1)).cuda()
        context_ids=torch.arange(self.emo_max).unsqueeze(0).repeat(encoder_output.size(0),1).cuda()
        fused=[torch.zeros(batch_size,self.emo_size).cuda(),torch.zeros(batch_size,inputs.size(2)).cuda()]
        for i in range(30):
            fused=[torch.zeros(batch_size,self.emo_size).cuda(),torch.zeros(batch_size,inputs.size(2)).cuda()]
            emb=torch.cat([self.emb(tokenss),emotion_input]+fused,-1)
#            currenttarget=target[:,i].unsqueeze(-1)
#            target_mask_0 = out_mask[:,i]
            hidden=self.rnn(emb,hidden)
            combined_vocab, realatt, realatt1,fused=self.step(hidden, emotion, context_ids, coverage, emo_mask, inputs, sourceids, inputlength, coverage1)
#            target_mask_0 = target_id.ne(0).detach()
#            output = combined_vocab.gather(1, currenttarget).add_(sys.float_info.epsilon).squeeze(-1)
            tokens=torch.argmax(combined_vocab,-1)
            tokenss=copy.deepcopy(tokens)
            tokenss[tokenss>=self.vocab_size]=1
#            lm_loss.append(output.log().mul(-1) * target_mask_0.float())
            coverage = coverage + realatt
            coverage1 = coverage1 + realatt1

                # Coverage Loss
                # take minimum across both attn_scores and coverage
#            _cov_loss, _ = torch.stack((coverage, realatt), 2).min(2)
#            cov_loss.append(_cov_loss.sum(1)* target_mask_0.float())
            out_tokens.append(tokens)
        outtokens=torch.stack(out_tokens,1).cpu().data.numpy()
        realwords=[]
        self.revextradict=[dict([(value,key) for key,value in item.items()]) for item in extravocablist]
        for  i in range(outtokens.shape[0]):
            tmp=[]
            for item in outtokens[i,:]:
                if item ==3 or 0:
                    break
                if item < self.vocab_size:
                    tmp.append(self.revvocab[item])
                elif item >= self.vocab_size+self.emo_max:
                    tmp.append(self.revextradict[i][item-self.vocab_size-self.emo_max])
                else:
                    tmp.append(self.revemodict[emotion_inputs[i].data.cpu().tolist()][item-self.vocab_size])
            realwords.append(tmp)
        return realwords
#        out_hidden =torch.stack([out_hidden],1)  #b,s,h
#        out_hidden=torch.index_select(out_hidden.reshape(-1,out_hidden.size(2)),dim=0,torch.arrange(batch_size).cuda()*eachlen+ouputlen)#b,h
#        out_loss=self.classloss(self.w5(out_hidden),emotion_inputs)
#        total_masked_loss = torch.stack(lm_loss, 1).sum(1).div(ouputlen) + self.lmbda * \
#                torch.stack(cov_loss, 1).sum(1).div(ouputlen)+self.alpha*out_loss
class DecoderRNN(BaseRNN):

    def __init__(self, vocab_size, embedding, embed_size, pemsize, sos_id, eos_id,
                 unk_id, max_len=100, n_layers=1, rnn_cell='gru',
                 bidirectional=True, input_dropout_p=0, dropout_p=0,
                 lmbda=1.5, USE_CUDA = torch.cuda.is_available(), mask=0):
        hidden_size = embed_size

        super(DecoderRNN, self).__init__(vocab_size, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        self.rnn = self.rnn_cell(embed_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.output_size = vocab_size
        self.hidden_size = hidden_size
        self.max_length = max_len
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.unk_id = unk_id
        self.mask = mask
        self.embedding = embedding
        self.lmbda = lmbda
        self.USE_CUDA = USE_CUDA
        #directions
        self.Wh = nn.Linear(hidden_size * 2, hidden_size)
        #output
        self.V = nn.Linear(hidden_size * 3, self.output_size)
        #params for attention
        self.Wih = nn.Linear(hidden_size, hidden_size)  # for obtaining e from encoder input
        self.Wfh = nn.Linear(hidden_size, hidden_size)  # for obtaining e from encoder field
        self.Ws = nn.Linear(hidden_size, hidden_size)  # for obtaining e from current state
        self.w_c = nn.Linear(1, hidden_size)  # for obtaining e from context vector
        self.v = nn.Linear(hidden_size, 1)
        # parameters for p_gen
        self.w_ih = nn.Linear(hidden_size, 1)    # for changing context vector into a scalar
        self.w_fh = nn.Linear(hidden_size, 1)    # for changing context vector into a scalar
        self.w_s = nn.Linear(hidden_size, 1)    # for changing hidden state into a scalar
        self.w_x = nn.Linear(embed_size, 1)     # for changing input embedding into a scalar
        # parameters for self attention
        self_size = pemsize * 2  # hidden_size +
        self.wp = nn.Linear(self_size, self_size)
        self.wc = nn.Linear(self_size, self_size)
        self.wa = nn.Linear(self_size, self_size)

    def get_matrix(self, encoderp):
        tp = torch.tanh(self.wp(encoderp))
        tc = torch.tanh(self.wc(encoderp))
        f = tp.bmm(self.wa(tc).transpose(1, 2))
        return F.softmax(f, dim=2)

    def self_attn(self, f_matrix, encoderi, encoderf):
        c_contexti = torch.bmm(f_matrix, encoderi)
        c_contextf = torch.bmm(f_matrix, encoderf)
        return c_contexti, c_contextf

    def decode_step(self, input_ids, coverage, _h, enc_proj, batch_size, max_enc_len,
                    enc_mask, c_contexti, c_contextf, embed_input, max_source_oov, f_matrix):
        dec_proj = self.Ws(_h).unsqueeze(1).expand_as(enc_proj)
        cov_proj = self.w_c(coverage.view(-1, 1)).view(batch_size, max_enc_len, -1)
        e_t = self.v(torch.tanh(enc_proj + dec_proj + cov_proj).view(batch_size*max_enc_len, -1))

        # mask to -INF before applying softmax
        attn_scores = e_t.view(batch_size, max_enc_len)
        del e_t
        attn_scores.data.masked_fill_(enc_mask.data.bool(), 0)
        attn_scores = F.softmax(attn_scores, dim=1)

        contexti = attn_scores.unsqueeze(1).bmm(c_contexti).squeeze(1)
        contextf = attn_scores.unsqueeze(1).bmm(c_contextf).squeeze(1)

        # output proj calculation
        p_vocab = F.softmax(self.V(torch.cat((_h, contexti, contextf), 1)), dim=1)
        # p_gen calculation
        p_gen = torch.sigmoid(self.w_ih(contexti) + self.w_fh(contextf) + self.w_s(_h) + self.w_x(embed_input))
        p_gen = p_gen.view(-1, 1)
        weighted_Pvocab = p_vocab * p_gen
        weighted_attn = (1-p_gen) * attn_scores

        if max_source_oov > 0:
            # create OOV (but in-article) zero vectors
            ext_vocab = torch.zeros(batch_size, max_source_oov)
            if self.USE_CUDA:
                ext_vocab=ext_vocab.cuda()
            combined_vocab = torch.cat((weighted_Pvocab, ext_vocab), 1)
            del ext_vocab
        else:
            combined_vocab = weighted_Pvocab
        del weighted_Pvocab       # 'Recheck OOV indexes!'
        # scatter article word probs to combined vocab prob.
        combined_vocab = combined_vocab.scatter_add(1, input_ids, weighted_attn)
        return combined_vocab, attn_scores

    def forward(self, max_source_oov=0, targets=None, targets_id=None, input_ids=None,
                enc_mask=None, encoder_hidden=None, encoderi=None, encoderf=None,
                encoderp=None, teacher_forcing_ratio=None, w2fs=None, fig=False):

        targets, batch_size, max_length, max_enc_len = self._validate_args(targets, encoder_hidden, encoderi, teacher_forcing_ratio)
        decoder_hidden = self._init_state(encoder_hidden)
        coverage = torch.zeros(batch_size, max_enc_len)
        enci_proj = self.Wih(encoderi.view(batch_size*max_enc_len, -1)).view(batch_size, max_enc_len, -1)
        encf_proj = self.Wfh(encoderf.view(batch_size*max_enc_len, -1)).view(batch_size, max_enc_len, -1)
        f_matrix = self.get_matrix(encoderp)
        enc_proj = enci_proj + encf_proj

        # get link attention scores
#        c_contexti, c_contextf = self.self_attn(f_matrix, encoderi, encoderf)
        c_contexti, c_contextf=encoderi, encoderf
        if self.USE_CUDA:
            coverage = coverage.cuda()
        if teacher_forcing_ratio:
            embedded = self.embedding(targets)
            embed_inputs = self.input_dropout(embedded)
            # coverage initially zero
            dec_lens = (targets > 0).float().sum(1)
            lm_loss, cov_loss = [], []
            hidden, _ = self.rnn(embed_inputs, decoder_hidden)
            # step through decoder hidden states
            for _step in range(max_length):
                _h = hidden[:, _step, :]
                target_id = targets_id[:, _step+1].unsqueeze(1)
                embed_input = embed_inputs[:, _step, :]

                combined_vocab, attn_scores = self.decode_step(input_ids, coverage, _h, enc_proj, batch_size,
                                                               max_enc_len, enc_mask, c_contexti, c_contextf,
                                                               embed_input, max_source_oov, f_matrix)
                # mask the output to account for PAD
                target_mask_0 = target_id.ne(0).detach()
                output = combined_vocab.gather(1, target_id).add_(sys.float_info.epsilon)
                lm_loss.append(output.log().mul(-1) * target_mask_0.float())

                coverage = coverage + attn_scores

                # Coverage Loss
                # take minimum across both attn_scores and coverage
                _cov_loss, _ = torch.stack((coverage, attn_scores), 2).min(2)
                cov_loss.append(_cov_loss.sum(1))
            # add individual losses
            total_masked_loss = torch.cat(lm_loss, 1).sum(1).div(dec_lens) + self.lmbda * \
                torch.stack(cov_loss, 1).sum(1).div(dec_lens)
            return total_masked_loss
        else:
            return self.evaluate(targets, batch_size, max_length, max_source_oov, c_contexti, c_contextf, f_matrix,
                                 decoder_hidden, enc_mask, input_ids, coverage, enc_proj, max_enc_len, w2fs, fig)

    def evaluate(self, targets, batch_size, max_length, max_source_oov, c_contexti, c_contextf, f_matrix,
                 decoder_hidden, enc_mask, input_ids, coverage, enc_proj, max_enc_len, w2fs, fig):
        lengths = np.array([max_length] * batch_size)
        decoded_outputs = []
        if fig:
            attn = []
        embed_input = self.embedding(targets)
        # step through decoder hidden states
        for _step in range(max_length):
            _h, _c = self.rnn(embed_input, decoder_hidden)
            combined_vocab, attn_scores = self.decode_step(input_ids, coverage,
                                                           _h.squeeze(1), enc_proj, batch_size, max_enc_len, enc_mask,
                                                           c_contexti, c_contextf, embed_input.squeeze(1),
                                                           max_source_oov, f_matrix)
            # not allow decoder to output UNK
            combined_vocab[:, self.unk_id] = 0
#            remain_enc=(1-enc_mask.byte()).sum(-1)
#            for ids in range(remain_enc.size(0)):
#                if remain_enc.data[ids]!=0:
#                    combined_vocab[:, self.eos_id] = 0
#                    combined_vocab[:, 0] = 0
#            combined_vocab[:, self.eos_id] = 0
            symbols = combined_vocab.topk(1)[1]
            if self.mask == 1:
                tmp_mask = torch.zeros_like(enc_mask, dtype=torch.uint8)
                for i in range(symbols.size(0)):
                    pos = (input_ids[i] == symbols[i]).nonzero()
                    if pos.size(0) != 0:
                        tmp_mask[i][pos] = 1
                enc_mask = torch.where(enc_mask.byte() > tmp_mask.byte(), enc_mask.byte(), tmp_mask)
                enc_mask= enc_mask.bool()
            if fig:
                attn.append(attn_scores)
            decoded_outputs.append(symbols.clone())
            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > _step) & eos_batches) != 0
                lengths[update_idx] = len(decoded_outputs)
            # change unk to corresponding field
            for i in range(symbols.size(0)):
                w2f = w2fs[i]
                if symbols[i].item() > self.vocab_size-1:
                    symbols[i] = w2f[symbols[i].item()]
            # symbols.masked_fill_((symbols > self.vocab_size-1), self.unk_id)
            embed_input = self.embedding(symbols)
            decoder_hidden = _c
            coverage = coverage + attn_scores
        if fig:
            return torch.stack(decoded_outputs, 1).squeeze(2), lengths.tolist(), f_matrix[0], \
                   torch.stack(attn, 1).squeeze(2)[0]
        else:
            return torch.stack(decoded_outputs, 1).squeeze(2), lengths.tolist()

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            h = self.Wh(h)
        return h

    def _validate_args(self, targets, encoder_hidden, encoder_outputs, teacher_forcing_ratio):
        if encoder_outputs is None:
            raise ValueError("Argument encoder_outputs cannot be None when attention is used.")
        else:
            max_enc_len = encoder_outputs.size(1)
        # inference batch size
        if targets is None and encoder_hidden is None:
            batch_size = 1
        else:
            if targets is not None:
                batch_size = targets.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default targets and max decoding length
        if targets is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no targets is provided.")
            # torch.set_grad_enabled(False)
            targets = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if self.USE_CUDA:
                targets = targets.cuda()
            max_length = self.max_length
        else:
            max_length = targets.size(1) - 1     # minus the start of sequence symbol

        return targets, batch_size, max_length, max_enc_len