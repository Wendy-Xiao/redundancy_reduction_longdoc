from collections import Counter
from random import random
from nltk import word_tokenize
from torch import nn
from torch.autograd import Variable
import pandas as pd
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence,pad_packed_sequence
import json
import re
import numpy as np
import torch
import torch.nn.functional as F
import os
from utils import *



# ExtSumLG model
class Attentive_context(nn.Module):
    def __init__(self,input_size,hidden_size, mlp_size, cell_type='gru'):
        super(Attentive_context, self).__init__()
        self.hidden_size = hidden_size
        self.cell = cell_type
        if self.cell == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size,num_layers=1, bidirectional = True)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size,num_layers=1, bidirectional = True)

        self.hidden2out = self.build_mlp(hidden_size*4,mlp_size,0.3)
        self.final_layer = nn.Linear(mlp_size, 1)
        self.feat_attn = nn.Linear(hidden_size*4,hidden_size*4,bias=False)
        self.context_vector = nn.Parameter(torch.rand(hidden_size*4,1))
        self.dropout_layer = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()

    def build_mlp(self,input,output,dropout):
        mlp = []
        mlp.append(nn.Linear(input, output))
        mlp.append(nn.ReLU())
        mlp.append(nn.Dropout(p=dropout, inplace=True))
        return nn.Sequential(*mlp)

    def attention_net(self, features, sent_representation):
        # features_tanh = torch.tanh(features)
        sent_representation = sent_representation.unsqueeze(1)
        sent_representation = sent_representation.expand(-1,3,-1) # [batch,3,hidden*2]
        f = torch.cat([sent_representation,features],-1) #(batch,3,hidden*4)
        context = torch.tanh(self.feat_attn(f)) # (batch,3,hidden*4)
        v = self.context_vector.unsqueeze(0).expand(context.size()[0],-1,-1)
        attn_weights = torch.bmm(context,v)

        soft_attn_weights = F.softmax(attn_weights, 1) #(batch,3,1)

        new_hidden_state = torch.bmm(features.transpose(1, 2), soft_attn_weights).squeeze(2) #[batch,hidden*2]
        return new_hidden_state

    def forward(self,inputs,input_length,section_indicator,begin,end,device,with_embedding=False):

        # Go through the GRU/LSTM model
        output, hidden = self.rnn(inputs)
        output,_ = pad_packed_sequence(output)
        if self.cell == 'lstm':
            hidden = hidden[0]

        # Hidden=[num_layers * 2, batch, hidden_size]
        hidden = hidden.permute(1,0,2)
        seq_length = output.size()[0]
        # Get the representation for document - global context
        doc_represent = hidden.contiguous().view(hidden.size()[0],2*hidden.size()[2])

        # doc_representation =[batch, 2*hidden_size]
        doc_represent = doc_represent.expand(seq_length,-1,-1)
        doc_represent = self.dropout_layer(doc_represent)
        # doc_representation =[seq_len, batch, doc_size]

        # Get the representation for section - local context
        local_context_representation = []
        padding_begin = torch.zeros((1,output.shape[1],2*self.hidden_size))
        padding_end = torch.zeros((1,output.shape[1],2*self.hidden_size))
        if torch.cuda.is_available():
            padding_begin = padding_begin.to(device)
            padding_end = padding_end.to(device)
        pad_output = torch.cat((padding_begin,output,padding_end),0)

        # output = [seq_len, batch, 2 * hidden_size]
        # LSTM-MINUS
        for i in range(output.shape[1]):
            # local_context_f = [batch,hidden_size]
            # local_context_b = [batch,hidden_size]

            local_context_f = pad_output[end[i],i,:self.hidden_size]\
                                -pad_output[begin[i],i,:self.hidden_size]
            local_context_b = pad_output[begin[i]+1,i,self.hidden_size:]\
                                -pad_output[end[i]+1,i,self.hidden_size:]
            local_context = torch.cat((local_context_f,local_context_b),-1)
            local_context_representation.append(local_context)
            del local_context_f,local_context_b,local_context

        # local_context_representation = [seq_len, batch, local_size]
        del pad_output,padding_begin,padding_end
        local_context_representation = torch.stack(local_context_representation,0)
        local_context_representation = torch.bmm(section_indicator,local_context_representation)
        local_context_representation = local_context_representation.permute(1,0,2)
        local_context_representation = self.dropout_layer(local_context_representation)
        output_drop = self.dropout_layer(output)

        ##Combine the local and global context by attention
        context = torch.stack((local_context_representation,doc_represent),2)
        attn_in = torch.cat((context,output_drop.unsqueeze(2).expand(-1,-1,2,-1)),3)
        attn_in = torch.tanh(attn_in)
        attn_weight = F.softmax(torch.matmul(self.feat_attn(attn_in),self.context_vector),dim=2)
        
        context = context.permute(0,1,3,2)
        weighted_context_representation = torch.matmul(context,attn_weight).squeeze(-1)

        # Final classifier layer
        mlp_in = torch.cat([weighted_context_representation,output_drop],-1)

        # mlp_out = [seq_len,batch,mlp_size]
        mlp_out = self.hidden2out(mlp_in)
        # out = [seq_len, batch, 1]
        out = self.final_layer(mlp_out)

        return out

# ExtSumLG + SR Decoder
class Attentive_context_sr(nn.Module):
    def __init__(self,input_size,hidden_size, mlp_size, cell_type='gru',\
                 sentence_size=100, document_size=200,
                 segments=4, max_position_weights=25,
                 segment_size=50, position_size=50):
        super(Attentive_context_sr, self).__init__()
        self.hidden_size = hidden_size
        self.cell = cell_type
        if self.cell == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size,num_layers=1, bidirectional = True)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size,num_layers=1, bidirectional = True)

        self.feat_attn = nn.Linear(hidden_size*4,hidden_size*4,bias=False)
        self.context_vector = nn.Parameter(torch.rand(hidden_size*4,1))
        self.dropout_layer = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()


        ####sr decoder
        # self.midtrans=nn.Linear(hidden_size*4,hidden_size*2)

        inp_size = hidden_size*4
        # if bidirectional:
        #     inp_size *= 2

        self.sentence_rep = nn.Sequential(
            nn.Linear(inp_size, sentence_size), nn.ReLU())
        self.content_logits = nn.Linear(sentence_size, 1)
        self.document_rep = nn.Sequential(
            nn.Linear(sentence_size, document_size), 
            nn.Tanh(), 
            nn.Linear(document_size, sentence_size))
        self.similarity = nn.Bilinear(
            sentence_size, sentence_size, 1, bias=False)
        self.bias = nn.Parameter(torch.FloatTensor([0]))

        self.max_position_weights = max_position_weights
        self.segments = segments
        self.position_encoder = nn.Sequential(
            nn.Embedding(max_position_weights + 1, position_size, 
                         padding_idx=0),
            nn.Linear(position_size, 1, bias=False))
        self.segment_encoder = nn.Sequential(
            nn.Embedding(segments + 1, segment_size, padding_idx=0),
            nn.Linear(segment_size, 1, bias=False))

        # self.redundancy_layer = nn.Linear(4*hidden_size, 1)

    def build_mlp(self,input,output,dropout):
        mlp = []
        mlp.append(nn.Linear(input, output))
        mlp.append(nn.ReLU())
        mlp.append(nn.Dropout(p=dropout, inplace=True))
        return nn.Sequential(*mlp)

    def attention_net(self, features, sent_representation):
        # features_tanh = torch.tanh(features)
        sent_representation = sent_representation.unsqueeze(1)
        sent_representation = sent_representation.expand(-1,3,-1) # [batch,3,hidden*2]
        f = torch.cat([sent_representation,features],-1) #(batch,3,hidden*4)
        context = torch.tanh(self.feat_attn(f)) # (batch,3,hidden*4)
        v = self.context_vector.unsqueeze(0).expand(context.size()[0],-1,-1)
        attn_weights = torch.bmm(context,v)

        soft_attn_weights = F.softmax(attn_weights, 1) #(batch,3,1)

        new_hidden_state = torch.bmm(features.transpose(1, 2), soft_attn_weights).squeeze(2) #[batch,hidden*2]
        return new_hidden_state

    def novelty(self, sentence_state, summary_rep):
        sim = self.similarity(
            sentence_state.squeeze(1), torch.tanh(summary_rep).squeeze(1))
        novelty = -sim.squeeze(1)
        return novelty

    def position_logits(self, length):
        batch_size = length.size(0)
        abs_pos = torch.arange(
            1, length.data[0].item() + 1, device=length.device)\
            .view(1, -1).repeat(batch_size, 1)

        chunk_size = (length.float() / self.segments).round().view(-1, 1)
        rel_pos = (abs_pos.float() / chunk_size).ceil().clamp(
            0, self.segments).long()

        abs_pos.data.clamp_(0, self.max_position_weights)
        pos_logits = self.position_encoder(abs_pos).squeeze(2)
        seg_logits = self.segment_encoder(rel_pos).squeeze(2)
        return pos_logits, seg_logits

    def forward(self,inputs,input_length,section_indicator,begin,end,device):
        # The GRU/LSTM model
        output, hidden = self.rnn(inputs)
        output,_ = pad_packed_sequence(output)
        if self.cell == 'lstm':
            hidden = hidden[0]
        # Hidden=[num_layers * 2, batch, hidden_size]
        # hidden = self.dropout_layer(hidden)
        hidden = hidden.permute(1,0,2)
        seq_length = output.size()[0]
        doc_represent = hidden.contiguous().view(hidden.size()[0],2*hidden.size()[2])
        # doc_representation =[batch, 2*hidden_size]

        doc_represent = doc_represent.expand(seq_length,-1,-1)
        doc_represent = self.dropout_layer(doc_represent)
        # doc_representation =[seq_len, batch, doc_size]
        local_context_representation = []
        padding_begin = torch.zeros((1,output.shape[1],2*self.hidden_size))
        padding_end = torch.zeros((1,output.shape[1],2*self.hidden_size))
        if torch.cuda.is_available():
            padding_begin = padding_begin.to(device)
            padding_end = padding_end.to(device)
        pad_output = torch.cat((padding_begin,output,padding_end),0)

        # output = [seq_len, batch, 2 * hidden_size]
        for i in range(output.shape[1]):
            # local_context_f = [batch,hidden_size]
            # local_context_b = [batch,hidden_size]

            local_context_f = pad_output[end[i],i,:self.hidden_size]\
                                -pad_output[begin[i],i,:self.hidden_size]
            local_context_b = pad_output[begin[i]+1,i,self.hidden_size:]\
                                -pad_output[end[i]+1,i,self.hidden_size:]
            local_context = torch.cat((local_context_f,local_context_b),-1)
            local_context_representation.append(local_context)
            del local_context_f,local_context_b,local_context
        # local_context_representation = [seq_len, batch, local_size]
        del pad_output,padding_begin,padding_end
        local_context_representation = torch.stack(local_context_representation,0)
        local_context_representation = torch.bmm(section_indicator,local_context_representation)
        local_context_representation = local_context_representation.permute(1,0,2)
        local_context_representation = self.dropout_layer(local_context_representation)
        output_drop = self.dropout_layer(output)
        context = torch.stack((local_context_representation,doc_represent),2)
        attn_in = torch.cat((context,output_drop.unsqueeze(2).expand(-1,-1,2,-1)),3)
        attn_in = torch.tanh(attn_in)
        attn_weight = F.softmax(torch.matmul(self.feat_attn(attn_in),self.context_vector),dim=2)
        
        context = context.permute(0,1,3,2)
        weighted_context_representation = torch.matmul(context,attn_weight).squeeze(-1)

        mlp_in = torch.cat([weighted_context_representation,output_drop],-1) #len*batch*(4*hidden)


        ### SR Decoder
        mlp_in = mlp_in.permute(1,0,2) # batch*len*(4*hidden)
        sentence_states = self.sentence_rep(mlp_in)
        del mlp_in
        content_logits = self.content_logits(sentence_states).squeeze(2)

        avg_sentence = sentence_states.sum(1).div_(
            input_length.view(-1, 1).float())
        doc_rep = self.document_rep(avg_sentence).unsqueeze(2)
        del avg_sentence
        salience_logits = sentence_states.bmm(doc_rep).squeeze(2)
        del doc_rep
        pos_logits, seg_logits = self.position_logits(input_length)

        static_logits = content_logits + salience_logits + pos_logits \
            + seg_logits + self.bias.unsqueeze(0)
        
        sentence_states = sentence_states.split(1, dim=1)
        summary_rep = torch.zeros_like(sentence_states[0])
        logits = []
        for step in range(input_length[0].item()):
            novelty_logits = self.novelty(sentence_states[step], summary_rep)
            logits_step = static_logits[:, step] + novelty_logits
            del novelty_logits
            prob = torch.sigmoid(logits_step)
            
            summary_rep += sentence_states[step] * prob.view(-1, 1, 1)
            logits.append(logits_step.view(-1, 1))
        logits = torch.cat(logits, 1).unsqueeze(-1) #batch * length * 1
        logits = logits.permute(1,0,2) #length * batch *1

        return logits

# ExtSumLG + NeuSum Decoder
class NeuSum(nn.Module):
    def __init__(self,input_size,hidden_size, mlp_size, cell_type='gru',\
                 ):
        super(NeuSum, self).__init__()
        self.hidden_size = hidden_size
        self.cell = cell_type
        if self.cell == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size,num_layers=1, bidirectional = True)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size,num_layers=1, bidirectional = True)

        self.feat_attn = nn.Linear(hidden_size*4,hidden_size*4,bias=False)
        self.context_vector = nn.Parameter(torch.rand(hidden_size*4,1))
        self.dropout_layer = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()


        ####neusum decoder
        self.pointer = Pointer(1,hidden_size*2,hidden_size*2,0.3,100)


    def attention_net(self, features, sent_representation):
        # features_tanh = torch.tanh(features)
        sent_representation = sent_representation.unsqueeze(1)
        sent_representation = sent_representation.expand(-1,3,-1) # [batch,3,hidden*2]
        f = torch.cat([sent_representation,features],-1) #(batch,3,hidden*4)
        context = torch.tanh(self.feat_attn(f)) # (batch,3,hidden*4)
        v = self.context_vector.unsqueeze(0).expand(context.size()[0],-1,-1)
        attn_weights = torch.bmm(context,v)

        soft_attn_weights = F.softmax(attn_weights, 1) #(batch,3,1)

        new_hidden_state = torch.bmm(features.transpose(1, 2), soft_attn_weights).squeeze(2) #[batch,hidden*2]
        return new_hidden_state

    def gen_all_masks(self, base_mask, targets):
        ###targets = batch * max_step
        ###base mask = batch * max_doc_length 1 if padding
        ### mask = batch * max_doc_length 

        batch_size = targets.shape[0]
        res = []
        res.append(base_mask)
        for i in range(targets.shape[1]):
            next_mask = res[-1].data.clone()
            for j in range(batch_size):
                if targets[j][i] !=-1:
                    next_mask[j][targets[j][i]] = 0
            # with torch.no_grad()
            next_mask.requires_grad = False
            res.append(next_mask)
        return res

    def gen_mask_with_length(self, doc_len, batch_size, lengths,device):
        mask = torch.ByteTensor(batch_size, doc_len).zero_().to(device)
        # ll = lengths.data.view(-1).tolist()
        for i in range(batch_size):
            for j in range(doc_len):
                if j >= lengths[i]:
                    mask[i][j] = 1
        mask = mask.float()
        return mask

    def forward(self,inputs,input_length,section_indicator,begin,end,device,max_point_step,targets,doc_sent_mask):
        output, hidden = self.rnn(inputs)
        output,_ = pad_packed_sequence(output)
        if self.cell == 'lstm':
            hidden = hidden[0]
        # Hidden=[num_layers * 2, batch, hidden_size]
        # hidden = self.dropout_layer(hidden)
        hidden = hidden.permute(1,0,2)
        seq_length = output.size()[0]
        doc_represent_init = hidden.contiguous().view(hidden.size()[0],2*hidden.size()[2])
        # doc_representation =[batch, 2*hidden_size]

        doc_represent = doc_represent_init.expand(seq_length,-1,-1)
        doc_represent = self.dropout_layer(doc_represent)
        # doc_representation =[seq_len, batch, doc_size]
        local_context_representation = []
        padding_begin = torch.zeros((1,output.shape[1],2*self.hidden_size))
        padding_end = torch.zeros((1,output.shape[1],2*self.hidden_size))
        if torch.cuda.is_available():
            padding_begin = padding_begin.to(device)
            padding_end = padding_end.to(device)
        pad_output = torch.cat((padding_begin,output,padding_end),0)

        # output = [seq_len, batch, 2 * hidden_size]
        for i in range(output.shape[1]):
            # local_context_f = [batch,hidden_size]
            # local_context_b = [batch,hidden_size]

            local_context_f = pad_output[end[i],i,:self.hidden_size]\
                                -pad_output[begin[i],i,:self.hidden_size]
            local_context_b = pad_output[begin[i]+1,i,self.hidden_size:]\
                                -pad_output[end[i]+1,i,self.hidden_size:]
            local_context = torch.cat((local_context_f,local_context_b),-1)
            local_context_representation.append(local_context)
            del local_context_f,local_context_b,local_context
        # local_context_representation = [seq_len, batch, local_size]
        del pad_output,padding_begin,padding_end
        local_context_representation = torch.stack(local_context_representation,0)
        local_context_representation = torch.bmm(section_indicator,local_context_representation)
        local_context_representation = local_context_representation.permute(1,0,2)
        local_context_representation = self.dropout_layer(local_context_representation)
        output_drop = self.dropout_layer(output)
        context = torch.stack((local_context_representation,doc_represent),2)
        attn_in = torch.cat((context,output_drop.unsqueeze(2).expand(-1,-1,2,-1)),3)
        attn_in = torch.tanh(attn_in)
        attn_weight = F.softmax(torch.matmul(self.feat_attn(attn_in),self.context_vector),dim=2)
        
        context = context.permute(0,1,3,2)
        weighted_context_representation = torch.matmul(context,attn_weight).squeeze(-1)

        # mlp_in = torch.cat([weighted_context_representation,output_drop],-1) #len*batch*(2*hidden)
        mlp_in = weighted_context_representation+output_drop #len*batch*(2*hidden)

        ### neusum decoder

        dec_hidden = doc_represent_init
        doc_context = mlp_in
        
         

        prev_att = torch.zeros(dec_hidden.shape).to(device)
        # doc_len = doc_context.shape[0]
        # batch_size = doc_context.shape[1]

        # doc_sent_mask=self.gen_mask_with_length( doc_len, batch_size, input_length,device)
        # batch * doc_length
        all_masks = self.gen_all_masks(doc_sent_mask, targets)


        doc_sent_mask.requires_grad=False
        scores = self.pointer(
            dec_hidden, doc_context,
            doc_sent_mask,all_masks,
            prev_att,max_point_step,
            targets.transpose(0, 1).contiguous())
        return scores


### The following Classes are grabbed from the official code of NEUSUM

class StackedGRU(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, h_0[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1


class Pointer(nn.Module):
    def __init__(self, layers,doc_enc_size,dec_rnn_size,dec_dropout,att_vec_size):
        self.layers = layers
        input_size = doc_enc_size

        super(Pointer, self).__init__()
        # self.rnn = StackedGRU(layers, input_size, dec_rnn_size, dec_dropout)
        self.rnn = nn.GRUCell(input_size, dec_rnn_size)
        # self.scorer = ScoreAttention(doc_enc_size, dec_rnn_size, att_vec_size)
        self.hidden_size = dec_rnn_size
        self.linear_q = nn.Linear(dec_rnn_size,att_vec_size)
        self.linear_d = nn.Linear(dec_rnn_size,att_vec_size)
        self.linear_s = nn.Linear(att_vec_size,1)


    def get_score(self,hidden, context, sent_mask):
        # context = seq_len*batch*(2*hidden)
        # hidden = 1*batch*(2*hidden)
        # mask = seq_len*batch

        energy = self.linear_s(torch.tanh(self.linear_q(hidden.expand(context.shape[0],-1,-1))+self.linear_d(context))).squeeze()
        if sent_mask is not None:
            energy = energy * sent_mask + (1-sent_mask) * (-1e8)
        energy = F.softmax(energy, dim=0)
        return energy

    def forward(self, hidden, context, doc_sent_mask, src_pad_masks,pre_att_hard,
                 max_step, prev_targets):
        """

        :param hidden: pointer network RNN hidden
        :param context: the document sentence vectors (doc_len, batch, dim)
        :param doc_sent_mask: doc_sent_mask for data pad masking (batch, doc_len)
        :param src_pad_masks: [src_pad_mask for t in times] for rule masking
        :param pre_att_hard: previous hard attention
        :param att_precompute_hard: hard attention precompute
        :param max_step:
        :param prev_targets: (step, batch)
        :return:
        """
        cur_context_hard = pre_att_hard

        all_scores = []
        # self.scorer.applyMask(doc_sent_mask)
        # hard_context_buf = context.view(-1, context.size(2))
        # batch_first_context = context.transpose(0, 1).contiguous()
        hard_context_buf = context
        for i in range(max_step):
            input_vector = cur_context_hard
            output= self.rnn(input_vector, hidden)
            hidden = output 
            ## reg_score: [seq_len * batch]
            reg_score = self.get_score(output, context, doc_sent_mask.permute(1,0)) # src_pad_masks[i]: batch*seq_length
            # print(src_pad_masks[i].permute(1,0))
            # print(reg_score)
            all_scores.append(reg_score)
            

            if self.training and max_step > 1:
                max_idx = prev_targets[i]
                # print(max_idx)
                # hard_max_idx = get_hard_attention_index(context.size(0), context.size(1), max_idx).cuda()
                # hard_max_idx = Variable(hard_max_idx, requires_grad=False, volatile=(not self.training))
                # cur_context_hard = hard_context_buf.index_select(dim=0, index=hard_max_idx)
                cur_context_hard = hard_context_buf[max_idx,torch.arange(max_idx.shape[0])]
                
            elif not self.training:
                max_score, max_idx = reg_score.max(dim=0)
                # print(max_idx)
                # hard_max_idx = get_hard_attention_index(context.size(0), context.size(1), max_idx).cuda()
                # hard_max_idx = Variable(hard_max_idx, requires_grad=False, volatile=(not self.training))
                # cur_context_hard = hard_context_buf.index_select(dim=0, index=hard_max_idx)
                cur_context_hard = hard_context_buf[max_idx,torch.arange(max_idx.shape[0])]

            doc_sent_mask = doc_sent_mask.data.clone()
            for j in range(max_idx.shape[0]):
                doc_sent_mask[j][max_idx[j]] = 0
            # with torch.no_grad()
            doc_sent_mask.requires_grad = False

        scores = torch.stack(all_scores)

        return scores


class ScoreAttention(nn.Module):
    def __init__(self, attend_dim, query_dim, att_dim):
        super(ScoreAttention, self).__init__()
        self.attend_dim = attend_dim
        self.query_dim = query_dim
        self.att_dim = att_dim
        self.linear_pre = nn.Linear(attend_dim, att_dim, bias=True)
        # self.linear_2 = nn.Linear(att_dim, att_dim, bias=True)
        self.linear_q = nn.Linear(query_dim, att_dim, bias=True)
        self.linear_v = nn.Linear(att_dim, 1, bias=True)

        self.sm = nn.Softmax(dim=1)
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context, precompute=None):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        if precompute is None:
            precompute00 = self.linear_pre(context.contiguous().view(-1, context.size(2)))
            precompute = precompute00.view(context.size(0), context.size(1), -1)  # batch x sourceL x att_dim
        targetT = self.linear_q(input).unsqueeze(1)  # batch x 1 x att_dim

        tmp10 = precompute + targetT.expand_as(precompute)  # batch x sourceL x att_dim
        tmp20 = F.tanh(tmp10)
        energy = self.linear_v(tmp20.view(-1, tmp20.size(2))).view(tmp20.size(0), tmp20.size(1))  # batch x sourceL

        if self.mask is not None:
            energy = energy * (1 - self.mask) + self.mask * (-1e8)
        energy = F.softmax(energy, dim=1)

        return energy

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.att_dim) + ' * ' + '(' \
               + str(self.attend_dim) + '->' + str(self.att_dim) + ' + ' \
               + str(self.query_dim) + '->' + str(self.att_dim) + ')' + ')'


