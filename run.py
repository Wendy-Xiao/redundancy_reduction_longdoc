from collections import Counter
from random import random
from nltk import word_tokenize
from torch import nn
from torch.autograd import Variable
import pandas as pd
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence,pad_packed_sequence
import re
import numpy as np
import torch
import torch.nn.functional as F
import os
from utils import *
import sys
from reward_function import compute_reward


def train_seq2seq_batch(data_batch, model, optimizer,pos_weight,device):
	document = data_batch['document']
	label = data_batch['labels']
	input_length = data_batch['input_length']
	indicators = data_batch['indicators']
	padded_lengths = data_batch['padded_lengths']

	total_data = torch.sum(input_length)
	end = torch.clamp(torch.cumsum(padded_lengths,1),0,input_length[0])
	begin = torch.cat((torch.zeros((len(input_length),1),dtype=torch.long),end[:,:-1]),1)

	if torch.cuda.is_available():
		document = document.to(device)
		label = label.to(device)
		input_length = input_length.to(device)
		indicators = indicators.to(device)
		end = end.to(device)
		begin= begin.to(device)

	out = model(document,input_length,indicators,begin,end,device)

	mask = label.gt(-1).float()
	# loss = F.binary_cross_entropy(out,label,weight = mask,reduction='sum',pos_weight=pos_weight)

	loss = F.binary_cross_entropy_with_logits(out,label,weight = mask,reduction='sum',pos_weight=pos_weight)
	model.zero_grad()
	loss.backward()
	optimizer.step()
	l = loss.data
	del document,label,input_length,indicators,end,begin,loss,out
	torch.cuda.empty_cache()
	return l,total_data


def train_seq2seq_batch_rl(data_batch, model, optimizer,pos_weight,device,lamb = 0.1,gamma=0.7):
	
	sigmoid = torch.nn.Sigmoid()
	document = data_batch['document']
	label = data_batch['labels']
	input_length = data_batch['input_length']
	indicators = data_batch['indicators']
	padded_lengths = data_batch['padded_lengths']
	# summary_representation = data_batch['summary_representation']
	sentence_lengths = data_batch['sentence_lengths']
	sentences_batch = [[' '.join(s) for s in d] for d in data_batch['sent_txt']]
	references_batch = data_batch['refs']
	# rouge_matrix_batch =data_batch['rouge_matrix']


	total_data = torch.sum(input_length)
	end = torch.clamp(torch.cumsum(padded_lengths,1),0,input_length[0])
	begin = torch.cat((torch.zeros((len(input_length),1),dtype=torch.long),end[:,:-1]),1)

	if torch.cuda.is_available():
		document = document.to(device)
		label = label.to(device)
		input_length = input_length.to(device)
		indicators = indicators.to(device)
		end = end.to(device)
		begin= begin.to(device)
		# summary_representation = summary_representation.to(device)


	out= model(document,input_length,indicators,begin,end,device)


	output,_ = pad_packed_sequence(document)

	out1 = out.squeeze(-1)
	scores = sigmoid(out1).data
	scores = scores.permute(1,0)

	reward,rl_label = compute_reward(scores,input_length,output,sentences_batch,references_batch,device,sentence_lengths,lamb=lamb)
	mask = label.gt(-1).float()

	# Option 1
	loss_ce = F.binary_cross_entropy_with_logits(out,label,weight = mask,reduction='sum',pos_weight=pos_weight)
	mask = mask*reward
	loss_rl = F.binary_cross_entropy_with_logits(out,rl_label,weight = mask,reduction='sum',pos_weight=pos_weight)
	loss = (1-gamma)*loss_ce+gamma*loss_rl

	model.zero_grad()
	loss.backward()

	optimizer.step()
	l = loss.data
	del document,label,input_length,indicators,end,begin,loss,out
	torch.cuda.empty_cache()
	return l,total_data


def train_seq2seq_batch_neusum(data_batch, model, optimizer,pos_weight,device):
	regression_crit = nn.KLDivLoss(reduction='none')
	document = data_batch['document']
	selections = data_batch['selections']
	scores_gain = data_batch['scores_gain']
	input_length = data_batch['input_length']
	indicators = data_batch['indicators']
	padded_lengths = data_batch['padded_lengths']
	# summary_representation = data_batch['summary_representation']
	sentence_lengths = data_batch['sentence_lengths']
	# sentences_batch = data_batch['sentences']
	# references_batch = data_batch['refs']
	# rouge_matrix_batch =data_batch['rouge_matrix']


	total_data = torch.sum(input_length)
	end = torch.clamp(torch.cumsum(padded_lengths,1),0,input_length[0])
	begin = torch.cat((torch.zeros((len(input_length),1),dtype=torch.long),end[:,:-1]),1)

	max_length = max(input_length)
	oracle_length = [s.shape[0] for s in selections]
	max_step = max(oracle_length)

	doc_sent_mask = torch.zeros(len(input_length),max_length)
	for i,l in enumerate(input_length):
		doc_sent_mask[i,:l]=1
	
	loss_mask = torch.zeros(len(oracle_length),max_step,max_length)
	for i,l in enumerate(oracle_length):
		# batch*step
		loss_mask[i,:l]=doc_sent_mask[i]

	selections = pad_sequence(selections,batch_first=True,padding_value=-1) #batch*longest_oracle
	scores_gain =  make_scores_gain(scores_gain,max_step,max_length)# batch*step*doc_length
	# print(scores_gain.shape)

	if torch.cuda.is_available():
		document = document.to(device)
		input_length = input_length.to(device)
		indicators = indicators.to(device)
		end = end.to(device)
		begin= begin.to(device)
		doc_sent_mask = doc_sent_mask.to(device)
		scores_gain = scores_gain.to(device)
		loss_mask = loss_mask.to(device)

		# summary_representation = summary_representation.to(device)

	doc_sent_scores = model(document,input_length,indicators,begin,end,device,max_step,selections,doc_sent_mask)
	loss = regression_loss(doc_sent_scores, scores_gain, loss_mask, regression_crit)

	model.zero_grad()
	loss.backward()

	optimizer.step()
	l = loss.data
	del document,input_length,indicators,end,begin,loss,doc_sent_mask,scores_gain,loss_mask,doc_sent_scores
	torch.cuda.empty_cache()
	return l,total_data

def make_scores_gain(scores_gain,max_step,max_doc_length,temp=200):
	####scores_gain: [[step*doc_length] for batch times]
	####out: [batch*step*doc_length]
	out = torch.zeros(len(scores_gain),max_step,max_doc_length)
	for i,score in enumerate(scores_gain):
		score = F.softmax(temp*score,dim=1)
		out[i,:score.shape[0],:score.shape[1]] = score
	return out

def regression_loss(pred_scores, gold_scores, mask, crit):
	"""
	:param pred_scores: (step, doc_len, batch)
	:param gold_scores: (batch, step, doc_len)
	:param mask: (batch, step,doc_len)
	:param crit:
	:return:
	"""

	pred_scores = pred_scores.permute(2,0,1)  # (batch, step, doc_len)
	if isinstance(crit, nn.KLDivLoss):
		# TODO: we better use log_softmax(), not log() here. log_softmax() is more numerical stable.
		pred_scores = torch.log(pred_scores + 1e-8)
	# gold_scores = gold_scores.view(*pred_scores.size())
	loss = crit(pred_scores, gold_scores)
	loss = loss * mask
	reduce_loss = loss.sum()
	return reduce_loss



def train_seq2seq_batch_newloss(data_batch, model, optimizer,pos_weight,device,beta=0.7):
	
	sigmoid = torch.nn.Sigmoid()
	document = data_batch['document']
	label = data_batch['labels']
	input_length = data_batch['input_length']
	indicators = data_batch['indicators']
	padded_lengths = data_batch['padded_lengths']
	# summary_representation = data_batch['summary_representation']
	sentence_lengths = data_batch['sentence_lengths']
	# sentences_batch = data_batch['sentences']
	# references_batch = data_batch['summary_text']
	# rouge_matrix_batch =data_batch['rouge_matrix']


	total_data = torch.sum(input_length)
	end = torch.clamp(torch.cumsum(padded_lengths,1),0,input_length[0])
	begin = torch.cat((torch.zeros((len(input_length),1),dtype=torch.long),end[:,:-1]),1)

	if torch.cuda.is_available():
		document = document.to(device)
		label = label.to(device)
		input_length = input_length.to(device)
		indicators = indicators.to(device)
		end = end.to(device)
		begin= begin.to(device)
		# summary_representation = summary_representation.to(device)



	out= model(document,input_length,indicators,begin,end,device)


	output,_ = pad_packed_sequence(document)

	out1 = out.squeeze(-1)
	scores = sigmoid(out1)
	scores = scores.permute(1,0)

	mask = label.gt(-1).float()

	# Option 1
	loss_ce = F.binary_cross_entropy_with_logits(out,label,weight = mask,reduction='sum',pos_weight=pos_weight)
	o1 = output.permute(1,0,2).unsqueeze(3)
	o1.requires_grad = False
	# sim_mat =  F.cosine_similarity(o1,o1.permute(0,3,2,1),dim=2)
	# try:
	# 	sim_mat =  torch.stack([F.cosine_similarity(o1[i],o1[i].permute(2,1,0),dim=1) for i in range(o1.size()[0])],0)
	# except:
	
	o1 = o1.cpu()

	mask = (torch.eye(o1.shape[1], o1.shape[1])==1)
	sim_mat =  torch.stack([F.cosine_similarity(o1[i],o1[i].permute(2,1,0),dim=1).masked_fill_(mask, 0) for i in range(o1.size()[0])],0)
	sim_mat = sim_mat.to(device)
	loss_redundancy = torch.sum(torch.bmm(torch.bmm(scores.unsqueeze(1), sim_mat),scores.unsqueeze(2)))

	loss = (1-beta)*loss_ce+beta*loss_redundancy

	model.zero_grad()
	loss.backward()

	optimizer.step()
	l = loss.data
	del document,label,input_length,indicators,end,begin,loss,out, sim_mat,loss_redundancy
	torch.cuda.empty_cache()
	return l,total_data


def eval_seq2seq(val_dataloader,model,hyp_path,ref_path,word_length_limit, sent_length_limit,\
				pos_weight,device,remove_stopwords,stemmer,\
				meteor=False,lcs=False,saveScores=False,save_filename=None,eval_red=False,\
				use_trigramblock=False,use_mmr=False,lambd=0,\
				use_neusum=False):
	model.eval()
	total_loss=0
	total_data=0
	hyp_path_list = []
	ref_path_list = []
	total_correct = 0
	all_selections = []
	all_oracle = []
	all_sections=[]
	all_ids=[]
	# all_attn_weight = []

	sigmoid = torch.nn.Sigmoid()
	for i,data in enumerate(val_dataloader):
		if use_mmr:
			summaryfiles,referencefiles,loss,num_data,select_ids,oracle,sections = eval_seq2seq_batch_rl(sigmoid,data, model,hyp_path,ref_path, \
																						word_length_limit, sent_length_limit,pos_weight,device,\
																						lambd=lambd)
		elif use_neusum:
			summaryfiles,referencefiles,loss,num_data,select_ids,oracle,sections = eval_seq2seq_batch_neusum(data, model,hyp_path,ref_path, \
																						word_length_limit, sent_length_limit,pos_weight,device)
		else:
			summaryfiles,referencefiles,loss,num_data,select_ids,oracle,sections = eval_seq2seq_batch(sigmoid,data, model,hyp_path,ref_path, \
																						word_length_limit, sent_length_limit,pos_weight,device,\
																						use_trigramblock)		

		hyp_path_list.extend(summaryfiles)
		ref_path_list.extend(referencefiles)
		# all_attn_weight.extend(attn_weight_batch)
		all_selections.extend(select_ids)
		all_ids.extend(data['id'])
		# all_sections.extend(sections)
		# all_oracle.extend(oracle)
		total_loss+=loss
		total_data+=num_data
		del data
		del loss
		# break

		# if i%200==1:
		# 	print('Batch %d, Loss: %f'%(i,total_loss/float(total_data)))
		

	rouge2,df = get_rouge(hyp_path_list, ref_path_list, remove_stopwords,stemmer,lcs)
	if eval_red:
		all_unigram_ratio,all_bigram_ratio,all_trigram_ratio,all_redundancy = get_redundancy_scores(hyp_path_list)

	if meteor:
		model_type = type(model).__name__
		get_meteor(hyp_path_list, ref_path_list,model_type)
	if saveScores:
		all_sections.append([0])
		# all_oracle.append([0])
		all_selections.append([0])
		all_ids.append('avg')
		all_unigram_ratio.append(sum(all_unigram_ratio)/len(all_unigram_ratio))
		all_bigram_ratio.append(sum(all_bigram_ratio)/len(all_bigram_ratio))
		all_trigram_ratio.append(sum(all_trigram_ratio)/len(all_trigram_ratio))
		all_redundancy.append(sum(all_redundancy)/len(all_redundancy))
		# all_attn_weight.append([0])
		df['id'] = pd.Series(all_ids,index =df.index)
		df['selections'] = pd.Series(np.array(all_selections),index =df.index)
		df['unigram_ratio'] = pd.Series(np.array(all_unigram_ratio),index =df.index)
		df['bigram_ratio'] = pd.Series(np.array(all_bigram_ratio),index =df.index)
		df['trigram_ratio'] = pd.Series(np.array(all_trigram_ratio),index =df.index)
		df['redundancy_score'] = pd.Series(np.array(all_redundancy),index =df.index)

		# df['oracle'] = pd.Series(np.array(all_oracle),index =df.index)
		# df['sections'] = pd.Series(np.array(all_sections),index =df.index)
		# df['attn_weight'] = pd.Series(np.array(all_attn_weight),index =df.index)
		df.to_csv('%s.csv'%(save_filename))
	return rouge2, total_loss/float(total_data)

def eval_seq2seq_batch_rl(sigmoid,data_batch,model,hyp_path, ref_path, word_length_limit, sent_length_limit, pos_weight,device,lambd=1):
	document = data_batch['document']
	label = data_batch['labels']
	input_length = data_batch['input_length']
	indicators = data_batch['indicators']
	padded_lengths = data_batch['padded_lengths']
	# rouge_matrix_batch =data_batch['rouge_matrix']

	total_data = torch.sum(input_length)
	end = torch.clamp(torch.cumsum(padded_lengths,1),0,input_length[0])
	begin = torch.cat((torch.zeros((len(input_length),1),dtype=torch.long),end[:,:-1]),1)
	if torch.cuda.is_available():
		document = document.to(device)
		label = label.to(device)
		input_length = input_length.to(device)
		indicators = indicators.to(device)
		end = end.to(device)
		begin= begin.to(device)



	reference = data_batch['refs']

	ids = data_batch['id']


	out= model(document,input_length,indicators,begin,end,device)


	output,_ = pad_packed_sequence(document)
	mask = label.gt(-1).float()
	loss = F.binary_cross_entropy_with_logits(out,label,weight = mask,reduction='sum',pos_weight=pos_weight)
	out = out.squeeze(-1)
	scores = sigmoid(out).data
	scores = scores.permute(1,0)

	summaryfile_batch, reffile_batch,selections= predict_redundancy_max(scores, ids, data_batch['sent_txt'], hyp_path, ref_path,reference, word_length_limit, sent_length_limit,output,device,lambd)
	label = label.squeeze(-1)
	label = label.permute(1,0)

	all_oracle = [list((label[i]==1).nonzero().squeeze(-1).cpu().numpy()) for i in range(label.shape[0])]
	sections = [list(torch.unique(end[i],sorted=True).cpu().numpy()) for i in range(end.shape[0])]
	del document,label,input_length,indicators,end,begin
	return summaryfile_batch,reffile_batch,loss.data,total_data,selections,all_oracle,sections


def eval_seq2seq_batch_neusum(data_batch, model,hyp_path,ref_path, word_length_limit, sent_length_limit,\
								pos_weight,device):
	regression_crit = nn.KLDivLoss(reduction='none')
	document = data_batch['document']
	selections = data_batch['selections']
	scores_gain = data_batch['scores_gain']
	input_length = data_batch['input_length']
	indicators = data_batch['indicators']
	padded_lengths = data_batch['padded_lengths']
	sentence_lengths = data_batch['sentence_lengths']
	ids = data_batch['id']

	reference = data_batch['refs']

	total_data = torch.sum(input_length)
	end = torch.clamp(torch.cumsum(padded_lengths,1),0,input_length[0])
	begin = torch.cat((torch.zeros((len(input_length),1),dtype=torch.long),end[:,:-1]),1)

	max_length = max(input_length)
	oracle_length = [s.shape[0] for s in selections]
	max_step = max(oracle_length)

	doc_sent_mask = torch.zeros(len(input_length),max_length)
	for i,l in enumerate(input_length):
		doc_sent_mask[i,:l]=1
	
	loss_mask = torch.zeros(len(oracle_length),max_step,max_length)
	for i,l in enumerate(oracle_length):
		# batch*step
		loss_mask[i,:l]=doc_sent_mask[i]

	selections = pad_sequence(selections,batch_first=True,padding_value=-1) #batch*longest_oracle
	scores_gain =  make_scores_gain(scores_gain,max_step,max_length)# batch*step*doc_length
	# print(scores_gain.shape)

	if torch.cuda.is_available():
		document = document.to(device)
		input_length = input_length.to(device)
		indicators = indicators.to(device)
		end = end.to(device)
		begin= begin.to(device)
		doc_sent_mask = doc_sent_mask.to(device)
		scores_gain = scores_gain.to(device)
		loss_mask = loss_mask.to(device)

		# summary_representation = summary_representation.to(device)

	doc_sent_scores = model(document,input_length,indicators,begin,end,device,max(40,max_step),selections,doc_sent_mask)
	# print(doc_sent_scores.shape)
	loss = regression_loss(doc_sent_scores[:max_step], scores_gain, loss_mask, regression_crit)
	doc_sent_scores = doc_sent_scores.permute(2,0,1)
	# print(doc_sent_scores.shape)
	summaryfile_batch, reffile_batch,selections= predict_neusum(doc_sent_scores, ids, data_batch['sent_txt'], hyp_path, ref_path,reference,word_length_limit, sent_length_limit)

	l = loss.data
	torch.cuda.empty_cache()

	# all_oracle = [list((label[i]==1).nonzero().squeeze(-1).cpu().numpy()) for i in range(label.shape[0])]
	all_oracle=None
	sections = [list(torch.unique(end[i],sorted=True).cpu().numpy()) for i in range(end.shape[0])]
	del document,input_length,indicators,end,begin,loss,doc_sent_mask,scores_gain,loss_mask,doc_sent_scores
	return summaryfile_batch,reffile_batch,l,total_data,selections,all_oracle,sections


def eval_seq2seq_batch(sigmoid,data_batch,model,hyp_path,ref_path, word_length_limit, sent_length_limit,\
						pos_weight,device,use_trigramblock=False):
	document = data_batch['document']
	label = data_batch['labels']
	input_length = data_batch['input_length']
	indicators = data_batch['indicators']
	padded_lengths = data_batch['padded_lengths']

	total_data = torch.sum(input_length)
	end = torch.clamp(torch.cumsum(padded_lengths,1),0,input_length[0])
	begin = torch.cat((torch.zeros((len(input_length),1),dtype=torch.long),end[:,:-1]),1)
	if torch.cuda.is_available():
		document = document.to(device)
		label = label.to(device)
		input_length = input_length.to(device)
		indicators = indicators.to(device)
		end = end.to(device)
		begin= begin.to(device)


	reference = data_batch['refs']

	ids = data_batch['id']

	out= model(document,input_length,indicators,begin,end,device)


	mask = label.gt(-1).float()
	loss = F.binary_cross_entropy_with_logits(out,label,weight = mask,reduction='sum',pos_weight=pos_weight)
	out = out.squeeze(-1)
	scores = sigmoid(out).data
	scores = scores.permute(1,0)

	if use_trigramblock:
		summaryfile_batch, reffile_batch,selections= predict_trigram_block(scores, ids, data_batch['sent_txt'], hyp_path, ref_path,reference,word_length_limit, sent_length_limit)
	else:
		summaryfile_batch, reffile_batch,selections= predict(scores, ids, data_batch['sent_txt'], hyp_path, ref_path,reference,word_length_limit, sent_length_limit)

	label = label.squeeze(-1)
	label = label.permute(1,0)

	all_oracle = [list((label[i]==1).nonzero().squeeze(-1).cpu().numpy()) for i in range(label.shape[0])]
	sections = [list(torch.unique(end[i],sorted=True).cpu().numpy()) for i in range(end.shape[0])]
	del document,label,input_length,indicators,end,begin
	return summaryfile_batch,reffile_batch,loss.data,total_data,selections,all_oracle,sections

def predict(score_batch, ids, src_txt_list, hyp_path, ref_path,tgt_txt, word_length_limit, sent_length_limit):
	#score_batch = [batch,seq_len]
	summaryfile_batch = []
	reffile_batch = []
	# all_ids = []
	selections = []

	for i in range(len(src_txt_list)):
		summary = []
		scores = score_batch[i,:len(src_txt_list[i])]
		sorted_linenum = [x for _,x in sorted(zip(scores,list(range(len((src_txt_list[i]))))),reverse=True)]
		selected_ids = [] 
		wc = 0
		uc = 0
		for j in sorted_linenum:
			selected_ids.append(j)
			summary.append(' '.join(src_txt_list[i][j]))
			wc+=len(src_txt_list[i][j])
			uc+=1

			if uc>=sent_length_limit:
				break
			if wc>=word_length_limit:
				break

		summary='\n'.join(summary)
		selections.append(selected_ids)


		fname = hyp_path+ids[i]+'.txt'
		of = open(fname,'w')
		of.write(summary)
		of.close()
		summaryfile_batch.append(fname)

		refname = ref_path+ids[i]+'.txt'
		of = open(refname,'w')
		of.write(tgt_txt[i])
		of.close()
		reffile_batch.append(refname)

	return summaryfile_batch, reffile_batch,selections

def predict_neusum(score_batch, ids, src_txt_list, hyp_path, ref_path,tgt_txt, word_length_limit, sent_length_limit):
	#score_batch = [batch,step,doc_length]
	summaryfile_batch = []
	reffile_batch = []
	# all_ids = []
	selections = []
	for i in range(len(src_txt_list)):
		summary = []
		
		selected_ids = [] 
		wc = 0
		uc = 0
		step = 0
		while len(selected_ids)<=len(src_txt_list[i]):
			scores = score_batch[i,step,:len(src_txt_list[i])]
			# sorted_linenum = [x for _,x in sorted(zip(scores,list(range(len((src_txt_list[i]))))),reverse=True)]
			j = torch.argmax(scores).item()
			selected_ids.append(j)
			summary.append(' '.join(src_txt_list[i][j]))
			wc+=len(src_txt_list[i][j])
			uc+=1
			step+=1
			# print(step)
			if uc>=sent_length_limit:
				break
			if wc>=word_length_limit:
				break

		summary='\n'.join(summary)
		selections.append(selected_ids)


		fname = hyp_path+ids[i]+'.txt'
		of = open(fname,'w')
		of.write(summary)
		of.close()
		summaryfile_batch.append(fname)

		refname = ref_path+ids[i]+'.txt'
		of = open(refname,'w')
		of.write(tgt_txt[i])
		of.close()
		reffile_batch.append(refname)

	return summaryfile_batch, reffile_batch,selections

def predict_trigram_block(score_batch, ids, src_txt_list, hyp_path, ref_path,tgt_txt, word_length_limit, sent_length_limit,attn_weight = None):
	#score_batch = [batch,seq_len]
	correct_num = 0
	summaryfile_batch = []
	reffile_batch=[]
	selections = []
	for i in range(len(src_txt_list)):
		summary = []
		scores = score_batch[i,:len(src_txt_list[i])]
		sorted_linenum = [x for _,x in sorted(zip(scores,list(range(len((src_txt_list[i]))))),reverse=True)]
		selected_ids = []
		uc = 0
		wc = 0
		current_trigrams = set()
		for j in sorted_linenum:
			check,tmp_current_trigrams = check_trigram(current_trigrams, src_txt_list[i][j], edu=False)
			# no trigram overlap
			if check:
				current_trigrams = tmp_current_trigrams
			else:
				continue
			summary.append(' '.join(src_txt_list[i][j]))
			selected_ids.append(j)
			wc+=len(src_txt_list[i][j])
			uc+=1
			if uc>=sent_length_limit:
				break
			if wc>=word_length_limit:
				break


		summary='\n'.join(summary)
		selections.append(selected_ids)


		fname = hyp_path+ids[i]+'.txt'
		of = open(fname,'w')
		of.write(summary)
		summaryfile_batch.append(fname)

		refname = ref_path+ids[i]+'.txt'
		of = open(refname,'w')
		of.write(tgt_txt[i])
		of.close()
		reffile_batch.append(refname)

	return summaryfile_batch, reffile_batch,selections


def predict_redundancy_max(score_batch, ids, src_txt_list,hyp_path,ref_path, \
						tgt_txt, word_length_limit, sent_length_limit,\
						output,device,lamb,attn_weight = None):
	#score_batch = [batch,seq_len]
	correct_num = 0
	summaryfile_batch = []
	reffile_batch=[]
	selections=[]
	# dim = output.size()[-1]
	# dim=output[0].size()[-1]
	for i in range(len(src_txt_list)):
		summary = []

		scores = score_batch[i,:len(src_txt_list[i])]
		sorted_linenum = [x for _,x in sorted(zip(scores,list(range(len(src_txt_list[i])))),reverse=True)]
		wc = 0
		uc=0
		selected_ids = [] 

		summary_representation=[]

		###### sent representation
		all_sent = output[:scores.size()[0],i,:].unsqueeze(2)


		while len(selected_ids)<=len(src_txt_list[i]):
			j = sorted_linenum[0]
			summary.append(' '.join(src_txt_list[i][j]))
			selected_ids.append(j)

			
			###### sent representation
			summary_representation.append(output[j,i,:])
			s = torch.stack(summary_representation,1).unsqueeze(0)


			redundancy_score =torch.max(F.cosine_similarity(all_sent,s,1),1)[0]
			# print(redundancy_score)
			# print(redundancy_score)
			scores[j] = -100
			final_scores = lamb*scores - ((1-lamb)*redundancy_score)
			sorted_linenum = [x for _,x in sorted(zip(final_scores,list(range(len(src_txt_list[i])))),reverse=True)]

			wc+=len(src_txt_list[i][j])
			uc+=1
			if uc>=sent_length_limit:
				break
			if wc>=word_length_limit:
				break

		summary='\n'.join(summary)
		selections.append(selected_ids)


		fname = hyp_path+ids[i]+'.txt'
		of = open(fname,'w')
		of.write(summary)
		summaryfile_batch.append(fname)

		refname = ref_path+ids[i]+'.txt'
		of = open(refname,'w')
		of.write(tgt_txt[i])
		of.close()
		reffile_batch.append(refname)

	return summaryfile_batch, reffile_batch,selections
