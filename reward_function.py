from scipy.stats import rv_discrete
import torch
import torch.nn.functional as F
import numpy as np
from utils import *


def greedy_max(doc_length,px,sentence_embed,sentences,device,sentence_lengths,length_limit=200,lamb=0.2):
	'''
	prob: sum should be 1
	sentence embed: [doc_length, embed_dim]
	'''
	x = list(range(doc_length))
	px = px.cpu().numpy()
	score=px
	prob = 1
	summary_representation = []
	bias = np.ones(px.shape)
	selected = []
	wc=0
	lengths=[]
	summary = []
	while wc<=length_limit:
		sample = np.argmax(score)

		selected.append(sample)
		wc+=sentence_lengths[sample]
		lengths.append(sentence_lengths[sample])
		summary.append(sentences[sample])

		summary_representation.append(sentence_embed[sample])
		s = torch.stack(summary_representation,1).unsqueeze(0)
		all_sent = sentence_embed[:doc_length,:].unsqueeze(2)
		redundancy_score =torch.max(F.cosine_similarity(all_sent,s,1),1)[0].cpu().numpy()

		score = lamb*px - ((1-lamb)*redundancy_score) + (1-lamb)*bias
		for i_sel in selected:
			score[i_sel] = 0
		# print(len(selected))
	summary ='\n'.join(summary)
	# summary_representation= summary_representation.to(device)
	return summary, prob, selected


def greedy_nommr(doc_length,px,sentence_embed,sentences,device,sentence_lengths,length_limit=200,lamb=0.2):
	'''
	prob: sum should be 1
	sentence embed: [doc_length, embed_dim]
	'''
	x = list(range(doc_length))
	px = px.cpu().numpy()
	score=px
	prob = 1
	bias = np.ones(px.shape)
	summary_representation = []

	selected = []
	wc=0
	lengths = []
	summary=[]
	while wc<=length_limit:

		sample = np.argmax(score)
		selected.append(sample)
		wc+=sentence_lengths[sample]
		lengths.append(sentence_lengths[sample])
		summary.append(sentences[sample])

		for i_sel in selected:
			score[i_sel] = 0
	summary = '\n'.join(summary)
	return summary, prob, selected


def compute_reward(score_batch,input_lengths,output,sentences_batch,reference_batch,device,sentence_lengths_batch,number_of_sample=5,lamb=0.1):
	reward_batch = []
	rl_label_batch = torch.zeros(output.size()[:2]).unsqueeze(2)
	for i_data in range(len(input_lengths)):
		# summary_i = summary_embed[i_data]
		doc_length = input_lengths[i_data]
		scores = score_batch[i_data,:doc_length]
		sentence_lengths = sentence_lengths_batch[i_data]
		sentence_embed = output[:doc_length,i_data,:]
		sentences = sentences_batch[i_data]
		reference = reference_batch[i_data]

		# final_choice = None
		result,prob,selected = greedy_nommr(doc_length,scores,sentence_embed,sentences,device,sentence_lengths,lamb = lamb)
		reward_greedy = get_rouge_single(result,reference)

		result,prob,selected = greedy_max(doc_length,scores,sentence_embed,sentences,device,sentence_lengths,lamb = lamb)
		reward_hi = get_rouge_single(result,reference)
		final_choice = selected

		# print(reward_hi-reward_greedy)
		reward_batch.append(reward_hi-reward_greedy)
		rl_label_batch[final_choice,i_data,:] = 1

	reward_batch = torch.FloatTensor(reward_batch).unsqueeze(0).to(device)
	rl_label_batch = rl_label_batch.to(device)
	reward_batch.requires_grad_(False)

	return reward_batch,rl_label_batch


