import os
from nltk.util import ngrams
from nltk.corpus import stopwords 
import json
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import entropy
import numpy as np
from utils import getEmbeddingMatrix
import torch.nn.functional as F
import rouge
from nltk import word_tokenize
import scipy.stats
import rouge_papier_v2
import pandas as pd
from dataset_statistics import *
import torch
import sys


def unsupervised_mmr(embedding_matrix,data_dir,output_dir,VOCABULARY_SIZE,gloveDir,length_limit=200,lamb=0.6,device=-1):
	data_dir = Path(data_dir)
	all_files = [path for path in data_dir.glob("*.pt")]
	sum_sim = 0
	hyp_paths=[]
	ref_paths=[]
	if not os.path.exists(output_dir+'/hyp/'):
		os.makedirs(output_dir+'/hyp/')
	if not os.path.exists(output_dir+'/ref/'):
		os.makedirs(output_dir+'/ref/')
	for f in all_files:
		all_data = torch.load(f)
		for data in all_data:
			fid = data['doc_id']
			document_l =[]
			query = torch.zeros((300))
			total_wc=0
			for s in data['sent_txt']:
				sent_l = [w2v.get(w,0) for w in s]
				if len(sent_l)==0:
					sent_l=[0]
				sent_embed = torch.FloatTensor(embedding_matrix[sent_l,:])
				query+=torch.sum(sent_embed,0)
				total_wc+=len(sent_l)
				sent_embed = torch.mean(sent_embed,0)
				# if device !=-1:
				# 	sent_embed.to(device)
				document_l.append(sent_embed)
			query = query/total_wc
			doc_embed = torch.stack(document_l,0).unsqueeze(2)
			doc_embed_tr = doc_embed.permute(2,1,0)
			# print(doc_embed.shape)
			sim_mat = torch.FloatTensor(F.cosine_similarity(doc_embed,doc_embed_tr,1))
			relevance = F.cosine_similarity(query.unsqueeze(0),doc_embed.squeeze(),1)
			score = relevance
			selected=[]
			select_wc = 0
			summary=[]
			while len(selected)<=len(data['sent_txt']):
				isent = torch.argmax(score).item()
				select_wc+=len(data['sent_txt'][isent])
				summary.append(' '.join(data['sent_txt'][isent]))
				selected.append(isent)
				redundancy_score = torch.max(sim_mat[selected],0)[0]
				score = lamb*relevance-(1-lamb)*redundancy_score
				for i_sample in selected:
					score[i_sample]=-100
				if select_wc>=length_limit:
					break


			hyp_file = output_dir+'/hyp/'+fid+'.txt'
			with open(hyp_file,'w') as of:
				of.write('\n'.join([sent.strip() for sent in summary]))

			ref_file = output_dir+'/ref/'+fid +'.txt'
			with open(ref_file,'w') as of:
				of.write('\n'.join([sent.strip() for sent in data['tgt_list_str']]))
			hyp_paths.append(hyp_file)
			ref_paths.append(ref_file)


	print('ROUGE scores:')
	sys_avg = get_rouge(hyp_paths, ref_paths)
	print('Redundancy scores: ')
	get_redundancy_scores(hyp_paths)
	return

if __name__ == '__main__':
	VOCABULARY_SIZE = 50000
	gloveDir = '../'
	for dataset in ['pubmed','arxiv']:
		doc_dir = '/scratch/wenxiao/scientific_paper_dataset/%s/test/'%(dataset)
		output_dir = '/scratch/wenxiao/tmp/unsupervised_mmr_%s/'%(dataset)

		with open('vocabulary_%s_full.json'%(dataset),'r') as f:
			w2v = json.load(f)
		gloveDir = gloveDir
		embedding_matrix = getEmbeddingMatrix(gloveDir, w2v, 300)

		# for lamb in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
		# 	print('Lambda is %f:'%(lamb))
		# 	unsupervised_mmr(embedding_matrix,doc_dir,output_dir,VOCABULARY_SIZE,gloveDir,length_limit=200,lamb=lamb,device=-1)
		# 	sys.stdout.flush()
		lamb=0.6
		unsupervised_mmr(embedding_matrix,doc_dir,output_dir,VOCABULARY_SIZE,gloveDir,length_limit=200,lamb=lamb,device=-1)
		sys.stdout.flush()
