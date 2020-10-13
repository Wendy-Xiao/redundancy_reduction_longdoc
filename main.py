from torch import nn
from collections import Counter
from random import random
from nltk import word_tokenize
from torch.autograd import Variable
import pandas as pd
import sys
import re
import numpy as np
import torch
import torch.nn.functional as F
import os
import json 
import random
import argparse
from data import *
from utils import *
from run import *
from models import *

from timeit import default_timer as timer

# Parse the arguments
parser = argparse.ArgumentParser()

# For model
parser.add_argument("--seed", type=int, default=None, help= "Set the seed of pytorch, so that you can regenerate the result.")
parser.add_argument("--model",type = str, default = 'ac', help = "The name of the model to train")
parser.add_argument("--modelpath",type = str, default = './model/', help = "The path of save the model")
parser.add_argument("--cell", default='gru', help="Choose one from gru, lstm")
parser.add_argument("--embedding_dim", type=int, default = 300, help = "Set the dimension of word_embedding")
parser.add_argument("--hidden_dim", type=int, default = 300, help = "Set the dimension of hidden state")
parser.add_argument("--mlp_size", type=int, default = 100, help = "Set the dimension of the integrated mlp layer")
parser.add_argument("--pretrained_model_path", type=str, default = '', help = "The pretrained model as the initial state of the MMR-Select+")

# For data
parser.add_argument("--batchsize", type=int, default = 32, help = "Set the size of batch")
parser.add_argument("--vocab_size", type=int, default = 50000, help = "vocabulary size")
parser.add_argument("--dataset", type=str, default = 'pubmed', help = "Dataset to train on, pubmed/arxiv")
parser.add_argument("--datapath", type=str, default = './', help = "the path to the dataset")
parser.add_argument("--gloveDir", type=str, default = '../', help = "Directory storing glove embedding")

# For learning
parser.add_argument("--learning_rate",type=float, default=1e-4,help="Whether use neusum decoder")
parser.add_argument("--device", type=int, default = 1, help = "device used to compute")

# For different redundancy reduction method.
parser.add_argument("--use_trigram_block", default=False, action='store_true', help="Whether use trigram block")
parser.add_argument("--use_newloss",default=False, action='store_true',help="Whether use new loss to train")
parser.add_argument("--use_rl",default=False, action='store_true',help="Whether use rl to train")
parser.add_argument("--use_mmr",default=False, action='store_true',help="Whether use mmr to evaluate")
parser.add_argument("--beta",type=float, default=0.30,help="Beta used in the new loss/rl")
parser.add_argument("--lambd",type=float, default=0.6,help="Lambd used in the rl")
parser.add_argument("--gamma",type=float, default=0.99,help="Lambd used in the rl")
parser.add_argument("--use_neusum",default=False, action='store_true',help="Whether use neusum decoder")


# For evaluation
parser.add_argument("--remove_stopwords", default=False,action='store_true', help = "if add this flag, then set remove_stopwords to be true")
parser.add_argument("--stemmer", default=False, action='store_true', help = "if add this flag, then set stemmer to be true")
parser.add_argument("--word_length_limit",type=int, default=200,help="The word limit of generated summary.")
parser.add_argument("--sent_length_limit",type=int, default=1000,help="The sentence limit of generated summary.")


args = parser.parse_args()
print(args)

train_input_path='%s/%s/train/'%(args.datapath,args.dataset)
val_input_path='%s/%s/valid/'%(args.datapath,args.dataset)
# if seed is given, set the seed for pytorch on both cpu and gpu
if args.seed:
	torch.manual_seed(args.seed)

# reference path and the temorary path to store the generated summaries of validation set
hyp_path = './tmp/%s/eval_hyp/'%(args.model)
ref_path = './tmp/%s/eval_ref'%(args.model)
if not os.path.exists(hyp_path):
	os.makedirs(hyp_path)
if not os.path.exists(ref_path):
	os.makedirs(ref_path)

# set the device the model running on
device = torch.device("cuda:%d"%(args.device))
torch.cuda.set_device(args.device)

train_neusum_path = None
val_neusum_path = None
use_neusum=False
if args.model=='ac_neusum':
	use_neusum=True
	train_neusum_path = '%s/%s/neusum_target/train/'%(args.dataset)
	val_neusum_path = '%s/%s/neusum_target/valid/'%(args.dataset)

# build the vocabulary dictionary
if os.path.exists('./vocabulary_%s.json'%(args.dataset)):
# if 'vocabulary_%s.json'%(args.dataset) in [path.name for path in Path('./').glob('*.json')]:
    with open('./vocabulary_%s.json'%(args.dataset),'r') as f:
        w2v = json.load(f)
    print('Load vocabulary from vocabulary_%s.json'%(args.dataset))
else: 
    all_tokens=get_all_text(train_input_dir)
    w2v = build_word2ind(all_tokens, args.vocab_size)
    with open('vocabulary_%s.json'%(args.dataset),'w') as f:
        json.dump(w2v,f)
sys.stdout.flush()

# get the pos weight, used in the loss function
# pos_weight = get_posweight(train_label_dir,args.train_file_list)
pos_weight = get_posweight(train_input_path)
# pos_weight = torch.FloatTensor([48.59])
if torch.cuda.is_available():
	pos_weight=pos_weight.to(device)

# build embedding matrix
gloveDir = args.gloveDir
embedding_matrix = getEmbeddingMatrix(gloveDir, w2v, args.embedding_dim)

# # set the dataset and dataloader for both training and validation set.
train_dataset = SummarizationDataset(w2v,embedding_matrix, args.embedding_dim,train_input_path,ref_required=True,\
					to_shuffle=True,is_test=False,\
					useNeusum=use_neusum,neusum_path=train_neusum_path)
train_dataloader = SummarizationDataLoader(train_dataset,batch_size=args.batchsize,useNeusum=use_neusum)

val_dataset = SummarizationDataset(w2v,embedding_matrix, args.embedding_dim,val_input_path,ref_required=True,\
					to_shuffle=False,is_test=True,\
					useNeusum=use_neusum,neusum_path=val_neusum_path)
val_dataloader = SummarizationDataLoader(val_dataset,batch_size=args.batchsize,useNeusum=use_neusum)



model_name = '%s_%s'%(args.dataset, args.model)
if args.model =='ac':
	model = Attentive_context(args.embedding_dim,args.hidden_dim, args.mlp_size,  cell_type=args.cell)
elif args.model =='ac_sr':
	model = Attentive_context_sr(args.embedding_dim,args.hidden_dim, args.mlp_size,  cell_type=args.cell)
elif args.model =='ac_neusum':
	model = NeuSum(args.embedding_dim,args.hidden_dim, args.mlp_size,  cell_type=args.cell)

if args.use_newloss:
	model_name+='_newloss_beta=%.2f'%(args.beta)
if args.use_rl:
	model_name+='_mmr+_lambd=%.2f_gamma=%.2f'%(args.lambd,args.gamma)

# set the directory to store models, make new if not exists
if not os.path.exists(args.modelpath):
	os.makedirs(args.modelpath)
MODEL_DIR = '%s/%s_best'%(args.modelpath, model_name)
print(model_name)


sys.stdout.flush()

# define the optimizer
optimizer = torch.optim.Adam(model.parameters(),lr = args.learning_rate,weight_decay=1e-5, betas=(0.9, 0.98), eps=1e-09)
best_r2 = 0
best_ce = 1000
train_loss=[]
val_loss = []
# lamb=0.6
# lamb=1
# beta=0.03
print('Start Training!')
time_start = timer()
time_epoch_end_old = time_start
total_step=0
total_loss=0
total_data=0

if torch.cuda.is_available():
	model=model.to(device)

eval_red = False
if args.use_rl:
	eval_red = True
	model.load_state_dict(torch.load(args.pretrained_model_path,map_location=device))
model.train()


for i,data in enumerate(train_dataloader):

	if args.use_newloss:
		l,num_data = train_seq2seq_batch_newloss(data,model,optimizer,pos_weight,device,beta=args.beta)
	elif args.use_rl:
		l,num_data = train_seq2seq_batch_rl(data, model, optimizer,pos_weight,device,lamb=args.lambd,gamma=args.gamma)
	elif args.model=='ac_neusum':
		l,num_data = train_seq2seq_batch_neusum(data, model, optimizer,pos_weight,device)
	else:
		l,num_data = train_seq2seq_batch(data,model,optimizer,pos_weight,device)

	total_loss+=l
	total_data+=num_data
	total_step+=1
	# Record loss 
	if total_step%200==0:
		print('Step %d, Loss: %f'%(total_step,total_loss/float(total_data)))
		sys.stdout.flush()

	# Validate
	if total_step%2000==0:
		r2, l = eval_seq2seq(val_dataloader,model,hyp_path,ref_path, args.word_length_limit, args.sent_length_limit,\
							pos_weight,device,args.remove_stopwords,args.stemmer,\
							use_trigramblock=args.use_trigram_block,use_neusum=use_neusum,\
							use_mmr=args.use_rl, lambd=args.lambd,\
							eval_red=eval_red)


		print('Validation loss: %f'%(l))
		if r2>best_r2:
			best_r2 = r2
			torch.save(model.state_dict(), MODEL_DIR)
			print('Saved as best model - highest r2.')
		if l<=best_ce:
			best_ce = l
			print('Lowest ce!')
		model.train()
		time_epoch_end_new = timer()
		print ('Seconds to execute to 2000 batches: ' + str(time_epoch_end_new - time_epoch_end_old))
		time_epoch_end_old = time_epoch_end_new
		sys.stdout.flush()

	if total_step==100000:
		break

print('Seconds to execute to whole training procedure: ' + str(time_epoch_end_old - time_start))




