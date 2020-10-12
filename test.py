from torch import nn
from collections import Counter
from random import random
from nltk import word_tokenize
from torch.autograd import Variable
import pandas as pd
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
import sys


# Parse the arguments
parser = argparse.ArgumentParser()

# For model
parser.add_argument("--model",type = str, default = 'ac', help = "The path to save models")
parser.add_argument("--modelpath",type = str, default = './pretrained_models/', help = "The path of save the model")
parser.add_argument("--cell", default='gru', help="Choose one from gru, lstm")
parser.add_argument("--embedding_dim", type=int, default = 300, help = "Set the dimension of word_embedding")
parser.add_argument("--hidden_dim", type=int, default = 300, help = "Set the dimension of hidden state")
parser.add_argument("--mlp_size", type=int, default = 100, help = "Set the dimension of the integrated mlp layer")

# For data
parser.add_argument("--batchsize", type=int, default = 32, help = "Set the size of batch")
parser.add_argument("--dataset", type=str, default = 'pubmed', help = "Dataset to train on, pubmed/arxiv")
parser.add_argument("--datapath", type=str, default = './', help = "the path to the dataset")
parser.add_argument("--gloveDir", type=str, default = '../', help = "Directory storing glove embedding")

# For different redundancy reduction methods
parser.add_argument("--use_trigram_block", default=False, action='store_true', help="Whether use trigram block")
parser.add_argument("--use_mmr",default=False, action='store_true',help="Whether use mmr-select")
parser.add_argument("--use_rl",default=False, action='store_true',help="Whether use mmr-select+")
parser.add_argument("--use_newloss",default=False, action='store_true',help="Whether use new loss to train")
parser.add_argument("--beta",type=float, default=0.99,help="Beta used in the new loss/rl")
parser.add_argument("--lambd",type=float, default=0.6,help="Lambda used in mmr")

# For evaluation
parser.add_argument("--device", type=int, default = 3, help = "device used to compute")
parser.add_argument("--remove_stopwords", action='store_true', help = "if add this flag, then set remove_stopwords to be true")
parser.add_argument("--stemmer", action='store_true', help = "if add this flag, then set stemmer to be true")
parser.add_argument("--word_length_limit",type=int, default=200,help="The word limit of generated summary.")
parser.add_argument("--sent_length_limit",type=int, default=1000,help="The sentence limit of generated summary.")


args = parser.parse_args()
print(args)


# Set the refpath (human-abstraction) and hyp-path(to store the generated summary)
if torch.cuda.is_available():
    device = torch.device("cuda:%d"%(args.device))
    torch.cuda.set_device(args.device)
else:
    device=torch.device("cpu")

test_input_path='%s/%s/test/'%(args.datapath, args.dataset)


hyp_path = './tmp/%s/eval_hyp/'%(args.model)
ref_path = './tmp/%s/eval_ref/'%(args.model)
if not os.path.exists(hyp_path):
    os.makedirs(hyp_path)
if not os.path.exists(ref_path):
    os.makedirs(ref_path)

word_length_limit, sent_length_limit = args.word_length_limit, args.sent_length_limit

# pos_weight = get_posweight(train_input_path)
pos_weight = torch.FloatTensor([48.59])
if torch.cuda.is_available():
    pos_weight=pos_weight.to(device)

use_neusum=False
test_neusum_path=None
if args.model=='ac_neusum':
    use_neusum=True
    test_neusum_path = '%s/%s/neusum_target/test/'%(args.datapath, args.dataset)


with open('vocabulary_%s.json'%(args.dataset),'r') as f:
    w2v = json.load(f)

gloveDir = args.gloveDir
embedding_matrix = getEmbeddingMatrix(gloveDir, w2v, args.embedding_dim)


test_dataset = SummarizationDataset(w2v,embedding_matrix, args.embedding_dim,test_input_path,ref_required=True,\
                    to_shuffle=False,is_test=True,\
                    useNeusum=use_neusum,neusum_path=test_neusum_path)
# val_dataset = bigpatentDataset(w2v,embedding_matrix, EMBEDDING_DIM,val_input_dir,feature_name = 'section_lengths_0.3',target_dir=val_label_dir,reference_dir = ref_path)
test_dataloader = SummarizationDataLoader(test_dataset,batch_size=args.batchsize,useNeusum=use_neusum)



print('Start loading model.')
# Initialize the model

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
    model_name+='_mmr+_lambd=%.2f_beta=%.2f'%(args.lambd,args.beta)

print(model_name)

MODEL_DIR = '%s/%s_best'%(args.modelpath, model_name)
# Load the pre-trained model
model.load_state_dict(torch.load(MODEL_DIR,map_location=device))
# Move to GPU


model=model.to(device)



model.eval()
print('Start evaluating.')
save_filename = model_name
if args.use_trigram_block:
    save_filename = '%s_trigramblock'%(model_name)
elif args.use_mmr:
    save_filename = '%s_mmr_lambd=%.2f'%(model_name,args.lambd) 

if word_length_limit!=200:
    save_filename=save_filename+'_%d'%(word_length_limit)

r2, l = eval_seq2seq(test_dataloader,model,hyp_path,ref_path, word_length_limit, sent_length_limit,\
                            pos_weight,device,args.remove_stopwords,args.stemmer,\
                            use_trigramblock=args.use_trigram_block,use_mmr=args.use_mmr,lambd=args.lambd, \
                            use_neusum=use_neusum,eval_red=True,\
                            saveScores=True,save_filename=save_filename)
print('test loss: %f'%(l))
sys.stdout.flush()



