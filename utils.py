from collections import Counter
from pathlib import Path
from random import random
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords 
import rouge_papier_v2
import pandas as pd
import re
import numpy as np
import os
import json 
import torch
import os
import subprocess
import rouge
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import entropy

# Utility functions

def get_posweight(inputs_dir):
    inputs_dir = Path(inputs_dir)
    all_files = [path for path in inputs_dir.glob("*.pt")]
    total_num=0
    total_pos=0
    for i in range(10):
        data = torch.load(all_files[i])
        for d in data:
            total_num+=len(d['labels'])
            total_pos+=sum(d['labels'])
    print('Compute pos weight done! There are %d sentences in total, with %d sentences as positive'%(total_num,total_pos))
    return torch.FloatTensor([(total_num-total_pos)/float(total_pos)])


def make_file_list(input_dir,file_list_file):
    of = open(file_list_file,'r')
    file_list = of.readlines()
    of.close()
    f_list = [Path(input_dir+'/'+f.strip()+'.json') for f in file_list]
    return f_list

def get_all_text(train_input_dir):
    if isinstance(train_input_dir,list):
        file_l = train_input_dir
    else:
        train_input = Path(train_input_dir)
        file_l = [path for path in train_input.glob("*.json")]
    all_tokens = []
    for f in file_l:
        with f.open() as of:
            d = json.load(of)
        tokens = [t for sent in d['inputs'] for t in (sent['tokens']+['<eos>'])]
        all_tokens.append(tokens)
    return all_tokens

def build_word2ind(utt_l, vocabularySize):
    word_counter = Counter([word for utt in utt_l for word in utt])
    print('%d words found!'%(len(word_counter)))
    vocabulary = ["<UNK>"] + [e[0] for e in word_counter.most_common(vocabularySize)]
    word2index = {word:index for index,word in enumerate(vocabulary)}
    global EOS_INDEX
    EOS_INDEX = word2index['<eos>']
    return word2index

# Build embedding matrix by importing the pretrained glove
def getEmbeddingMatrix(gloveDir, word2index, embedding_dim):
    '''Refer to the official baseline model provided by SemEval.'''
    embeddingsIndex = {}
    # Load the embedding vectors from ther GloVe file
    embeddingMatrix = np.zeros((len(word2index) , embedding_dim),dtype='float32')
    with open(os.path.join(gloveDir, 'glove.6B.300d.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in word2index.keys():
                i = word2index[word]
                embeddingVector = np.asarray(values[1:], dtype='float32')
                embeddingMatrix[i] = embeddingVector
            # embeddingsIndex[word] = embeddingVector
    # Minimum word index of any word is 1. 
    # embeddingMatrix = np.zeros((len(word2index) , embedding_dim),dtype='float32')
    # for word, i in word2index.items():
    #     embeddingVector = embeddingsIndex.get(word)
    #     if embeddingVector is not None:
    #         # words not found in embedding index will be all-zeros.
    #         embeddingMatrix[i] = embeddingVector
    embeddingMatrix = torch.tensor(embeddingMatrix)
    return embeddingMatrix

def build_embedding_matrix(gloveDir):
    word2index={}
    embedding_matrix = []
    word2index['[unk]'] = 0
    embedding_matrix.append(np.zeros(300,dtype='float32'))
    idx = 1
    # Load the embedding vectors from ther GloVe file
    with open(os.path.join(gloveDir, 'glove.6B.300d.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddingVector = np.asarray(values[1:], dtype='float32')
            word2index[word] =idx
            embedding_matrix.append(embeddingVector)
            idx+=1
    # embedding_matrix = [np.zeros(embedding_matrix[0].shape,dtype='float32')]+embedding_matrix
    embedding_matrix = torch.tensor(embedding_matrix)
    return word2index,embedding_matrix


def get_rouge(hyp_pathlist, ref_pathlist,remove_stopwords,stemmer,lcs=False):
    path_data = []
    uttnames = []
    for i in range(len(hyp_pathlist)):
        path_data.append([hyp_pathlist[i], [ref_pathlist[i]]])
        uttnames.append(os.path.splitext(hyp_pathlist[i])[0].split('/')[-1])

    config_text = rouge_papier_v2.util.make_simple_config_text(path_data)
    config_path = './config'
    of = open(config_path,'w')
    of.write(config_text)
    of.close()
    uttnames.append('Average')
    df,avgfs,conf = rouge_papier_v2.compute_rouge(
        config_path, max_ngram=2, lcs=True, 
        remove_stopwords=remove_stopwords,stemmer=stemmer,set_length = False,return_conf=True)
    # df['data_ids'] = pd.Series(np.array(uttnames),index =df.index)
    avg = df.iloc[-1:].to_dict("records")[0]
    c = conf.to_dict("records")
    # if lcs:
    # print(c)
    print("Rouge-1 r score: %f, Rouge-1 p score: %f, Rouge-1 f-score: %f, 95-conf(%f-%f)"%(\
            avg['rouge-1-r'],avg['rouge-1-p'],avg['rouge-1-f'],c[0]['lower_conf_f'],c[0]['upper_conf_f']))
    print("Rouge-2 r score:%f, Rouge-2 p score: %f, Rouge-2 f-score:%f, 95-conf(%f-%f)"%(\
        avg['rouge-2-r'],avg['rouge-2-p'],avg['rouge-2-f'],c[1]['lower_conf_f'],c[1]['upper_conf_f']))
    print("Rouge-L r score:%f, Rouge-L p score: %f, Rouge-L f-score:%f, 95-conf(%f-%f)"%(\
        avg['rouge-L-r'],avg['rouge-L-p'],avg['rouge-L-f'],c[2]['lower_conf_f'],c[2]['upper_conf_f']))



    return avgfs[1],df

def output_to_dict(output):
    """
    Convert the ROUGE output into python dictionary for further
    processing.

    """
    #0 ROUGE-1 Average_R: 0.02632 (95%-conf.int. 0.02632 - 0.02632)
    pattern = re.compile(
        r"(\d+) (ROUGE-\S+) (Average_\w): (\d.\d+) "
        r"\(95%-conf.int. (\d.\d+) - (\d.\d+)\)")
    results = {}
    for line in output.split("\n"):
        match = pattern.match(line)
        if match:
            sys_id, rouge_type, measure, result, conf_begin, conf_end = \
                match.groups()
            measure = {
                'Average_R': 'recall',
                'Average_P': 'precision',
                'Average_F': 'f_score'
                }[measure]
            rouge_type = rouge_type.lower().replace("-", '_')
            key = "{}_{}".format(rouge_type, measure)
            results[key] = float(result)
            results["{}_cb".format(key)] = float(conf_begin)
            results["{}_ce".format(key)] = float(conf_end)
    return results


def get_rouge_v2(hyp_pathlist, ref_pathlist, length_limit,remove_stopwords,stemmer,lcs=False,):
    path_data = []
    uttnames = []
    for i in range(len(hyp_pathlist)):
        path_data.append([hyp_pathlist[i], [ref_pathlist[i]]])
        uttnames.append(os.path.splitext(hyp_pathlist[i])[0].split('/')[-1])

    config_text = rouge_papier_v2.util.make_simple_config_text(path_data)
    config_path = './config'
    of = open(config_path,'w')
    of.write(config_text)
    of.close()
    uttnames.append('Average')
    output = rouge_papier_v2.compute_rouge(
        config_path, max_ngram=2, lcs=lcs, 
        remove_stopwords=remove_stopwords,stemmer=stemmer,set_length = False, length=length_limit)
    c = conf.to_dict("records")
    # results = output_to_dict(output)

    # df['data_ids'] = pd.Series(np.array(uttnames),index =df.index)
    # avg = df.iloc[-1:].to_dict("records")[0]
    if lcs:
    # print(c)
        print("Rouge-1 r score: %f, Rouge-1 p score: %f, Rouge-1 f-score: %f, 95-conf(%f-%f)"%(\
                avg['rouge-1-r'],avg['rouge-1-p'],avg['rouge-1-f'],c[0]['lower_conf_f'],c[0]['upper_conf_f']))
        print("Rouge-2 r score:%f, Rouge-1 p score: %f, Rouge-2 f-score:%f, 95-conf(%f-%f)"%(\
            avg['rouge-2-r'],avg['rouge-2-p'],avg['rouge-2-f'],c[1]['lower_conf_f'],c[1]['upper_conf_f']))
        print("Rouge-L r score:%f, Rouge-1 p score: %f, Rouge-L f-score:%f, 95-conf(%f-%f)"%(\
            avg['rouge-L-r'],avg['rouge-L-p'],avg['rouge-L-f'],c[2]['lower_conf_f'],c[2]['upper_conf_f']))
    else: 
        print("Rouge-1 r score: %f, Rouge-1 p score: %f, Rouge-1 f-score: %f, 95-conf(%f-%f)"%(\
            avg['rouge-1-r'],avg['rouge-1-p'],avg['rouge-1-f'],c[0]['lower_conf_f'],c[0]['upper_conf_f']))
        print("Rouge-2 r score:%f, Rouge-1 p score: %f, Rouge-2 f-score:%f, 95-conf(%f-%f)"%(\
            avg['rouge-2-r'],avg['rouge-2-p'],avg['rouge-2-f'],c[1]['lower_conf_f'],c[1]['upper_conf_f']))

    return results['rouge_2_f_score'],df



def get_meteor(hyp_pathlist,ref_pathlist,model_type):
    all_ref =[]
    all_hyp = []
    total_num = len(hyp_pathlist)
    for i in range(total_num):
        of = open(ref_pathlist[i],'r')
        c = of.readlines()
        c = [i.strip('\n') for i in c]
        of.close()
        all_ref.append(' '.join(c))

        of = open(hyp_pathlist[i],'r')
        c = of.readlines()
        c = [i.strip('\n') for i in c]
        of.close()
        all_hyp.append(' '.join(c))

    of = open('all_ref_inorder.txt','w')
    of.write('\n'.join(all_ref))
    of.close()


    of = open('all_hyp_inorder.txt','w')
    of.write('\n'.join(all_hyp))
    of.close()

    of = open('meteor_out_%s.txt'%(model_type),'w')
    subprocess.call(['java','-Xmx2G','-jar','/ubc/cs/research/nlp/wenxiao/official_code/meteor-1.5/meteor-1.5.jar','all_hyp_inorder.txt','all_ref_inorder.txt','-norm','-f','system1'],stdout=of)
    of.close()

def rouge_matrix(data):

    evaluator = rouge.Rouge(metrics=['rouge-n','rouge-l'], max_n=2, limit_length=False,apply_avg=False,apply_best=False)
    sentences = [s['text'] for s in data['inputs']]
    n = len(sentences)
    all_scores = np.zeros((n,n))
    i = 0
    for i in range(n):
        sent = sentences[i]
        scores = evaluator.get_scores(sent,sentences[i:])
        f1 = np.array(scores['rouge-1'][0]['f'])
        f2 = np.array(scores['rouge-2'][0]['f'])
        fl = np.array(scores['rouge-l'][0]['f'])
        avg_fs = np.mean([f1,f2,fl],0)
        all_scores[i,i:] = avg_fs
    all_scores=all_scores+ all_scores.transpose() - np.eye(n)

    return all_scores

def get_rouge_python(hyp,ref):
    evaluator = rouge.Rouge(metrics=['rouge-n','rouge-l'], max_n=2, limit_length=False,apply_avg=False,apply_best=False)
    scores = evaluator.get_scores(hyp,ref)
    f1 = torch.FloatTensor([scores['rouge-1'][i]['f'][0] for i in range(len(scores['rouge-1']))])
    f2 = torch.FloatTensor([scores['rouge-2'][i]['f'][0] for i in range(len(scores['rouge-2']))])
    fl = torch.FloatTensor([scores['rouge-l'][i]['f'][0] for i in range(len(scores['rouge-l']))])
    avg_fs = (f1+f2+fl)/3
    return avg_fs

def get_rouge_single(hyp,ref):
    evaluator = rouge.Rouge(metrics=['rouge-n','rouge-l'], max_n=2, limit_length=False,apply_avg=False,apply_best=False)
    scores = evaluator.get_scores(hyp,ref)
    f1 = np.array(scores['rouge-1'][0]['f'])
    f2 = np.array(scores['rouge-2'][0]['f'])
    fl = np.array(scores['rouge-l'][0]['f'])
    avg_fs = np.mean([f1,f2,fl],0)
    return avg_fs

# return true if there is no same trigram, false otherwise.
def check_trigram(current_trigrams, units, edu=True):
    if edu:
        new_trigrams = set()
        for unit in units:
            new_trigrams.union(set(ngrams(unit,3)))
    else:
        new_trigrams = set(ngrams(units,3))
    return len(current_trigrams.intersection(new_trigrams))==0, current_trigrams.union(new_trigrams)

def get_redundancy_scores(all_file):
    sum_unigram_ratio = 0
    sum_bigram_ratio = 0
    sum_trigram_ratio = 0
    all_unigram_ratio = []
    all_bigram_ratio = []
    all_trigram_ratio = []

    sum_redundancy = 0
    stop_words = set(stopwords.words('english'))
    count = CountVectorizer()
    all_redundancy = []

    number_file = len(all_file)

    for i in range(number_file):
        of = open(all_file[i],'r')
        lines = of.readlines()
        all_txt=[]
        for line in lines:
            all_txt.extend(word_tokenize(line.strip()))
        if len(all_txt)<=5:
            print(all_file[i])
            continue

        # uniq n-gram ratio
        all_unigram = list(ngrams(all_txt,1))
        uniq_unigram = set(all_unigram)
        unigram_ratio = len(uniq_unigram)/len(all_unigram)
        sum_unigram_ratio+=unigram_ratio

        all_bigram = list(ngrams(all_txt,2))
        uniq_bigram = set(all_bigram)
        bigram_ratio = len(uniq_bigram)/len(all_bigram)
        sum_bigram_ratio+=bigram_ratio

        all_trigram = list(ngrams(all_txt,3))
        uniq_trigram = set(all_trigram)
        trigram_ratio = len(uniq_trigram)/len(all_trigram)
        sum_trigram_ratio+=trigram_ratio

        all_unigram_ratio.append(unigram_ratio)
        all_bigram_ratio.append(bigram_ratio)
        all_trigram_ratio.append(trigram_ratio)



        # NID score
        num_word = len(all_txt)
        all_txt = [w for w in all_txt if not w in stop_words]
        all_txt = [' '.join(all_txt)]

        x = count.fit_transform(all_txt)
        bow = x.toarray()[0]
        # max_possible_entropy = entropy(np.ones(bow.shape))
        # num_word = sum(bow)
        # print(num_word)
        max_possible_entropy = np.log(num_word)
        e = entropy(bow)
        redundancy = (1-e/max_possible_entropy)
        sum_redundancy+= redundancy
        all_redundancy.append(redundancy)

    print('Number of documents: %d, average unique unigram ratio is %f, average unique bigram ratio is %f, average unique trigram ratio is %f, NID score is %f.'
            %(number_file,sum_unigram_ratio/number_file,sum_bigram_ratio/number_file,sum_trigram_ratio/number_file, sum_redundancy/number_file))
    return all_unigram_ratio,all_bigram_ratio,all_trigram_ratio,all_redundancy


if __name__ == '__main__':
    # oracle_path = '/scratch/wenxiao/pubmed/oracle/test/'
    # abstract_path = '/scratch/wenxiao/pubmed/human-abstracts/test/'
    # lead_path = '/scratch/wenxiao/pubmed/lead/test/'
    oracle_path = '/ubc/cs/research/nlp/wenxiao/official_code/test_hyp/oracle-bigpatent_a/'
    lead_path = '/ubc/cs/research/nlp/wenxiao/official_code/test_hyp/lead-bigpatent_a/'
    abstract_path = '/scratch/wenxiao/bigpatent/bigPatentData_splitted/a/human-abstracts/test/'

    d = Path(oracle_path)
    uttnames = [str(path.stem) for path in d.glob("*.txt")]
    lead_pathlist = []
    oracle_pathlist = []
    ref_pathlist = []
    for n in uttnames:
        lead_pathlist.append(lead_path+n+'.txt')
        oracle_pathlist.append(oracle_path+n+'.txt')
        ref_pathlist.append(abstract_path+n+'.txt')

    get_meteor(oracle_pathlist,ref_pathlist,'oracle')
    get_meteor(lead_pathlist,ref_pathlist,'lead')

