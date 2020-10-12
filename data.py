from torch.utils.data import DataLoader, Dataset,IterableDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence,pad_packed_sequence
from pathlib import Path
from torch import nn
import torch.nn.functional as F
import json
import utils
import torch
from models import *
import collections
from nltk import word_tokenize
from random import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer


class SummarizationDataset(IterableDataset):
	def __init__(self,word2index, embedding_matrix, embedding_size, inputs_dir,ref_required=False,\
				to_shuffle=True,is_test=False,\
				useNeusum=False, neusum_path=None):
		if isinstance(inputs_dir,list):
			self._input_files = inputs_dir
		else:
			inputs_dir = Path(inputs_dir)
			self._input_files = [path for path in inputs_dir.glob("*.pt")]
		self.shuffle=to_shuffle
		self._input_files = sorted(self._input_files)
		if self.shuffle:
			shuffle(self._input_files)
		self.is_test=is_test
		self._w2i = word2index
		self.ref_required=ref_required
		self.embedding_matrix = embedding_matrix
		self.useNeusum=useNeusum
		self.neusum_path=neusum_path
		

	def _loaddata(self,idx):
		file = self._input_files[idx]
		self.cur_data = torch.load(file)
		# print(file)
		if self.shuffle:
			shuffle(self.cur_data)
		if (idx==len(self._input_files)-1) and self.shuffle:
			shuffle(self._input_files)
	# def __len__(self):
	# 	return len(self._inputs)

	def preprocessing(self,data):
		out = {}

		out['id'] = data['doc_id']
		out['labels'] = data['labels']
		out['sent_embed'],out['sentence_lengths'] = self.get_sent_embedding(data['sent_txt'])
		out['num_sentences'] = out['sent_embed'].shape[0]
		# out['sent_tfidf'] = self.get_tfidf_sent(data['sent_txt'])
		tfidf_vec = TfidfVectorizer()
		out['sent_tfidf'] = torch.tensor(tfidf_vec.fit_transform([' '.join(s) for s in data['sent_txt']]).toarray(),dtype=torch.float)
		out['section_lengths'] = data['sent_per_sec']

		# If the reference is given, load the reference
		out['reference'] = None	
		out['sent_txt'] = data['sent_txt']
		if self.ref_required:
			out['reference']= '\n'.join([sent.strip() for sent in data['tgt_list_str']])
		if self.useNeusum:
			t = torch.load(self.neusum_path+'/'+out['id']+'.pt')
			out['selection'] = t['picked']
			raw_gain = t['target']

			out['score_gain'] = raw_gain-torch.min(raw_gain,dim=1)[0].unsqueeze(1).expand_as(raw_gain)/(torch.max(raw_gain,dim=1)[0].unsqueeze(1).expand_as(raw_gain)-torch.min(raw_gain,dim=1)[0].unsqueeze(1).expand_as(raw_gain))

		return out

	def get_sent_embedding(self, sent_txt):
		document_l = []
		sentence_lengths=[]
		for sent in sent_txt:
			sent_l = []
			if len(sent)!=0:
				sent_l = [self._w2i.get(word,0) for word in sent]
				# for word in sent:
					# sent_l.append(self.lookup_table.get(word,np.zeros(300,dtype='float32')))
				# sent_embed = torch.mean(torch.tensor(sent_l),dim=0)
			else:
				sent_l = [0]
			sentence_lengths.append(len(sent_l))
			sent_embed = torch.mean(self.embedding_matrix[sent_l,:],dim=0)
			document_l.append(sent_embed)
		return torch.stack(document_l,0),sentence_lengths #length * embedding


	def __iter__(self):
		# for i in range(len(self._input_files)):
		if not self.is_test:
			i=0
			while (True):
				self._loaddata(i)
				while len(self.cur_data) !=0:
					data = self.cur_data.pop()

					out = self.preprocessing(data)
					yield out 
				i = (i+1)%(len(self._input_files))

		if self.is_test:
			for i in range(len(self._input_files)):
				self._loaddata(i)
				while len(self.cur_data) !=0:
					data = self.cur_data.pop()
					if self.useNeusum:
						neusum_file = self.neusum_path+'/'+data['doc_id']+'.pt'
						if not os.path.exists(neusum_file):
							continue
					out = self.preprocessing(data)
					yield out 

class SummarizationDataLoader(DataLoader):
	def __init__(self,dataset, batch_size=1, max_length = 2000, useNeusum=False):
		super(SummarizationDataLoader, self).__init__(
			dataset, batch_size=batch_size, collate_fn =self.avgsent_batch)
		self.max_length = max_length
		self.useNeusum=useNeusum
	def avgsent_batch(self,batch):
		batch.sort(key=lambda x: x["num_sentences"], reverse=True)
		out = {}
		out['id'] = []
		doc_batch = []
		labels_batch = []
		doc_lengths = []
		out['refs'] = []
		out['sent_txt'] = []
		summary_representation = []
		out['sentence_lengths'] = []
		section_length_batch = []
		rouge_matrix = []
		sentences = []
		references = []
		tfidf = []
		neusum_selections = []
		neusum_scoregain = []
		for d in batch:
			out['id'].append(d['id'])
			# doc_l = torch.FloatTensor(d['num_sentences'],d['document'][0].size()[1])
			# for i in range(len(d['document'])):
			# 	doc_l[i,:] = torch.mean(d['document'][i],0)
			doc_l=d['sent_embed']
			doc_batch.append(doc_l)
			labels_batch.append(torch.FloatTensor(d['labels']).unsqueeze(1))
			doc_lengths.append(d['num_sentences'])
			out['sent_txt'].append(d['sent_txt'])
			out['sentence_lengths'].append(d['sentence_lengths'])
			section_length_batch.append(d['section_lengths'])
			tfidf.append(d['sent_tfidf'])

			if d['reference']!=None:
				out['refs'].append(d['reference'])
			if self.useNeusum:
				neusum_selections.append(d['selection'])
				neusum_scoregain.append(d['score_gain'])

		indicators,padded_lengths = self.build_section_indicators_and_pad(section_length_batch,doc_lengths[0])
		out['indicators'] = indicators
		out['padded_lengths'] = padded_lengths
		out['sent_tfidf'] = tfidf

		if self.useNeusum:

			out['selections'] = neusum_selections #batch*longest_oracle

			out['scores_gain'] = neusum_scoregain # batch*step*doc_length

		padded_doc_batch = pad_sequence(doc_batch,padding_value=-1)
		padded_labels_batch = pad_sequence(labels_batch,padding_value=-1)
		packed_padded_doc_batch = pack_padded_sequence(padded_doc_batch,doc_lengths)
		out['document'] = packed_padded_doc_batch
		out['labels'] = padded_labels_batch
		out['input_length'] = torch.LongTensor(doc_lengths)

		return out

	def build_section_indicators_and_pad(self,section_length_batch,max_seq_length):
		max_section_num = max([len(i) for i in section_length_batch])
		batch_size = len(section_length_batch)
		# padded lengths
		padded_lengths = torch.zeros((batch_size,max_section_num),dtype=torch.int)
		# indicators
		indicators = torch.zeros((batch_size,max_seq_length,max_section_num))

		for i_sec in range(batch_size):
			section_lengths = torch.LongTensor(section_length_batch[i_sec])
			padded_lengths[i_sec,:section_lengths.shape[0]] = section_lengths
			end = torch.clamp(torch.cumsum(section_lengths,0),0,max_seq_length)
			begin = torch.cat((torch.LongTensor([0]),end[:-1]),0)
			for i in range(len(begin)):
				indicators[i_sec,begin[i]:end[i],i]=1

		return indicators,padded_lengths

	def wordlevel_batch(self,batch):
		batch.sort(key=lambda x: x["num_words"], reverse=True)
		out = {}
		out['id'] = []
		doc_batch = []
		labels_batch = []
		doc_lengths = []
		doc_lengths_word=[]
		out['refs'] = []
		out['filenames'] = []
		
		valid_sentence_length = []
		section_length_batch = []
		sentence_length_batch = []
		for d in batch:
			out['id'].append(d['id'])
			doc_l=torch.cat(d['document'],0)
			doc_batch.append(doc_l)
			
			doc_lengths.append(d['num_sentences'])
			doc_lengths_word.append(d['num_words'])
			out['filenames'].append(d['filename'])
			if d['reference']!=None:
				out['refs'].append(d['reference'])
			# Word level section length
			section_length_batch.append(self.build_section_lengths_word(d['section_lengths'],d['sentence_lengths']))
			if self.max_length > 0:
				i = self.cut_labels(d['sentence_lengths'],self.max_length)
				valid_sentence_length
				labels_batch.append(torch.FloatTensor(d['labels'][:i]).unsqueeze(1))
				sentence_length_batch.append(d['sentence_lengths'][:i])
			else:
				labels_batch.append(torch.FloatTensor(d['labels']).unsqueeze(1))
				sentence_length_batch.append(d['sentence_lengths'])

		if self.max_length>0:
			max_length = self.max_length
		else:
			max_length = doc_lengths_word[0]

		sec_indicators,sec_padded_lengths = self.build_indicators_and_pad(section_length_batch,max_length)
		sent_indicators,sent_padded_lengths = self.build_indicators_and_pad(sentence_length_batch,max_length)
		out['sec_indicators'] = sec_indicators
		out['sec_padded_lengths'] = sec_padded_lengths
		out['sent_indicators'] = sent_indicators
		out['sent_padded_lengths'] = sent_padded_lengths

		padded_doc_batch = pad_sequence(doc_batch,padding_value=-1)
		padded_labels_batch = pad_sequence(labels_batch,padding_value=-1)
		packed_padded_doc_batch = pack_padded_sequence(padded_doc_batch,doc_lengths_word)
		out['document'] = packed_padded_doc_batch
		out['labels'] = padded_labels_batch
		out['input_length'] = torch.LongTensor(doc_lengths_word)
		return out

	def build_indicators_and_pad(self,length_batch,  max_seq_length):
		max_section_num = max([len(i) for i in length_batch])
		batch_size = len(length_batch)
		# padded lengths
		padded_lengths = torch.zeros((batch_size,max_section_num),dtype=torch.int)
		# indicators
		indicators = torch.zeros((batch_size,max_seq_length,max_section_num))

		for i_sec in range(batch_size):
			lengths = torch.LongTensor(length_batch[i_sec])
			if lengths.shape[0] > max_seq_length:
				padded_lengths[i_sec,:max_seq_length] = lengths[:max_seq_length]
			else:
				padded_lengths[i_sec,:lengths.shape[0]] = lengths
			end = torch.clamp(torch.cumsum(lengths,0),0,max_seq_length)
			begin = torch.cat((torch.LongTensor([0]),end[:-1]),0)
			for i in range(len(begin)):
				indicators[i_sec,begin[i]:end[i],i]=1
		return indicators,padded_lengths

	def build_section_lengths_word(self,section_length, sentence_length):
		start=0
		section_lengths_word = []
		for sec_len in section_length:
			section_lengths_word.append(sum(sentence_length[start:start+sec_len]))
		return section_lengths_word

	def cut_labels(self,sentence_length, max_seq_length):
		if sum(sentence_length) < max_seq_length:
			return None
		else:
			i=0
			while sum(sentence_length[:i])<= max_seq_length:
				i+=1
			return i
