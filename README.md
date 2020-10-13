# Redundancy Reduction of Extractive Summarization
This is the official code for the paper ['Systematically Exploring Redundancy Reduction in Summarizing Long Documents']() (AACL 2020)

In this paper, we systematically explored ways for redundancy reduction for extractive summarization on long documents.

## Installation
Make sure you have `python 3` and `pytorch` installed.

First need to install the tool [rouge_papier_v2](https://github.com/Wendy-Xiao/Extsumm_local_global_context/tree/master/rouge_papier_v2).  
```
python setup.py install.
```
(This is a modified version from https://github.com/kedz/rouge_papier)

Other dependencies needed: `numpy`, `pandas`, `nltk->\[word_tokenizer,stopwords\]`

## Data and Trained Models
We do the experiments on two scientific paper datasets, Pubmed and arXiv, and you can find the preprocessed data [here]().

The trained models that we showed in the paper is [here]().

## Evaluation

If you download the trained model and data, and put them in folder `./pretrained_models` and `./scientific_paper_dataset/` , respectively, then you can use the following commands to evaluate the trained models.

```
python test.py --modelpath ./pretrained_models --datapath ./scientific_paper_dataset/ --dataset pubmed 
```
For different models, you need to add different arguments seeing below:

1. The original model (ExtSumLG) `--model ac`

2. ExtSumLG + SR Decoder `--model ac_sr`

3. ExtSumLG + NeuSum Decoder `--model ac_neusum `

4. ExtSumLG + RdLoss (beta=0.3) `--model ac --beta 0.3`

5. ExtSumLG + Trigram Block `--model ac --use_trigram_block`

6. ExtSumLG + MMR-Select ($`\lambda =0.6`$) `--model ac --use_mmr --lambd 0.6`

7. ExtSumLG + MMR-Select+ (lambda=0.6, $`\gamma =0.99`$) `--model ac --use_rl --use_mmr --lambd 0.6 --gamma 0.99`



## Train Your Own
If you want to train your own model, you can run the following commands. 

```
python main.py --modelpath ./pretrained_models --datapath ./scientific_paper_dataset/ --dataset pubmed 
```
For different models, you need to add different arguments seeing below:

1. The original model (ExtSumLG) `--model ac`

2. ExtSumLG + SR Decoder `--model ac_sr`

3. ExtSumLG + NeuSum Decoder `--model ac_neusum `

4. ExtSumLG + RdLoss (beta=0.3) `--model ac --beta 0.3`

5. ExtSumLG + MMR-Select+ (lambda=0.6, gamma=0.99) `--model ac --use_rl --use_mmr --lambd 0.6 --gamma 0.99`

You are free to play with different hyper parameters, which can be found in `main.py`.

## Cite this paper
Coming Soon.


