import json
import os
from collections import defaultdict
import argparse
import math
import re
from tqdm import tqdm

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='MAG')
parser.add_argument('--model', default='specter')
parser.add_argument('--label_only', type=int, default=1)
parser.add_argument('--topN', type=int, default=5)
args = parser.parse_args()
dataset = args.dataset
topN = args.topN

if not os.path.exists(f'./Sandbox/Data/{dataset}/'):
	os.makedirs(f'./Sandbox/Data/{dataset}/')

def process_text(text):
	text = text.lower()
	text = re.sub(r'[^a-z]', ' ', text)
	words = text.split()
	return words

train_cnt = 0
with open(f'../{dataset}/{dataset}_papers.json') as fin:
	for line in tqdm(fin):
		train_cnt += 1

if args.label_only == 0:
	word2cnt = defaultdict(int)
	with open(f'../{dataset}/{dataset}_papers.json') as fin:
		for line in tqdm(fin):
			data = json.loads(line)
			text = data['title']
			for x in data['abstract']:
				text += ' ' + x['text']
			for x in data['body_text']:
				text += ' ' + x['text']	
			words = set(process_text(text))
			for word in words:
				word2cnt[word] += 1
	word2idx = {}
	word2idf = {}
	for word in word2cnt:
		if word2cnt[word] >= 5:
			word2idx[word] = len(word2idx)
			word2idf[word] = math.log(train_cnt/word2cnt[word])
	print(len(word2idx))

label2idx = {}
with open(f'../{dataset}/{dataset}_id2label.json') as fin:
	for line in tqdm(fin):
		data = json.loads(line)
		label = data['label']
		label2idx[label] = len(label2idx)
with open(f'./Sandbox/Data/{dataset}/label2id.txt', 'w') as fout:
	for label in label2idx:
		fout.write(label+'\t'+str(label2idx[label])+'\n')

if args.label_only == 0:
	with open(f'../{dataset}/{dataset}_papers.json') as fin1, \
		 open(f'../{dataset}/{dataset}_predictions_{args.model}.json') as fin2, \
		 open(f'./Sandbox/Data/{dataset}/trn_X_Xf.txt', 'w') as fou1, \
		 open(f'./Sandbox/Data/{dataset}/trn_X_Y.txt', 'w') as fou2, \
		 open(f'./Sandbox/Data/{dataset}/tst_X_Xf.txt', 'w') as fou3, \
		 open(f'./Sandbox/Data/{dataset}/tst_X_Y.txt', 'w') as fou4:
		fou1.write(str(train_cnt)+' '+str(len(word2idx))+'\n')
		fou2.write(str(train_cnt)+' '+str(len(label2idx))+'\n')
		fou3.write(str(train_cnt)+' '+str(len(word2idx))+'\n')
		fou4.write(str(train_cnt)+' '+str(len(label2idx))+'\n')
		for line1, line2 in tqdm(zip(fin1, fin2)):
			data1 = json.loads(line1)
			text = data1['title']
			for x in data1['abstract']:
				text += ' ' + x['text']
			for x in data1['body_text']:
				text += ' ' + x['text']	
			words = process_text(text)

			bow = defaultdict(float)
			for word in words:
				if word in word2idx:
					bow[word2idx[word]] += word2idf[word]
			bow_str = []
			for word in bow:
				bow_str.append(str(word)+':'+str(bow[word]))
			fou1.write(' '.join(bow_str)+'\n')
			fou3.write(' '.join(bow_str)+'\n')

			label_str = []
			for label in data1['label']:
				label_str.append(str(label2idx[label])+':1')		
			fou4.write(' '.join(label_str)+'\n')

			data2 = json.loads(line2)
			predictions = [x[0] for x in data2['predictions'][:topN]]
			label_str = []
			for label in predictions:
				label_str.append(str(label2idx[label])+':1')		
			fou2.write(' '.join(label_str)+'\n')
else:
	with open(f'../{dataset}/{dataset}_predictions_{args.model}.json') as fin2, \
		 open(f'./Sandbox/Data/{dataset}/trn_X_Y.txt', 'w') as fou2:
		fou2.write(str(train_cnt)+' '+str(len(label2idx))+'\n')
		for line2 in tqdm(fin2):
			data2 = json.loads(line2)
			predictions = [x[0] for x in data2['predictions'][:topN]]
			label_str = []
			for label in predictions:
				label_str.append(str(label2idx[label])+':1')		
			fou2.write(' '.join(label_str)+'\n')
