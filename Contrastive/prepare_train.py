import json
import argparse
import os
from collections import defaultdict
import random
from tqdm import tqdm

# P->P<-P
def one_intermediate_node(dataset, doc2text, docs, metadata):
	meta2doc = defaultdict(set)
	doc2meta = {}
	with open(f'{dataset}/{dataset}_train.json') as fin:
		for idx, line in enumerate(tqdm(fin)):
			data = json.loads(line)
			doc = data['paper']
			if doc not in doc2text:
				continue

			metas = data[metadata]
			if not isinstance(metas, list):
				metas = [metas]
			for meta in metas:
				meta2doc[meta].add(doc)
			doc2meta[doc] = set(metas)

	with open(f'{dataset}_input/dataset.txt', 'w') as fout:
		for idx, doc in enumerate(tqdm(doc2meta)):
			# sample positive
			metas = doc2meta[doc]
			dps = []
			for meta in metas:
				candidates = list(meta2doc[meta])
				if len(candidates) > 1:
					while True:
						dp = random.choice(candidates)
						if dp != doc:
							dps.append(dp)
							break
			if len(dps) == 0:
				continue
			dp = random.choice(dps)

			# sample negative
			while True:
				dn = random.choice(docs)
				if dn != doc and dn != dp:
					break	
					
			fout.write(f'1\t{doc2text[doc]}\t{doc2text[dp]}\n')
			fout.write(f'0\t{doc2text[doc]}\t{doc2text[dn]}\n')
			fout.flush()


parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='MAG', type=str)
args = parser.parse_args()
dataset = args.dataset

doc2text = {}
docs = []
with open(f'../Cleaned/{dataset}_cleaned.json') as fin:
	for line in tqdm(fin):
		data = json.loads(line)
		if len(data['text']) == 0:
			continue
		doc = data['mag_id']
		if random.random() < 0.5:
			text = data['text'][0]
		else:
			text = random.choice(data['text'])
		doc2text[doc] = text
		docs.append(doc)

# P->P<-P
one_intermediate_node(dataset, doc2text, docs, 'reference')