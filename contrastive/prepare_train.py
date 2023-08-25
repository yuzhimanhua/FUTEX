import json
import argparse
import os
from collections import defaultdict
import random
from tqdm import tqdm

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='MAGCS', type=str)
args = parser.parse_args()
dataset = args.dataset

if not os.path.exists(f'{dataset}_input/'):
	os.mkdir(f'{dataset}_input/')

paper2text = {}
papers = []
with open(f'../{dataset}/{dataset}_paper.json') as fin:
	for line in tqdm(fin):
		data = json.loads(line)
		paper = data['paper']
		texts = [x['text'] for x in data['paragraphs']]
		if len(texts) == 0:
			continue
		if random.random() < 0.5:
			text = texts[0]
		else:
			text = random.choice(texts)
		paper2text[paper] = text
		papers.append(paper)

ref2paper = defaultdict(set)
paper2ref = {}
with open(f'../{dataset}/{dataset}_paper.json') as fin:
	for line in tqdm(fin):
		data = json.loads(line)
		paper = data['paper']
		if paper not in paper2text:
			continue
		refs = data['reference']
		for ref in refs:
			ref2paper[ref].add(paper)
		paper2ref[paper] = set(refs)

with open(f'{dataset}_input/dataset.txt', 'w') as fout:
	for paper in tqdm(paper2ref):
		# sample positive
		refs = paper2ref[paper]
		dps = []
		for ref in refs:
			candidates = list(ref2paper[ref])
			if len(candidates) > 1:
				while True:
					dp = random.choice(candidates)
					if dp != paper:
						dps.append(dp)
						break
		if len(dps) == 0:
			continue
		dp = random.choice(dps)
		# sample negative
		while True:
			dn = random.choice(papers)
			if dn != paper and dn != dp:
				break
		fout.write(f'1\t{paper2text[paper]}\t{paper2text[dp]}\n')
		fout.write(f'0\t{paper2text[paper]}\t{paper2text[dn]}\n')
		fout.flush()
