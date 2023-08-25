import json
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='MAGCS')
args = parser.parse_args()
dataset = args.dataset

label2emb = {}
with open(f'{dataset}/{dataset}_label.json') as fin1, open(f'{dataset}/{dataset}_label_emb.txt') as fin2:
	for line1, line2 in tqdm(zip(fin1, fin2)):
		data1 = json.loads(line1)
		label = data1['label']
		data2 = line2.strip().split()
		emb = [float(x) for x in data2]
		emb = emb / np.linalg.norm(emb)
		label2emb[label] = emb

with open(f'{dataset}/{dataset}_paper.json') as fin1, \
	 open(f'{dataset}/{dataset}_candidates.json') as fin2, \
	 open(f'{dataset}/{dataset}_paper_emb.txt') as fin3, \
	 open(f'{dataset}/{dataset}_predictions_hierarchy.json', 'w') as fout:
	for line1, line2, line3 in tqdm(zip(fin1, fin2, fin3)):
		data1 = json.loads(line1)
		data2 = json.loads(line2)
		data3 = line3.strip().split()
		assert data1['paper'] == data2['paper']
		
		p_emb = [float(x) for x in data3]
		norm = np.linalg.norm(p_emb)
		if norm > 1e-9:
			p_emb = p_emb / norm
		label2score = {}
		for label in data2['matched_label']:
			l_emb = label2emb[label]
			label2score[label] = np.dot(p_emb, l_emb)
		score_sorted = sorted(label2score.items(), key=lambda x: x[1], reverse=True)
		top5 = [k for k, v in score_sorted[:5]]

		out = {}
		out['paper'] = data1['paper']
		out['predictions'] = score_sorted
		fout.write(json.dumps(out)+'\n')
