import json
from tqdm import tqdm
import numpy as np
import math
import argparse

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='MAG')
parser.add_argument('--model', default='specter')
args = parser.parse_args()
dataset = args.dataset
model_name = args.model.split('_')[0].split('-')[0]

label2emb = {}
with open(f'{dataset}/{dataset}_id2label.json') as fin1, open(f'{dataset}/{dataset}_label_emb_{model_name}.txt') as fin2:
	for line1, line2 in tqdm(zip(fin1, fin2)):
		data1 = json.loads(line1)
		label = data1['label']

		data2 = line2.strip().split()
		emb = [float(x) for x in data2]
		emb = emb / np.linalg.norm(emb)
		label2emb[label] = emb

p1 = p3 = p5 = cnt = 0
n3 = n5 = 0
with open(f'{dataset}/{dataset}_papers.json') as fin1, open(f'{dataset}/{dataset}_matched.json') as fin2, \
	 open(f'{dataset}/{dataset}_paper_emb_{model_name}.txt') as fin3, open(f'{dataset}/{dataset}_predictions_{model_name}.json', 'w') as fout:
	for line1, line2, line3 in tqdm(zip(fin1, fin2, fin3)):
		data1 = json.loads(line1)
		data2 = json.loads(line2)
		data3 = line3.strip().split()
		assert data1['mag_id'] == data2['mag_id']
		
		p_emb = [float(x) for x in data3]
		p_emb = p_emb / np.linalg.norm(p_emb)
		label2score = {}
		for label in data2['candidates']:
			l_emb = label2emb[label]
			label2score[label] = np.dot(p_emb, l_emb)
		score_sorted = sorted(label2score.items(), key=lambda x: x[1], reverse=True)
		top5 = [k for k, v in score_sorted[:5]]

		out = {}
		out['mag_id'] = data1['mag_id']
		out['predictions'] = score_sorted
		fout.write(json.dumps(out)+'\n')

		prec1 = prec3 = prec5 = 0
		dcg3 = dcg5 = idcg3 = idcg5 = 0
		for rank, label in enumerate(top5):
			if label in data1['label']:
				if rank < 1:
					prec1 += 1
				if rank < 3:
					prec3 += 1
					dcg3 += 1/math.log2(rank+2)
				if rank < 5:
					prec5 += 1
					dcg5 += 1/math.log2(rank+2)

		for rank in range(min(3, len(data1['label']))):
			idcg3 += 1/math.log2(rank+2)
		for rank in range(min(5, len(data1['label']))):
			idcg5 += 1/math.log2(rank+2)

		p1, p3, p5 = p1+prec1, p3+prec3/3, p5+prec5/5
		n3, n5 = n3+dcg3/idcg3, n5+dcg5/idcg5
		cnt += 1

p1, p3, p5, n3, n5 = p1/cnt, p3/cnt, p5/cnt, n3/cnt, n5/cnt
print(p1, p3, p5, n3, n5)
with open('scores.txt', 'a') as fout:
	fout.write('{:.4f}'.format(p1)+'\t'+'{:.4f}'.format(p3)+'\t'+'{:.4f}'.format(p5)+'\t'+ \
			   '{:.4f}'.format(n3)+'\t'+'{:.4f}'.format(n5)+'\n')
