import json
from tqdm import tqdm
from collections import defaultdict
import math
import argparse

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='MAG')
parser.add_argument('--model', default='specter')
parser.add_argument('--mnli', type=int, default=1)
args = parser.parse_args()
dataset = args.dataset
model_name = args.model.split('_')[0].split('-')[0]

p1 = p3 = p5 = cnt = 0
n3 = n5 = 0
if args.mnli == 0:
	with open(f'{dataset}/{dataset}_papers.json') as fin0, \
		 open(f'{dataset}/{dataset}_predictions_{model_name}.json') as fin1, \
		 open(f'{dataset}/{dataset}_predictions_{model_name}_full.json') as fin2, \
		 open(f'{dataset}/{dataset}_predictions_{model_name}_ensemble.json', 'w') as fout:
		for line0, line1, line2 in tqdm(zip(fin0, fin1, fin2)):
			data0 = json.loads(line0)
			data1 = json.loads(line1)
			data2 = json.loads(line2)
			mrr = defaultdict(float)
			for rank, x in enumerate(data1['predictions']):
				mrr[x[0]] += 1/(rank+1)
			for rank, x in enumerate(data2['predictions']):
				mrr[x[0]] += 1/(rank+1)

			score_sorted = sorted(mrr.items(), key=lambda x: x[1], reverse=True)
			top5 = [k for k, v in score_sorted[:5]]

			out = {}
			out['mag_id'] = data1['mag_id']
			out['predictions'] = score_sorted
			fout.write(json.dumps(out)+'\n')

			prec1 = prec3 = prec5 = 0
			dcg3 = dcg5 = idcg3 = idcg5 = 0
			for rank, label in enumerate(top5):
				if label in data0['label']:
					if rank < 1:
						prec1 += 1
					if rank < 3:
						prec3 += 1
						dcg3 += 1/math.log2(rank+2)
					if rank < 5:
						prec5 += 1
						dcg5 += 1/math.log2(rank+2)

			for rank in range(min(3, len(data0['label']))):
				idcg3 += 1/math.log2(rank+2)
			for rank in range(min(5, len(data0['label']))):
				idcg5 += 1/math.log2(rank+2)

			p1, p3, p5 = p1+prec1, p3+prec3/3, p5+prec5/5
			n3, n5 = n3+dcg3/idcg3, n5+dcg5/idcg5
			cnt += 1
elif args.mnli == 1:
	with open(f'{dataset}/{dataset}_papers.json') as fin0, \
		 open(f'{dataset}/{dataset}_predictions_{model_name}.json') as fin1, \
		 open(f'{dataset}/{dataset}_predictions_{model_name}_full.json') as fin2, \
		 open(f'{dataset}/{dataset}_predictions_roberta_mnli.json') as fin3, \
		 open(f'{dataset}/{dataset}_predictions_{model_name}_ensemble.json', 'w') as fout:
		for line0, line1, line2, line3 in tqdm(zip(fin0, fin1, fin2, fin3)):
			data0 = json.loads(line0)
			data1 = json.loads(line1)
			data2 = json.loads(line2)
			data3 = json.loads(line3)
			mrr = defaultdict(float)
			for rank, x in enumerate(data1['predictions']):
				mrr[x[0]] += 1/(rank+1)
			for rank, x in enumerate(data2['predictions']):
				mrr[x[0]] += 1/(rank+1)
			for rank, x in enumerate(data3['predictions']):
				mrr[x[0]] += 1/(rank+1)

			score_sorted = sorted(mrr.items(), key=lambda x: x[1], reverse=True)
			top5 = [k for k, v in score_sorted[:5]]

			out = {}
			out['mag_id'] = data1['mag_id']
			out['predictions'] = score_sorted
			fout.write(json.dumps(out)+'\n')

			prec1 = prec3 = prec5 = 0
			dcg3 = dcg5 = idcg3 = idcg5 = 0
			for rank, label in enumerate(top5):
				if label in data0['label']:
					if rank < 1:
						prec1 += 1
					if rank < 3:
						prec3 += 1
						dcg3 += 1/math.log2(rank+2)
					if rank < 5:
						prec5 += 1
						dcg5 += 1/math.log2(rank+2)

			for rank in range(min(3, len(data0['label']))):
				idcg3 += 1/math.log2(rank+2)
			for rank in range(min(5, len(data0['label']))):
				idcg5 += 1/math.log2(rank+2)

			p1, p3, p5 = p1+prec1, p3+prec3/3, p5+prec5/5
			n3, n5 = n3+dcg3/idcg3, n5+dcg5/idcg5
			cnt += 1

p1, p3, p5, n3, n5 = p1/cnt, p3/cnt, p5/cnt, n3/cnt, n5/cnt
print(p1, p3, p5, n3, n5)
with open('scores.txt', 'a') as fout:
	fout.write('{:.4f}'.format(p1)+'\t'+'{:.4f}'.format(p3)+'\t'+'{:.4f}'.format(p5)+'\t'+ \
			   '{:.4f}'.format(n3)+'\t'+'{:.4f}'.format(n5)+'\n')