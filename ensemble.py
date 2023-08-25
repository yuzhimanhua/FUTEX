import json
from tqdm import tqdm
from collections import defaultdict
import math
import argparse

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='MAGCS')
parser.add_argument('--mnli', type=int, default=1)
args = parser.parse_args()
dataset = args.dataset

if args.mnli == 0:
	with open(f'{dataset}/{dataset}_paper.json') as fin0, \
		 open(f'{dataset}/{dataset}_predictions_network.json') as fin1, \
		 open(f'{dataset}/{dataset}_predictions_hierarchy.json') as fin2, \
		 open(f'{dataset}/{dataset}_predictions_ensemble.json', 'w') as fout:
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
			out['paper'] = data1['paper']
			out['predictions'] = score_sorted
			fout.write(json.dumps(out)+'\n')

elif args.mnli == 1:
	with open(f'{dataset}/{dataset}_paper.json') as fin0, \
		 open(f'{dataset}/{dataset}_predictions_network.json') as fin1, \
		 open(f'{dataset}/{dataset}_predictions_hierarchy.json') as fin2, \
		 open(f'{dataset}/{dataset}_predictions_mnli.json') as fin3, \
		 open(f'{dataset}/{dataset}_predictions_ensemble.json', 'w') as fout:
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
			out['paper'] = data1['paper']
			out['predictions'] = score_sorted
			fout.write(json.dumps(out)+'\n')
