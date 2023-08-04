import json
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--output_dir', required=True, type=str)
parser.add_argument('--architecture', required=True, type=str)
args = parser.parse_args()
dataset = args.dataset

pred = []
with open(os.path.join(args.output_dir, f'prediction_{args.architecture}.txt')) as fin:
	for line in fin:
		data = float(line.strip())
		pred.append(data)

i = 0
with open(f'{dataset}/{dataset}_candidates.json') as fin, \
     open(os.path.join(args.output_dir, f'{dataset}_predictions_fute_1.json'), 'w') as fout:
	for line in tqdm(fin):
		data = json.loads(line)
		out = {}
		out['mag_id'] = data['paper']

		labels = data['predicted_label']
		l = len(labels)
		sim = {}
		for label, score in zip(labels, pred[i:i+l]):
			sim[label] = score
		sim_sorted = sorted(sim.items(), key=lambda x:x[1], reverse=True)
		out['predictions'] = sim_sorted
		fout.write(json.dumps(out)+'\n')

		i += l
