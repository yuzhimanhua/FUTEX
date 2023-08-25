import json
import argparse

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='MAGCS')
parser.add_argument('--model', default='ensemble')
args = parser.parse_args()
dataset = args.dataset
model_name = args.model

label2id = {}
id2label = {}
with open(f'./Sandbox/Data/{dataset}/label2id.txt') as fin:
	for line in fin:
		data = line.strip().split()
		label = data[0]
		idx = data[1]
		label2id[label] = idx
		id2label[idx] = label

papers = []
preds_prev = []
with open(f'../{dataset}/{dataset}_predictions_{model_name}.json') as fin:
	for line in fin:
		data = json.loads(line)
		papers.append(data['paper'])
		pred_prev = [label2id[x[0]] for x in data['predictions']]
		preds_prev.append(pred_prev)

preds = []
with open(f'./Sandbox/Results/{dataset}/score_mat.txt') as fin, \
	 open(f'../{dataset}/{dataset}_predictions_futex.json', 'w') as fout:
	for idx, line in enumerate(fin):
		if idx == 0:
			continue
		data = line.strip().split()
		scores = {}
		for y in data:
			y_tup = y.split(':')
			scores[y_tup[0]] = float(y_tup[1])
		scores_sorted = sorted(scores.items(), key=lambda x:x[1], reverse=True)

		pred_prev = preds_prev[idx-1]
		pred = pred_prev + [y[0] for y in scores_sorted if y[0] not in pred_prev]

		out = {}
		out['paper'] = papers[idx-1]
		out['predictions'] = [[id2label[x], 1] for x in pred if x in id2label]
		fout.write(json.dumps(out)+'\n')
