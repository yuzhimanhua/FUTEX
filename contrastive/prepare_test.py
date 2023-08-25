import json
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='MAGCS', type=str)
args = parser.parse_args()
dataset = args.dataset

if not os.path.exists(f'{dataset}_input/'):
	os.mkdir(f'{dataset}_input/')

paper2text = {}
with open(f'../{dataset}/{dataset}_paper.json') as fin:
	for line in tqdm(fin):
		data = json.loads(line)
		paper = data['paper']
		text = (data['title'] + ' ' + data['abstract']).strip()
		paper2text[paper] = text

label2text = {}
with open(f'../{dataset}/{dataset}_label.json') as fin:
	for line in tqdm(fin):
		data = json.loads(line)
		label = data['label']
		text = data['combined_text']
		label2text[label] = text

with open(f'../{dataset}/{dataset}_candidates.json') as fin, open(f'{dataset}_input/test.txt', 'w') as fout:
	for line in tqdm(fin):
		data = json.loads(line)
		paper_text = paper2text[data['paper']]	
		labels = data['matched_label']
		for label in labels:
			label_text = label2text[label]
			fout.write(f'1\t{paper_text}\t{label_text}\n')
			fout.flush()
