from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import json
import argparse

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='MAGCS')
args = parser.parse_args()
dataset = args.dataset

# huggingface==4.0.0
device = 0
classifier = pipeline(task='sentiment-analysis', model='roberta-large-mnli', device=device, return_all_scores=True)
tokenizer = AutoTokenizer.from_pretrained('roberta-large-mnli')

label2name = {}
with open(f'{dataset}/{dataset}_label.json') as fin:
	for line in tqdm(fin):
		data = json.loads(line)
		label = data['label']
		name = data['name'][0]
		label2name[label] = name

max_paper_len = 480
max_label_len = 20
with open(f'{dataset}/{dataset}_paper.json') as fin1, \
	 open(f'{dataset}/{dataset}_candidates.json') as fin2, \
	 open(f'{dataset}/{dataset}_predictions_mnli.json', 'w') as fout:
	for line1, line2 in tqdm(zip(fin1, fin2)):
		data1 = json.loads(line1)
		data2 = json.loads(line2)
		assert data1['paper'] == data2['paper']

		text = (data1['title'] + ' ' + data1['abstract']).strip()
		tokens = tokenizer(text, truncation=True, max_length=max_paper_len)
		text = tokenizer.decode(tokens["input_ids"][1:-1])

		score = {}
		for label in data2['matched_label']:
			name = label2name[label]
			name = ' '.join(name.split())
			tokens = tokenizer(name, truncation=True, max_length=max_label_len)
			name = tokenizer.decode(tokens["input_ids"][1:-1])
			input = f'{text} </s></s> this document is about {name}.'
			output = classifier(input)
			score[label] = output[0][-1]['score']
		score_sorted = sorted(score.items(), key=lambda x:x[1], reverse=True)
		top5 = [k for k, v in score_sorted[:5]]

		out = {}
		out['paper'] = data1['paper']
		out['predictions'] = score_sorted
		fout.write(json.dumps(out)+'\n')
