from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import json
import argparse
import math

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='MAG')
args = parser.parse_args()
dataset = args.dataset

# huggingface==4.0.0
device = 4
classifier = pipeline(task='sentiment-analysis', model='facebook/bart-large-mnli', device=device, return_all_scores=True)
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')

label2name = {}
with open(f'{dataset}/{dataset}_id2label.json') as fin:
	for line in tqdm(fin):
		data = json.loads(line)
		label = data['label']
		name = data['name'][0]
		label2name[label] = name

max_paper_len = 480
max_label_len = 20
p1 = p3 = p5 = cnt = 0
n3 = n5 = 0
with open(f'{dataset}/{dataset}_papers.json') as fin1, open(f'{dataset}/{dataset}_matched.json') as fin2, \
	 open(f'{dataset}/{dataset}_predictions_bart_mnli.json', 'w') as fout:
	for line1, line2 in tqdm(zip(fin1, fin2)):
		data1 = json.loads(line1)
		data2 = json.loads(line2)
		assert data1['mag_id'] == data2['mag_id']

		abstract = data1['title']
		for x in data1['abstract']:
			abstract += ' ' + x['text']
		text = abstract.lower()
		text = ' '.join(text.split())
		tokens = tokenizer(text, truncation=True, max_length=max_paper_len)
		text = tokenizer.decode(tokens["input_ids"][1:-1])

		score = {}
		for label in data2['candidates']:
			name = label2name[label]
			name = ' '.join(name.split())
			tokens = tokenizer(name, truncation=True, max_length=max_label_len)
			name = tokenizer.decode(tokens["input_ids"][1:-1])
			input = f'{text} </s></s> this document is about {name}.'
			# hypothesis = f'this document is about {name}.'
			# tokens = tokenizer(text, hypothesis, truncation='only_first')
			# input = tokenizer.decode(tokens["input_ids"][1:-1])
			output = classifier(input)
			score[label] = output[0][-1]['score']
		score_sorted = sorted(score.items(), key=lambda x:x[1], reverse=True)
		top5 = [k for k, v in score_sorted[:5]]

		out = {}
		out['mag_id'] = data1['mag_id']
		out['predictions'] = score_sorted
		fout.write(json.dumps(out)+'\n')
		fout.flush()
		
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
