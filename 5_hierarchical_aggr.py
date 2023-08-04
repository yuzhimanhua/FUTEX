import torch
import json
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='MAG')
parser.add_argument('--model', default='specter')
args = parser.parse_args()
dataset = args.dataset
model_name = args.model.split('_')[0].split('-')[0]

device = torch.device(4)

bert_model = f'/shared/data2/yuz9/BERT_models/{args.model}/'
tokenizer = BertTokenizer.from_pretrained(bert_model)
model = BertModel.from_pretrained(bert_model, output_hidden_states=True).to(device)
model.eval()

def process_section(text, sections):
	idx = 0
	while True:
		section = f'{text}XXX{idx}'
		if section not in sections or section == sections[-1]:
			return section
		idx += 1

min_length = 10
with open(f'{dataset}/{dataset}_papers.json') as fin, open(f'{dataset}/{dataset}_paper_fullemb_{model_name}.txt', 'w') as fout:
	for line in tqdm(fin):
		data = json.loads(line)
		
		# calculate abstract
		sections = []
		sec2emb = {}
		sec2cnt = {}
		for x in data['abstract']:
			words = x['text'].split()
			if len(words) <= min_length:
				continue

			section = process_section(x['section'].strip().lower(), sections)
			if len(sections) == 0 or section != sections[-1]:
				sections.append(section)
				sec2emb[section] = np.zeros(768)
				sec2cnt[section] = 0.0
			
			text = x['text'].lower()
			text = ' '.join(text.split())
			input_ids = torch.tensor(tokenizer.encode(text, max_length=512, truncation=True)).unsqueeze(0).to(device)
			outputs = model(input_ids)
			hidden_states = outputs[2][-1][0]
			emb = hidden_states[0].cpu().detach().numpy()
			
			sec2emb[section] += emb
			sec2cnt[section] += 1
		
		abs_emb = np.zeros(768)
		for section in sections:
			abs_emb += sec2emb[section] / sec2cnt[section]
		if len(sec2cnt) > 0:
			abs_emb /= len(sec2cnt)
		
		# calculate body text
		sections = []
		sec2emb = {}
		sec2cnt = {}
		for x in data['body_text']:
			words = x['text'].split()
			if len(words) <= min_length:
				continue

			section = process_section(x['section'].strip().lower(), sections)
			if len(sections) == 0 or section != sections[-1]:
				sections.append(section)
				sec2emb[section] = np.zeros(768)
				sec2cnt[section] = 0.0
			
			text = x['text'].lower()
			text = ' '.join(text.split())
			input_ids = torch.tensor(tokenizer.encode(text, max_length=512, truncation=True)).unsqueeze(0).to(device)
			outputs = model(input_ids)
			hidden_states = outputs[2][-1][0]
			emb = hidden_states[0].cpu().detach().numpy()
			
			sec2emb[section] += emb
			sec2cnt[section] += 1
		
		body_emb = np.zeros(768)
		for section in sections:
			body_emb += sec2emb[section] / sec2cnt[section]
		if len(sec2cnt) > 0:
			body_emb /= len(sec2cnt)

		full_emb = (abs_emb + body_emb)/2
		out = [str(round(x, 5)) for x in full_emb]
		fout.write(' '.join(out)+'\n')
		