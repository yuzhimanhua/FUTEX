import torch
import json
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='MAGCS')
parser.add_argument('--model', default='specter')
parser.add_argument('--full_text', type=int, default=1)
args = parser.parse_args()
dataset = args.dataset

device = torch.device(0)

bert_model = f'./{args.model}/'
tokenizer = BertTokenizer.from_pretrained(bert_model)
model = BertModel.from_pretrained(bert_model, output_hidden_states=True).to(device)
model.eval()

with open(f'{dataset}/{dataset}_paper.json') as fin, open(f'{dataset}/{dataset}_paper_emb.txt', 'w') as fout:
	for line in tqdm(fin):
		data = json.loads(line)
		
		# calculate title and abstract
		text = (data['title'] + ' ' + data['abstract']).strip()
		input_ids = torch.tensor(tokenizer.encode(text, max_length=512, truncation=True)).unsqueeze(0).to(device)
		outputs = model(input_ids)
		hidden_states = outputs[2][-1][0]
		abs_emb = hidden_states[0].cpu().detach().numpy()

		# calculate full text
		if args.full_text == 1:
			sec2emb = {}
			sec2cnt = {}
			for x in data['paragraphs']:
				section = x['section']
				if section not in sec2emb:
					sec2emb[section] = np.zeros(768)
					sec2cnt[section] = 0.0
				
				text = x['text'].lower()
				input_ids = torch.tensor(tokenizer.encode(text, max_length=512, truncation=True)).unsqueeze(0).to(device)
				outputs = model(input_ids)
				hidden_states = outputs[2][-1][0]
				emb = hidden_states[0].cpu().detach().numpy()
				
				sec2emb[section] += emb
				sec2cnt[section] += 1
			
			body_emb = np.zeros(768)
			for section in sec2emb:
				body_emb += sec2emb[section] / sec2cnt[section]
			if len(sec2emb) > 0:
				body_emb /= len(sec2emb)
		else:
			body_emb = abs_emb

		full_emb = (abs_emb + body_emb)/2
		out = [str(round(x, 5)) for x in full_emb]
		fout.write(' '.join(out)+'\n')
		