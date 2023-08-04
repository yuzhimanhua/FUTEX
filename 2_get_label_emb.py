import torch
import json
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='MAG')
parser.add_argument('--model', default='specter')
args = parser.parse_args()
dataset = args.dataset
model_name = args.model.split('_')[0].split('-')[0]

device = torch.device(5)

bert_model = f'/shared/data2/yuz9/BERT_models/{args.model}/'
tokenizer = BertTokenizer.from_pretrained(bert_model)
model = BertModel.from_pretrained(bert_model, output_hidden_states=True).to(device)
model.eval()

with open(f'{dataset}/{dataset}_id2label.json') as fin, open(f'{dataset}/{dataset}_label_emb_{model_name}.txt', 'w') as fout:
	for line in tqdm(fin):
		data = json.loads(line)
		text = data['combined_text']

		input_ids = torch.tensor(tokenizer.encode(text, max_length=512, truncation=True)).unsqueeze(0).to(device)
		outputs = model(input_ids)
		hidden_states = outputs[2][-1][0]
		# emb = torch.mean(hidden_states, dim=0).cpu()
		emb = hidden_states[0].cpu()

		out = [str(round(x.item(), 5)) for x in emb]
		fout.write(' '.join(out)+'\n')