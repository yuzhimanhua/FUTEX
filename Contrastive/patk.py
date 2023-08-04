import json
import numpy as np
from functools import partial
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Union, Optional, List, Iterable, Hashable
import os
import argparse

import warnings
warnings.filterwarnings('ignore')

TPredict = np.ndarray
TTarget = Union[Iterable[Iterable[Hashable]], csr_matrix]
TMlb = Optional[MultiLabelBinarizer]
TClass = Optional[List[Hashable]]

def get_mlb(classes: TClass = None, mlb: TMlb = None, targets: TTarget = None):
	if classes is not None:
		mlb = MultiLabelBinarizer(classes, sparse_output=True)
	if mlb is None and targets is not None:
		if isinstance(targets, csr_matrix):
			mlb = MultiLabelBinarizer(range(targets.shape[1]), sparse_output=True)
			mlb.fit(None)
		else:
			mlb = MultiLabelBinarizer(sparse_output=True)
			mlb.fit(targets)
	return mlb


def get_precision(prediction: TPredict, targets: TTarget, mlb: TMlb = None, classes: TClass = None, top=5):
	mlb = get_mlb(classes, mlb, targets)
	if not isinstance(targets, csr_matrix):
		targets = mlb.transform(targets)
	prediction = mlb.transform(prediction[:, :top])
	return prediction.multiply(targets).sum() / (top * targets.shape[0])

get_p_1 = partial(get_precision, top=1)
get_p_3 = partial(get_precision, top=3)
get_p_5 = partial(get_precision, top=5)


def get_ndcg(prediction: TPredict, targets: TTarget, mlb: TMlb = None, classes: TClass = None, top=5):
	mlb = get_mlb(classes, mlb, targets)
	log = 1.0 / np.log2(np.arange(top) + 2)
	dcg = np.zeros((targets.shape[0], 1))
	if not isinstance(targets, csr_matrix):
		targets = mlb.transform(targets)
	for i in range(top):
		p = mlb.transform(prediction[:, i: i+1])
		dcg += p.multiply(targets).sum(axis=-1) * log[i]
	return np.average(dcg / log.cumsum()[np.minimum(targets.sum(axis=-1), top) - 1])

get_n_3 = partial(get_ndcg, top=3)
get_n_5 = partial(get_ndcg, top=5)


def get_inv_propensity(train_y: csr_matrix, a=0.55, b=1.5):
	n, number = train_y.shape[0], np.asarray(train_y.sum(axis=0)).squeeze()
	c = (np.log(n) - 1) * ((b + 1) ** a)
	return 1.0 + c * (number + b) ** (-a)


def get_psp(prediction: TPredict, targets: TTarget, inv_w: np.ndarray, mlb: TMlb = None,
			classes: TClass = None, top=5):
	mlb = get_mlb(classes, mlb)
	if not isinstance(targets, csr_matrix):
		targets = mlb.transform(targets)
	prediction = mlb.transform(prediction[:, :top]).multiply(inv_w)
	num = prediction.multiply(targets).sum()
	t, den = csr_matrix(targets.multiply(inv_w)), 0
	for i in range(t.shape[0]):
		den += np.sum(np.sort(t.getrow(i).data)[-top:])
	return num / den

get_psp_1 = partial(get_psp, top=1)
get_psp_3 = partial(get_psp, top=3)
get_psp_5 = partial(get_psp, top=5)


def get_psndcg(prediction: TPredict, targets: TTarget, inv_w: np.ndarray, mlb: TMlb = None,
			   classes: TClass = None, top=5):
	mlb = get_mlb(classes, mlb)
	log = 1.0 / np.log2(np.arange(top) + 2)
	psdcg = 0.0
	if not isinstance(targets, csr_matrix):
		targets = mlb.transform(targets)
	for i in range(top):
		p = mlb.transform(prediction[:, i: i+1]).multiply(inv_w)
		psdcg += p.multiply(targets).sum() * log[i]
	t, den = csr_matrix(targets.multiply(inv_w)), 0.0
	for i in range(t.shape[0]):
		num = min(top, len(t.getrow(i).data))
		den += -np.sum(np.sort(-t.getrow(i).data)[:num] * log[:num])
	return psdcg / den

get_psndcg_3 = partial(get_psndcg, top=3)
get_psndcg_5 = partial(get_psndcg, top=5)



parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--output_dir', required=True, type=str)
args = parser.parse_args()
dataset = args.dataset

preds = []
targets = []
with open(f'{dataset}/{dataset}_candidates.json') as fin1, \
	 open(os.path.join(args.output_dir, f'{dataset}_predictions_fute_1.json')) as fin2:
	for line1, line2 in zip(fin1, fin2):
		data1 = json.loads(line1)
		targets.append(data1['label'])
		data2 = json.loads(line2)
		pred = [x[0] for x in data2['predictions']]
		pred = pred[:5] + ['PAD']*(5-len(pred))
		preds.append(pred)
# targets = targets[:45000]
# preds = preds[:45000]
preds = np.array(preds)

mlb = MultiLabelBinarizer(sparse_output=True)
targets = mlb.fit_transform(targets)
p1, p3, p5, n3, n5 = get_p_1(preds, targets, mlb), get_p_3(preds, targets, mlb), get_p_5(preds, targets, mlb), \
	                 get_n_3(preds, targets, mlb), get_n_5(preds, targets, mlb)
print('P@1:', p1, ', ', \
	  'P@3:', p3, ', ', \
	  'P@5:', p5, ', ', \
	  'NDCG@3:', n3, ', ', \
	  'NDCG@5:', n5)
with open('scores.txt', 'a') as fout:
	fout.write('{:.4f}'.format(p1)+'\t'+'{:.4f}'.format(p3)+'\t'+'{:.4f}'.format(p5)+'\t'+ \
			   '{:.4f}'.format(n3)+'\t'+'{:.4f}'.format(n5)+'\n')

train_labels = []
with open(f'{dataset}/{dataset}_train.json') as fin:
	for line in fin:
		data = json.loads(line)
		train_labels.append(data['label'])
inv_w = get_inv_propensity(mlb.transform(train_labels), 0.55, 1.5)
psp1, psp3, psp5, psn3, psn5 = get_psp_1(preds, targets, inv_w, mlb), get_psp_3(preds, targets, inv_w, mlb), get_psp_5(preds, targets, inv_w, mlb), \
	                           get_psndcg_3(preds, targets, inv_w, mlb), get_psndcg_5(preds, targets, inv_w, mlb)
print('PSP@1:', psp1, ', ', \
	  'PSP@3:', psp3, ', ', \
	  'PSP@5:', psp5, ', ', \
	  'PSN@3:', psn3, ', ', \
	  'PSN@5:', psn5)
with open('scores.txt', 'a') as fout:
	fout.write('{:.4f}'.format(psp1)+'\t'+'{:.4f}'.format(psp3)+'\t'+'{:.4f}'.format(psp5)+'\t'+ \
			   '{:.4f}'.format(psn3)+'\t'+'{:.4f}'.format(psn5)+'\n')