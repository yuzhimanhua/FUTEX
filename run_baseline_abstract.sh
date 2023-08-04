dataset=MAG

# model="scibert_scivocab_uncased"
model="specter"
# model="biolinkbert-base"
python 1_get_paper_emb.py --dataset ${dataset} --model ${model}
python 2_get_label_emb.py --dataset ${dataset} --model ${model}
python 3_calcsim_abs.py --dataset ${dataset} --model ${model}