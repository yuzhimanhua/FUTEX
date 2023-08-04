dataset=Art

# model="scibert_scivocab_uncased"
model="specter"
# model="linkbert-base"
python 5_hierarchical_aggr.py --dataset ${dataset} --model ${model}
python 2_get_label_emb.py --dataset ${dataset} --model ${model}
python 6_calcsim_full.py --dataset ${dataset} --model ${model}