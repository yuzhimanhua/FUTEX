#!/bin/bash

dataset=MAG
model=fute_2

python data.py --dataset $dataset --model $model --label_only 1 --topN 5
./sample_run.sh $dataset $model
