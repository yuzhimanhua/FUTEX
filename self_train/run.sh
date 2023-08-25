dataset=$1
model=ensemble

data_dir="./Sandbox/Data/${dataset}"
results_dir="./Sandbox/Results/${dataset}"
model_dir="./Sandbox/Results/${dataset}/model"

trn_ft_file="${data_dir}/trn_X_Xf.txt"
trn_lbl_file="${data_dir}/trn_X_Y.txt"
tst_ft_file="${data_dir}/tst_X_Xf.txt"
tst_lbl_file="${data_dir}/tst_X_Y.txt"
score_file="${results_dir}/score_mat.txt"

python preprocess.py --dataset ${dataset} --model ${model} --topN 5
./Parabel/parabel_train ${trn_ft_file} ${trn_lbl_file} ${model_dir} -t 3 -s 0 -T 3 -b 1.0 -c 1.0 -m 100 -tcl 0.1 -tce 0 -e 0.0001 -n 20 -k 0 -q 0
./Parabel/parabel_predict ${tst_ft_file} ${model_dir} ${score_file} -t 3
python postprocess.py --dataset ${dataset} --model ${model}
