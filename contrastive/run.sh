dataset=$1
model=$2
architecture=cross

if [ "${dataset}" == "Art" ]; then
	epochs=20
else
	epochs=2
fi

python3 main.py --bert_model ../${model} \
				--train_dir ${dataset}_input/ --test_file ${dataset}_input/test.txt \
				--output_dir ${dataset}_output/ --architecture ${architecture} --use_pretrain \
				--train_batch_size 8 --eval_batch_size 128 --num_train_epochs ${epochs}
python3 main.py --bert_model ../${model} \
				--train_dir ${dataset}_input/ --test_file ${dataset}_input/test.txt \
				--output_dir ${dataset}_output/ --architecture ${architecture} --use_pretrain --eval \
				--train_batch_size 8 --eval_batch_size 128 --num_train_epochs ${epochs}

python3 postprocess.py --dataset ${dataset} --output_dir ${dataset}_output/ --architecture ${architecture}
