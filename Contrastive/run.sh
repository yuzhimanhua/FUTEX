dataset=MAG
architecture=cross

python3 main.py --bert_model /shared/data2/yuz9/BERT_models/specter/ \
				--train_dir ${dataset}_input/ --test_file ${dataset}_input/test.txt \
				--output_dir ${dataset}_output/ --architecture ${architecture} --use_pretrain \
				--train_batch_size 8 --eval_batch_size 128 --num_train_epochs 3
python3 main.py --bert_model /shared/data2/yuz9/BERT_models/specter/ \
				--train_dir ${dataset}_input/ --test_file ${dataset}_input/test.txt \
				--output_dir ${dataset}_output/ --architecture ${architecture} --use_pretrain --eval \
				--train_batch_size 8 --eval_batch_size 128 --num_train_epochs 3

python3 postprocess.py --dataset ${dataset} --output_dir ${dataset}_output/ --architecture ${architecture}
python3 patk.py --dataset ${dataset} --output_dir ${dataset}_output/
