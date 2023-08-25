dataset=MAGCS
model=specter

green=`tput setaf 2`
reset=`tput sgr0`

echo "${green}=====Module 1: Network-aware contrastive fine-tuning=====${reset}"
cd contrastive/
./prepare.sh ${dataset}
./run.sh ${dataset} ${model}
cd ../

echo "${green}=====Module 2: Hierarchy-aware aggregation=====${reset}"
python get_label_emb.py --dataset ${dataset} --model ${model}
python get_paper_emb.py --dataset ${dataset} --model ${model} --full_text 0
python bienc_sim.py --dataset ${dataset}
python roberta_large_mnli.py --dataset ${dataset}
python ensemble.py --dataset ${dataset} --mnli 1

echo "${green}=====Module 3: Self-training=====${reset}"
cd self_train/
./run.sh ${dataset}
cd ../

echo "${green}=====Evaluation=====${reset}"
python evaluation.py --dataset ${dataset} --model futex
