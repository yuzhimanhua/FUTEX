dataset=MAG
# dataset=PubMed
# dataset=Art

green=`tput setaf 2`
reset=`tput sgr0`

echo "${green}=====Step 1: Preparing Testing Data=====${reset}"
python prepare_test.py --dataset ${dataset}

echo "${green}=====Step 2: Generating Training Data=====${reset}"
python prepare_train.py --dataset ${dataset}

if [ "${dataset}" == "MAG" ]; then
	head -90000 ${dataset}_input/dataset.txt > ${dataset}_input/train.txt
	sed -n '90001,99000p' ${dataset}_input/dataset.txt > ${dataset}_input/dev.txt
elif [ "${dataset}" == "PubMed" ]; then
	head -300000 ${dataset}_input/dataset.txt > ${dataset}_input/train.txt
	sed -n '300001,330000p' ${dataset}_input/dataset.txt > ${dataset}_input/dev.txt
else
	head -160 ${dataset}_input/dataset.txt > ${dataset}_input/train.txt
	sed -n '161,170p' ${dataset}_input/dataset.txt > ${dataset}_input/dev.txt
fi