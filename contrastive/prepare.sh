dataset=$1

python prepare_train.py --dataset ${dataset}

if [ "${dataset}" == "MAGCS" ]; then
	head -90000 ${dataset}_input/dataset.txt > ${dataset}_input/train.txt
	sed -n '90001,99000p' ${dataset}_input/dataset.txt > ${dataset}_input/dev.txt
elif [ "${dataset}" == "PubMed" ]; then
	head -300000 ${dataset}_input/dataset.txt > ${dataset}_input/train.txt
	sed -n '300001,330000p' ${dataset}_input/dataset.txt > ${dataset}_input/dev.txt
elif [ "${dataset}" == "Art" ]; then
	head -150 ${dataset}_input/dataset.txt > ${dataset}_input/train.txt
	sed -n '151,170p' ${dataset}_input/dataset.txt > ${dataset}_input/dev.txt
fi

python prepare_test.py --dataset ${dataset}
