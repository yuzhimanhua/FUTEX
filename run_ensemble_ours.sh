dataset=Art
if [ "${dataset}" == "MAG" ]; then
	mnli=0
else
	mnli=1
fi

python 9_ensemble_ours.py --dataset ${dataset} --mnli ${mnli}
python 8_pspatk.py --dataset ${dataset} --model fute_2 --patk 0