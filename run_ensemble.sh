dataset=Art
if [ "${dataset}" == "MAG" ]; then
    mnli=0
else
    mnli=1
fi

# python 7_ensemble.py --dataset ${dataset} --model scibert --mnli ${mnli}
# python 7_ensemble.py --dataset ${dataset} --model oagbert --mnli ${mnli}
python 7_ensemble.py --dataset ${dataset} --model specter --mnli ${mnli}
# python 7_ensemble.py --dataset ${dataset} --model linkbert --mnli ${mnli}