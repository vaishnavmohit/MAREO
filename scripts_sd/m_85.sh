#!/bin/bash

# activate  env
source ../env/mareo/bin/activate

model_name=$1
device=0
lr=$2
step=${3:-2}
contextnorm="contextnorm"

for r in {1..10}
do
	python ./train_and_eval.py 	--model_name ${model_name} --norm_type ${contextnorm} \
								--lr ${lr} --task same_diff --m_holdout 85 --epochs 100 \
								--run $r --device ${device} --step ${step} 
done

# 200 epochs for qv2