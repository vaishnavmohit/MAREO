#!/bin/bash

# activate  env
source ../env/mareo/bin/activate

model_name=$1
device=0
lr=$2
step=${3:-9}
contextnorm="contextnorm"

for r in {1..10}
do
	python ./train_and_eval_dist3.py --model_name $model_name --norm_type $contextnorm \
									--lr ${lr} --task dist3 --m_holdout 50 --epochs 50 \
									--run $r --device $device --step ${step} 
done