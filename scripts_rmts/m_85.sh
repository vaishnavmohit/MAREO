#!/bin/bash

# activate  env
source ../env/mareo/bin/activate

model_name=$1
device=0
lr=$2
step=${3:-6}
contextnorm="tasksegmented_contextnorm"

for r in {1..10}
do
	python ./train_and_eval_rmts.py --model_name $model_name --norm_type $contextnorm \
									--lr ${lr} --task RMTS --m_holdout 85 --epochs 50 \
									--run $r --device $device --step ${step} \
									--train_gen_method full_space --test_gen_method subsample
done

# 200 epochs for qv2