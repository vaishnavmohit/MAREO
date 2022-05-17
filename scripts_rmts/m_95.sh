#!/bin/bash

# activate  env
source ../env/mareo/bin/activate

model_name=$1
device=0
lr=$2
step=${3:-6}
epochs=300
contextnorm="tasksegmented_contextnorm"

for r in {1..10}
do
	python ./train_and_eval_rmts.py 	--model_name $model_name --norm_type $contextnorm \
										--lr ${lr} --task RMTS --m_holdout 95 --epochs 300 \
										--run $r --device $device --step ${step} \
										--train_gen_method full_space --test_gen_method subsample
done

# 400 for best result