#!/bin/bash

hidden_sizes=$1
device=$2
hs=($(echo $hidden_sizes | tr "," "\n"))

for h in "${hs[@]}"
do
  for run in $(seq 1 3)
  do
    echo $h $run
    python -m scripts.train_edgegcn --expr_name edgegcn --dataset DGraphFin --dataset_dir /data/huhy/datasets/ --hidden_size $h --dropout 0 --device $2 --learning_rate 0.003 --batch_size 8196 --time_encoder normal > log/edgegcn_$h_$run.log
  done
done