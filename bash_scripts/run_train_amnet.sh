#!/bin/bash

hidden_sizes=$1
device=$2
hs=($(echo $hidden_sizes | tr "," "\n"))

for h in "${hs[@]}"
do
  for run in $(seq 1 3)
  do
    echo $h $run
    python -m scripts.train_amnet --expr_name amnet --dataset DGraphFin --dataset_dir /data/huhy/datasets/ --hidden_size $h --device $device > log/amnet_$h_$run.log
  done
done