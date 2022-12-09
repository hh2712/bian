#!/bin/bash

hidden_sizes=$1
device=$2
hs=($(echo $hidden_sizes | tr "," "\n"))

for h in "${hs[@]}"
do
  for run in $(seq 1 3)
  do
    echo $h $run
    python -m scripts.train_node_edge_agg_v3 --expr_name edge_attr_agg --dataset DGraphFin --dataset_dir /data/huhy/datasets/ --hidden_size $h --device $device --batch_size 512 --coalesce > log/edge_attr_agg_$h_$run.log
  done
done