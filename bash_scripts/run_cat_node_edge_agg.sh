#!/bin/bash

hidden_sizes=$1
device=$2
hs=($(echo $hidden_sizes | tr "," "\n"))

for h in "${hs[@]}"
do
  for run in $(seq 1 3)
  do
    echo $h $run
    python -m scripts.train_node_edge_agg --expr_name cat_node_edge_agg --dataset DGraphFin --dataset_dir /data/huhy/datasets/ --batch_size 512 --hidden_size $h --device $device --time_encoder normal > log/cat_node_edge_agg_$h_$run.log
  done
done
