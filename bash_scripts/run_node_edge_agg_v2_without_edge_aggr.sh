#!/bin/bash

hidden_sizes=$1
device=$2
hs=($(echo $hidden_sizes | tr "," "\n"))

for h in "${hs[@]}"
do
  for run in $(seq 1 3)
  do
    echo $h $run
    python -m scripts.train_node_edge_agg_v2_without_edge_aggr --expr_name node_edge_agg_without_edge_aggr --dataset DGraphFin --dataset_dir /data/huhy/datasets/ --batch_size 512 --hidden_size $h --device $device --time_encoder normal > log/node_edge_agg_without_edge_aggr_$h_$run.log
  done
done

