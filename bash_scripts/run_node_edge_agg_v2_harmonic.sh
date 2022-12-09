#!/bin/bash


#!/bin/bash

hidden_sizes=$1
device=$2
hs=($(echo $hidden_sizes | tr "," "\n"))

for h in "${hs[@]}"
do
  for run in $(seq 1 3)
  do
    echo $h $run
    python -m scripts.train_node_edge_agg_v2 --expr_name node_edge_agg_harmonic --dataset DGraphFin --dataset_dir /data/huhy/datasets/ --batch_size 512 --hidden_size $h --device $device --time_encoder harmonic > log/node_edge_agg_harmonic_$h_$run.log
  done
done

