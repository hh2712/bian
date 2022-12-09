#!/bin/bash


curpath=`pwd`
echo $curpath
num_run=1
hidden_size="128"
dropout="0"


find_gpu()
{
    res=$(nvidia-smi | \
        grep -E "[0-9]+MiB\s*/\s*[0-9]+MiB" | \
        awk '{print ($9" "$11)}' | \
        sed "s/\([0-9]\{1,\}\)MiB \([0-9]\{1,\}\)MiB/\1 \2/" | \
        awk '{print $2 - $1}')

	# 按可用显存降序排列
	# 格式：<GPU ID> <FREE MEMORY>
	# 外套 `()` 可以转为列表
	i=0
	res=($(for s in $res; do echo $i $s && i=`expr 1 + $i`; done | \
		sort -n -k 2 -r))

	# 第一个参数：需要 GPU 数，默认 = 1
	n_gpu_req=${1-"1"}
	# 第二个参数：可用显存下界，超过才选，默认 = 0
	mem_lb=${2-"0"}
	echo "Requiring ${n_gpu_req} GPUs with at least ${mem_lb}MB free memory"

    gpu_id=-1
    n=0
	for i in $(seq 0 2 `expr ${#res[@]} - 1`); do
		gid=${res[i]}
		mem=${res[i+1]}
#		echo $gid: $mem
		if [ $n -lt ${n_gpu_req} -a $mem -ge ${mem_lb} ]; then
			if [ $n -eq 0 ]; then
				gpu_id=$gid
			else
				gpu_id=${gpu_id}","$gid
			fi
			n=`expr 1 + $n`
		else
			# 要么够数，要么后面的 GPU 可用显存都不够（因为已经降序排）
			break
		fi
    done

	# if [ $n -lt ${n_gpu_req} ]; then
}


for h in $hidden_size
do
  for d in $dropout
  do
    for run in $(seq 1 3)
    do
      gpu_id=-1
      find_gpu 1
      echo $h $d $run
      python -m scripts.train_node_edge_agg --hidden_size $h --dropout $d --device $gpu_id
    done
  done
done

