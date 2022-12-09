## BIAN 
By Hanyi Hu, Long Zhang, Shuan Li, Zhi Liu, Yao Yang and Chongning Na

### Python Dependencies
    torch
    torch_geometric
    torchmetrics
    torch_scatter
    mlflow

### Reproduce Experiment BIAN Result
#### BIAN with t2v
    bash bash_scripts/run_node_edge_agg_v2_harmonic.sh ${hidden_size} ${gpu_device}

#### BIAN with Random Temporal Encoding Weight Initialization
    bash bash_scripts/run_node_edge_agg_v2.sh ${hidden_size} ${gpu_device}

#### BIAN with Edge Attribute Information
	bash bash_scripts/run_train_edge_attr_agg.sh ${hidden_size} ${gpu_device}

#### BIAN with t2v and Edge Attribute Information
	bash bash_scripts/run_node_edge_agg_v4.sh ${hidden_size} ${gpu_device}

#### BIAN with t2v and Edge Attribute Information
	bash bash_scripts/run_node_edge_agg_v4.sh ${hidden_size} ${gpu_device}


#### BIAN w/o Fusion by attention
	bash bash_scripts/run_cat_node_edge_agg.sh ${hidden_size} ${gpu_device}


#### Edge Branch Only
	bash bash_scripts/run_train_edgegcn.sh ${hidden_size} ${gpu_device}

### Citation
If you use these models in your research, please cite:

	@article{hu2022fradulent,
	  title={Fradulent User Detection Via Behavior Information Aggregation Network (BIAN) On Large-Scale Financial Social Network},
	  author={Hu, Hanyi and Zhang, Long and Li, Shuan and Liu, Zhi and Yang, Yao and Na, Chongning},
	  journal={arXiv preprint arXiv:2211.06315},
	  year={2022}
	}