# RAWExplainer
This repository contains the source code, datasets and configuration files for RAW-Explainer (Random Walk Explainer)

## Overview
RAW-Explainer is a random-walk–based explanation method that learns to extract small, connected subgraphs from knowledge graphs to explain GNN link prediction models. 


## Installation

We recommend using Conda with Python 3.10.

```bash
conda create --name RAWEx python=3.10
conda activate RAWEx
# conda install -c conda-forge cxx-compiler
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 cudatoolkit=11.8 -c pytorch -c nvidia
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-geometric==2.4.0 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install numpy==1.26.4 wandb ninja easydict pyyaml torcheval matplotlib pandas
conda install dglteam::dgl-cuda11.1
```


## Experiments on WN18RR, FB15k-237, NELL-995 and YAGO3-10
### GNN Training
To train a GNN in the default setup on WN18RR, run the following command.
```
python script/train_nbfnet.py -c config/nbfnet/wn18rr-0b.yaml --gpus [0] --use_wandb yes
```

### Training Robust GNNs
To train a robust GNN (GNN trained with randomized edge dropping), use either the default training setup or DDP approach with specification for `cfg.model.randomized_edge_drop` (refer to `config/nbfnet/wn18rr-015b.yaml`).

The result will be stored in `./experiments/NBFNet/{dataset-name}/{time-stamp}`.

## Explanations
The explainers are split into two categories: 
1. **Global Explainers** which include PGExplainer and RWExplainer
2. **Instance Explainers** which include GNNExplainer, PaGE-Link, RandomWalk and BFS


**Running Global Explainers**

To run any of the global explainers, use the following command:
```bash
python script/global_explanation.py -c config/explain/{config_file} --gpus [0] --use_wandb yes
```

**Running Instance Explainers**

To run any of the instance explainers, use the following command:

```bash
python script/instance_explanation.py -c config/explain/{config_file} --gpus [0] --use_wandb yes
```

The result will be stored in `./explanation/NBFNet/{dataset-name}/{time-stamp}`.

For models trained on YAGO3-10 (20% sampling), use `script/instance_explanation_sample.py` instead.

## Downstream Evaluation using GNN Backbone
Once explanations are generated, run the following command to feed the explanations into GNN backbone for evaluation using standard metrics. Ensure correct paths are set to `explainer_eval.expl_dir` and `checkpoint` for explanation directory and GNN backbone respectively.
```bash
python script/train_nbfnet.py -c config/nbfnet/wn18rr-0b-eval.yaml --gpus [0] --use_wandb yes
```

## End-to-end Training
Alternatively, both explanation generation and downstream evaluation can be performed end-to-end using the following command:
```
# Global Explainers
python script/global_explanation.py -c config/explain/{config_file} --gpus [0] --use_wandb yes --train_config config/nbfnet/wn18rr-0b-eval.yaml
```


```
# Instance Explainers
python script/instance_explanation.py -c config/explain/{config_file} --gpus [0] --use_wandb yes --train_config config/nbfnet/wn18rr-0b-eval.yaml
```

The result will be stored in `./explanation/NBFNet/{dataset-name}/{time-stamp}`.


## License

This project is licensed under the MIT License: see the LICENSE file for details.
