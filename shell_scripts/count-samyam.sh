#!/bin/bash

for k in 100
do
    python script/count_components.py -c config/new_nbfnet/wn18rr-count-samyam.yaml --gpus [0] --use_wandb no --k "$k"
done