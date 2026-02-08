#!/bin/bash

# Miniconda installation
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda init --all

# Ultra Env setup
conda create --name Ultra python=3.10
conda activate Ultra
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 cudatoolkit=11.8 -c pytorch -c nvidia
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-geometric==2.4.0 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install numpy==1.26.4 wandb ninja easydict pyyaml torcheval matplotlib pandas
conda install dglteam::dgl-cuda11.1

## ENV MAY NEED TO BE UPDATED
conda activate Ultra && conda remove -y mkl mkl-service mkl_fft mkl_random intel-openmp --force && conda install -y mkl mkl-service mkl_fft mkl_random intel-openmp -c conda-forge
conda activate Ultra && python -c "import torch; print('PyTorch version:', torch.__version__); print('Import successful!')"
conda activate Ultra && conda install -y "mkl<2025" "mkl-service<2025" "intel-openmp<2025" -c conda-forge --force-reinstall
conda activate Ultra && python -c "import torch; print('PyTorch version:', torch.__version__); print('Import successful!')"

## wandb setup
# on the machine with wandb installed, run the following command to get the api key (or get it from the website):
cat ~/.netrc
# on the new machine, run the following to login to wandb and paste the api key:
wandb login 

# GitHub
git clone https://github.com/ryoji-kubo/Graph-Transformer.git
cd Graph-Transformer
git checkout samplex
