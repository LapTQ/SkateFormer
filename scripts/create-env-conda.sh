#!/bin/bash -l

# conda_dir=/mnt/hdd10tb/Users/laptq/miniconda3
# [[ ! -d $conda_dir ]] && mkdir -p $conda_dir \
#     && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $conda_dir/miniconda.sh \
#         && bash $conda_dir/miniconda.sh -b -u -p $conda_dir \
#             && rm -rf $conda_dir/miniconda.sh \
#                 && $conda_dir/bin/conda init bash
# export PATH=$conda_dir/bin:$PATH

# conda_env_parent=/mnt/hdd10tb/Users/laptq/BlockGCN
conda_env_parent=/mnt/ssd8tb/shared_workspace/laptq/DeGCN_pytorch
conda_env_name=.conda-DeGCN
conda_env_path=$conda_env_parent/$conda_env_name
[[ ! -d $conda_env_path ]] && conda create --prefix=$conda_env_path python=3.10 -y
eval "$(conda shell.bash hook)"
conda init
conda activate $conda_env_path
which python3

pip install -e torchlight

pip install -r requirements.txt --default-timeout=10000