# Install the packages in r1-v .
cd src/r1-v 
pip install -e ".[dev]"

# Addtional modules
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils torchvision

# vLLM support 
pip install vllm==0.7.2

# fix transformers version
pip install git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef

# need to install `cxx11abiFALSE` version of flash_attn, otherwise raise error
# https://github.com/Dao-AILab/flash-attention/issues/457
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
# pip install flash-attn --no-build-isolation

# extra dependencies for internvl
pip install timm==1.0.14

# install the dependencies for jupyter notebooks
pip install jupyterlab ipywidgets pickleshare

# extra dependencies for janus-pro
pip install attrdict==2.0.1
