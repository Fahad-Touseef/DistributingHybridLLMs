#!usr/bin/bash
source /miniconda3/etc/profile.d/conda.sh
conda activate py312


deepspeed sample_deepspeed.py\
    --deepspeed_config=ds_config.jsons