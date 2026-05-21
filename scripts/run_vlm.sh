#!/bin/bash

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

python src/merge.py \
  --run_name "six_llava_rank16" \
  --model_dirs llava_cfg/vizwiz_vqa_rank16.yaml \
  llava_cfg/iconqa_rank16.yaml \
  llava_cfg/chartqa_rank16.yaml \
  llava_cfg/docvqa_rank16.yaml \
  llava_cfg/coco_rank16.yaml \
  llava_cfg/flickr_rank16.yaml \
  --batch_size 1 --lamda 2.4 --kernel --learning_rate 3e-4
