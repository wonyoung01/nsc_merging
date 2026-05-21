#!/bin/bash

mkdir -p ./checkpoints/llava

# With hf CLI installed

hf download wylee01/LLaVA-1.5-7B-VizWizVQA-LoRA --local-dir ./checkpoints/llava/vizwiz_vqa/
hf download wylee01/LLaVA-1.5-7B-IconQA-LoRA --local-dir ./checkpoints/llava/iconqa/
hf download wylee01/LLaVA-1.5-7B-ChartQA-LoRA --local-dir ./checkpoints/llava/chartqa/
hf download wylee01/LLaVA-1.5-7B-DocVQA-LoRA --local-dir ./checkpoints/llava/docvqa/
hf download wylee01/LLaVA-1.5-7B-COCO-LoRA --local-dir ./checkpoints/llava/coco/
hf download wylee01/LLaVA-1.5-7B-Flickr30k-LoRA --local-dir ./checkpoints/llava/flickr/
