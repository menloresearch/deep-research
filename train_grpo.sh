#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 --deepspeed_config configs/zero3.json verifiers-deepresearch/verifiers/examples/trl_deepresearch_search_visit_site_offline.py "$@" 