#!/bin/bash

# NCCL configs
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1 
export NCCL_P2P_DISABLE=1

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --multi_gpu --num_processes 4 --config_file configs/zero3_low_mem.json verifiers-deepresearch/verifiers/examples/trl_deepresearch_search_visit_site_offline.py "$@" 