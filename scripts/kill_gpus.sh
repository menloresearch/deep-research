#!/bin/bash

echo "Killing all Python processes and clearing GPU memory..."
pkill -f python || true
sleep 2
pkill -9 -f python || true
sleep 1
python -c "import torch; torch.cuda.empty_cache()" || true
echo "GPU processes killed and memory cleared." 