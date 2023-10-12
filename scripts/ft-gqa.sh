#!/usr/bin/env bash
# Finetune on Multilingual Complex Wikidata Questions
#
#SBATCH --job-name gqa
#SBATCH --output=runs/R-%x.%j.out
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

# debug info
hostname
which python
nvidia-smi

# test cuda
python -c "import torch; print('device_count:', torch.cuda.device_count())"
python -c "import torch_geometric; print('torch_geometric version:', torch_geometric.__version__)"

export TOKENIZERS_PARALLELISM=true
export WANDB__SERVICE_WAIT=300

# run pretrain
# python -u train.py --finetune --config configs/gqa.yaml --config-profile finetune_gqa --run-name ft-gqa --tune-lr

# Resume from checkpoint
# python -u train.py --finetune --config configs/gqa.yaml --config-profile finetune_gqa \
#     --run-name ft-gqa --checkpoint-path logs/gqa/pdeeenyi/checkpoints/epoch=99-step=5300.ckpt \
#     --wandb-id pdeeenyi

# Test
python -u eval.py --config configs/gqa.yaml --config-profile test_gqa \
    --run-name test-gqa --model t5-gnn
