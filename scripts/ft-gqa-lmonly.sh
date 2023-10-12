#!/usr/bin/env bash
# Finetune on Graph Question Answering (GQA) dataset
#
#SBATCH --job-name gqa-lmonly
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

# unfrozen
# python -u train_lm.py --config configs/gqa.yaml --config-profile finetune_gqa_lmonly \
#     --run-name ft-gqa-lmonly-unfrozen  --num-trainable-blocks -1
# continue unfrozen
# python -u train_lm.py --config configs/gqa.yaml --config-profile finetune_gqa_lmonly \
#     --run-name ft-gqa-lmonly-unfrozen  --num-trainable-blocks -1 \
#     --checkpoint-path logs/gqa/eoqaf32k/checkpoints/epoch=98-step=5247.ckpt \
#     --wandb-id eoqaf32k

# frozen with adapter lora
# python -u train_lm.py --config configs/gqa.yaml --config-profile finetune_gqa_lmonly \
#     --run-name ft-gqa-lmonly-lora  --num-trainable-blocks 0 --adapter lora --tune-lr

# frozen with adapter pfeiffer
# python -u train_lm.py --config configs/gqa.yaml --config-profile finetune_gqa_lmonly \
#     --run-name ft-gqa-lmonly-pfeiffer  --num-trainable-blocks 0 --adapter pfeiffer --tune-lr

# Test
python -u eval.py --config configs/gqa.yaml --config-profile test_gqa_lmonly \
    --run-name test-gqa-lmonly --model t5
