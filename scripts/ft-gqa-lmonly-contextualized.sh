#!/usr/bin/env bash
# Finetune on Graph Question Answering (GQA) dataset
#
#SBATCH --job-name gqa-lmctx
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
# python -u train_lm.py --config configs/gqa.yaml --config-profile finetune_gqa_lmonly_contextualized \
#     --run-name ft-gqa-lmonly-ctx-unfrozen-fp16  --num-trainable-blocks -1 --fp16

# frozen with adapter lora
# python -u train_lm.py --config configs/gqa.yaml --config-profile finetune_gqa_lmonly_contextualized \
#     --run-name ft-gqa-lmonly-ctx-lora  --num-trainable-blocks 0 --adapter lora --tune-lr

# frozen with adapter pfeiffer
# python -u train_lm.py --config configs/gqa.yaml --config-profile finetune_gqa_lmonly_contextualized \
#     --run-name ft-gqa-lmonly-ctx-pfeiffer  --num-trainable-blocks 0 --adapter pfeiffer

# Test
# python -u eval.py --config configs/gqa.yaml --config-profile test_gqa_lmonly_contextualized \
#     --run-name test-gqa-lmonly-ctx-pfeiffer --model t5 --add-adapter

python -u eval.py --config configs/gqa.yaml --accelerator gpu \
 --config-profile test_gqa_lmonly_contextualized --run-name test-gqa-lmonly-ctx-pfeiffer \
 --model t5 --add-adapter
