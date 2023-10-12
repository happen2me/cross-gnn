"""Pretrain the T5-based encoder-decoder architectured dragon model.
"""
import os
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateFinder
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoModelForSeq2SeqLM
from transformers.adapters import LoRAConfig, PfeifferConfig, CompacterConfig

from dataset.lmgnn import load_data as load_lmgnn_data
from dataset.multiple_choice import load_data as load_multiple_choice_data
from lightning.lit_t5 import LitT5
from lightning.lit_multiple_choice import LitT5LMForMultipleChoice
from utils.common import load_args


def get_adapter_config(adapter_name):
    assert adapter_name in ['lora', 'pfeiffer', 'compacter']
    if adapter_name == 'pfeiffer':
        config = PfeifferConfig()
    elif adapter_name == 'lora':
        config = LoRAConfig()
    elif adapter_name == 'compacter':
        config = CompacterConfig()
    return config

def freeze_params(model, num_trainable_blocks):
    """Freeze all the parameters except the last num_trainable_blocks 
    and lm head in the decoder and the lm_head.

    Args:
        model (T5ForConditionalGeneration)
        num_trainable_blocks (int): Number of trainable blocks in the decoder.
            if num_trainable_blocks == -1, all the parameters will be trainable.
    """
    if num_trainable_blocks == -1:
        for param in model.parameters():
            param.requires_grad = True
        return
    for param in model.parameters():
        param.requires_grad = False
    num_decoder_blocks = len(model.decoder.block)
    for i, block in enumerate(model.decoder.block):
        if i > num_decoder_blocks - num_trainable_blocks - 1:
            for param in block.parameters():
                param.requires_grad = True
    for param in model.lm_head.parameters():
        param.requires_grad = True


def main(args):
    # 0. Set seed
    # Setting all the random seeds to the same value.
    # This is important in a distributed training setting. 
    # Each rank will get its own set of initial weights. 
    # If they don't match up, the gradients will not match either,
    # leading to training that may not converge.
    pl.seed_everything(1)

    # 1. Load configs
    run_name = args.run_name
    num_trainable_blocks = args.num_trainable_blocks
    multiple_choice = args.multiple_choice
    inject_choice = args.inject_choice
    adapter_name = args.adapter
    devices = args.devices
    fp16 = args.fp16
    mode = 'finetune'
    config_profile = args.config_profile

    # 2. Load data
    # Set collator and dataset according to the task: pretrain is mainly devided into two types:
    # with or without graph
    dummy_graph = True
    train_kwargs={'encoder_input': args.encoder_input, 'decoder_label': args.decoder_label}
    train_kwargs={'encoder_input': args.encoder_input, 'decoder_label': args.decoder_label}
    val_kwargs={'encoder_input': args.encoder_input, 'decoder_label': args.decoder_label}
    if mode == 'finetune' and not multiple_choice:
        val_kwargs['decoder_label'] = 'raw_answers'

    if multiple_choice:
        train_loader, val_loader = load_multiple_choice_data(
            args,
            corrupt=False,
            dummy_graph=dummy_graph,
            num_workers=8,
            train_kwargs=train_kwargs,
            val_kwargs=val_kwargs,
            num_choices=args.num_choices,
            has_choice_graph=inject_choice,)
    else:
        train_loader, val_loader = load_lmgnn_data(
            args,
            corrupt=False,
            dummy_graph=dummy_graph,
            num_workers=8,
            train_kwargs=train_kwargs,
            val_kwargs=val_kwargs,)

    # 3. Create encoder and decoder
    t5 = AutoModelForSeq2SeqLM.from_pretrained(args.encoder_name_or_path)
    freeze_params(t5, num_trainable_blocks)
    if adapter_name is not None:
        config = get_adapter_config(adapter_name)
        t5.add_adapter("adapter", config=config)
        t5.train_adapter("adapter")

    # 4. Create wandb logger
    now = datetime.now().strftime('%m%d%H%M')
    run_name = f"{args.run_name}-{now}"
    offline = args.wandb_mode in ['offline', 'disabled']
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    wandb_kwargs = {}
    if args.wandb_id:
        wandb_kwargs['id'] = args.wandb_id
    wandb_logger = WandbLogger(project=args.wandb_project, offline=offline, name=run_name,
                               group=config_profile, save_dir=args.log_dir, **wandb_kwargs)
    wandb_logger.experiment.config.update(vars(args), allow_val_change=True)

    # 5. Create pytorch lightning model
    if multiple_choice:
        model_cls = LitT5LMForMultipleChoice
    else:
        model_cls = LitT5
    model = model_cls(args, t5)

    # 6. Callbacks: lr finder and checkpoint
    callbacks = []
    if args.tune_lr:
        lr_finder = LearningRateFinder()
        callbacks.append(lr_finder)
    if mode == 'finetune' and not multiple_choice:
        checkpoint_callback = ModelCheckpoint(monitor="em", mode="max", save_weights_only=True,)
        callbacks.append(checkpoint_callback)

    # 7. Create trainer
    optional_kwargs = {}
    if fp16:
        optional_kwargs['precision'] = '16-mixed'
    trainer = pl.Trainer(max_epochs=args.n_epochs, fast_dev_run=args.fast_dev_run,
                         default_root_dir=os.path.join(args.save_dir, args.run_name),
                         accelerator='gpu', strategy=args.strategy, logger=wandb_logger,
                         callbacks=callbacks, gradient_clip_val=0.5,
                         accumulate_grad_batches=8, devices=devices, **optional_kwargs)

    # 8. Train
    if args.restore_training:
        resume_checkpoint = args.checkpoint_path
        assert resume_checkpoint, "No checkpoint to resume training. (got {resume_checkpoint})"
    else:
        resume_checkpoint = None
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=resume_checkpoint)


if __name__ == '__main__':
    # To properly utilize a CUDA device with tensor cores
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark = True
    # To avoid 'too many open files' error
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/lmgnn.yaml')
    parser.add_argument('--config-profile', type=str, required=True)
    parser.add_argument('--run-name', type=str, required=True)
    parser.add_argument('--num-trainable-blocks', type=int, help='Number of trainable blocks in the decoder.')
    parser.add_argument('--multiple-choice', action='store_true', help='Whether it is multiple choice task.')
    parser.add_argument('--inject-choice', action='store_true', help='Whether to inject choice into the input.')
    parser.add_argument('--adapter', default=None, help='Whether to add adapter to the model. No adapter if not set.')
    parser.add_argument('--devices', type=int, default=1, help='Number of devices to use.')
    parser.add_argument('--fp16', action='store_true', help='Whether to use fp16.')
    parser.add_argument('--tune-lr', action='store_true', help='Whether to tune learning rate.')
    parser.add_argument('--wandb-id', type=str, default=None)
    parser.add_argument('--checkpoint-path', type=str, default=None)
    args = parser.parse_args()

    # Delete arguments that are not set so that they won't override the config file
    if not args.checkpoint_path:
        del args.checkpoint_path

    loaded_args = load_args(config_path=args.config, profile=args.config_profile)
    loaded_args.__dict__.update(args.__dict__)

    main(loaded_args)
