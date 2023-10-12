"""Pretrain the T5-based encoder-decoder architectured dragon model.
"""
import os
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoModelForSeq2SeqLM
from transformers.adapters import PfeifferConfig

from dataset.lmgnn import load_test_data
from lightning.lit_seq2seq import LitT5Seq2Seq
from lightning.lit_t5 import LitT5
from models.flamingo import FlamingoConfig
from models.t5 import FlamingoT5Decoder
from utils.common import load_args
from utils.model_utils import construct_encoder


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
    config_profile = args.config_profile
    devices = args.devices

    # 2. Load data
    # Set collator and dataset according to the task: pretrain is mainly devided into two types:
    # with or without graph
    dummy_graph = hasattr(args, 'no_graph') and args.no_graph
    test_kwargs={'encoder_input': args.encoder_input, 'decoder_label': 'raw_answers'}
    test_loader = load_test_data(
        args,
        dummy_graph=dummy_graph,
        num_workers=8,
        test_kwargs=test_kwargs,)

    # 4. Create pytorch lightning model
    if args.model == 't5-gnn':
        encoder = construct_encoder(args)
        decoder_config = FlamingoConfig(
            d_model=encoder.config.d_model,
            dim_media=args.gnn_dim,
            xattn_dim_head=args.xattn_dim_head,
            xattn_heads=args.xattn_heads,
            xattn_every=args.xattn_every,
            xattn_ff_mult=args.xattn_ff_mult,
            lm_name_or_path=args.encoder_name_or_path,)
        decoder = FlamingoT5Decoder(decoder_config, encoder.get_input_embeddings())
        model = LitT5Seq2Seq.load_from_checkpoint(
            args.checkpoint_path, strict=False,
            args=args, encoder=encoder, decoder=decoder,
            freeze_lm=args.freeze_lm, freeze_non_lm=args.freeze_non_lm,
            map_location=args.accelerator,
            mode='finetune'
        )
    elif args.model == 't5':
        t5 = AutoModelForSeq2SeqLM.from_pretrained(args.encoder_name_or_path)
        if args.add_adapter:
            config = PfeifferConfig()
            t5.add_adapter("adapter", config=config)
            t5.train_adapter("adapter")
        model = LitT5.load_from_checkpoint(args.checkpoint_path, strict=False, args=args,
                                           model=t5, map_location=args.accelerator)
    else:
        raise ValueError(f"Unknown model type: {args.model}")


    # 5. Create trainer
    now = datetime.now().strftime('%m%d%H%M')
    run_name = f"{args.run_name}-{now}"
    offline = args.wandb_mode in ['offline', 'disabled']
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    wandb_logger = WandbLogger(project=args.wandb_project, offline=offline, name=run_name,
                               group=config_profile, save_dir=args.log_dir)
    wandb_logger.experiment.config.update(vars(args), allow_val_change=True)

    trainer = pl.Trainer(max_epochs=args.n_epochs, fast_dev_run=args.fast_dev_run,
                         default_root_dir=os.path.join(args.save_dir, args.run_name),
                         accelerator=args.accelerator, strategy=args.strategy, logger=wandb_logger,
                         gradient_clip_val=0.5,
                         accumulate_grad_batches=8, devices=devices)

    # 6. Train
    trainer.test(model, dataloaders=test_loader)


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
    parser.add_argument('--model', type=str, choices=('t5', 't5-gnn'), required=True)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--checkpoint-path', type=str, default=None)
    parser.add_argument('--add-adapter', action='store_true', help='Add pfeiffer adapter to T5')
    args = parser.parse_args()

    # Delete arguments that are not set so that they won't override the config file
    if not args.checkpoint_path:
        del args.checkpoint_path

    loaded_args = load_args(config_path=args.config, profile=args.config_profile)
    loaded_args.__dict__.update(args.__dict__)

    main(loaded_args)
