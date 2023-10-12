"""Test DDP training with WandbLogger."""
from copy import deepcopy

import pytest
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from lightning.lit_seq2seq import LitT5Seq2Seq



def test_ddp_and_wandb(args, train_loader, encoder, decoder):
    """Test DDP training with WandbLogger. Skip if no GPU available."""
    if torch.cuda.device_count() == 0:
        # skip test if no GPU
        pytest.skip('No GPU available for testing DDP')
    args = deepcopy(args)
    args.batch_size = 2
    model = LitT5Seq2Seq(
        args=args,
        encoder=encoder,
        decoder=decoder,
        freeze_lm=False,
        freeze_non_lm=False,
        mode='pretrain')
    wandb_logger = WandbLogger(project=args.wandb_project, offline=True)
    device_cnt = 2 if torch.cuda.device_count() > 1 else 1
    trainer = Trainer(accelerator='gpu', strategy='ddp', devices=device_cnt,
                      logger=wandb_logger, fast_dev_run=True)
    trainer.fit(model, train_loader)
