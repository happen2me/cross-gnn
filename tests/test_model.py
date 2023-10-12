"""Test model forward and backward pass."""""
import pytest
import torch
from transformers.optimization import Adafactor
from pytorch_lightning import Trainer

from lightning.lit_seq2seq import LitT5Seq2Seq


def test_forward(dummy_batch, model):
    """Test model forward pass."""
    (input_ids, attention_mask, decoder_labels,
        node_ids, node_type_ids, adj_lengths,
        edge_index, edge_type) = dummy_batch
    with torch.no_grad():
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=decoder_labels,
            node_ids=node_ids,
            node_type_ids=node_type_ids,
            adj_lengths=adj_lengths,
            edge_index=edge_index,
            edge_type=edge_type,
            return_dict=True
        )


def test_backward(dummy_batch, model):
    """Test model backward pass."""
    (input_ids, attention_mask, decoder_labels,
        node_ids, node_type_ids, adj_lengths,
        edge_index, edge_type) = dummy_batch
    optimizer = Adafactor(model.parameters(), lr=0.001, relative_step=False)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=decoder_labels,
        node_ids=node_ids,
        node_type_ids=node_type_ids,
        adj_lengths=adj_lengths,
        edge_index=edge_index,
        edge_type=edge_type,
        return_dict=True
    )
    loss = outputs.loss
    loss.backward()
    optimizer.step()


def test_lightning(args, train_loader, encoder, decoder):
    """Test lightning model forward pass."""
    model = LitT5Seq2Seq(
        args=args,
        encoder=encoder,
        decoder=decoder,
        freeze_lm=False,
        freeze_non_lm=False,
        mode='pretrain')
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model, train_dataloaders=train_loader)
