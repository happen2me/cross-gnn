from abc import abstractmethod
import evaluate
import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, RMSprop
from transformers import AutoTokenizer
from transformers.optimization import Adafactor
from deepspeed.ops.adam import DeepSpeedCPUAdam

from models.t5 import T5Seq2Seq


class LitT5ForMultipleChoice(pl.LightningModule):
    @staticmethod
    def squash_choice_dim(tensor):
        return tensor.view(-1, *tensor.shape[2:])

    @abstractmethod
    def batch_forward(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        output = self.batch_forward(batch)
        losses = output.loss
        # Convert loss to scores: lower loss equals higher score
        scores = -losses
        # Cross entropy over the scores
        choice_labels = batch[3]
        choice_labels = choice_labels.view(-1)
        loss = torch.nn.functional.cross_entropy(scores, choice_labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.batch_forward(batch)
        losess = output.loss
        assert len(losess.shape) == 2
        predictions = np.argmin(losess.cpu().numpy(), axis=-1).reshape(-1)
        choice_labels = batch[3].cpu().numpy().reshape(-1)
        accuracy = self.evaluator.compute(predictions=predictions, references=choice_labels)
        self.validation_step_outputs.append(accuracy)
        return accuracy

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if len(outputs) > 0:
            keys = outputs[0].keys()
            scores = {k: np.mean([o[k] for o in outputs]) for k in keys}
        else:
            scores = {}
        self.log_dict(scores)
        self.validation_step_outputs.clear()
        return scores

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def configure_optimizers(self):
        """Create an optimizer for the model, optionally using different learning rates for different layers.
        If use_ddp is True, the optimizer will be wrapped by DistributedDataParallel.
        """
        parameters = self.model.parameters()
        learning_rate = float(self.learning_rate)
        if self.args.optimizer == "deepspeed_offload":
            optimizer = DeepSpeedCPUAdam(parameters, lr=learning_rate)
        elif self.args.optimizer == "adamw":
            optimizer = AdamW(parameters, lr=learning_rate)
        elif self.args.optimizer == "adafactor":
            # Set according to https://discuss.huggingface.co/t/t5-finetuning-tips/684/3
            optimizer = Adafactor(parameters, lr=learning_rate, scale_parameter=False,
                                  clip_threshold=1.0, relative_step=False)
        elif self.args.optimizer == "rmsprop":
            optimizer = RMSprop(parameters, lr=learning_rate)
        else:
            raise NotImplementedError(f"Optimizer {self.args.optimizer} is not supported.")
        # Load the optimizer state if resuming
        return optimizer


class LitT5GNNForMultipleChoice(LitT5ForMultipleChoice):
    """T5 with GNN encoder for multiple choice tasks."""
    def __init__(self, args, encoder, decoder, freeze_lm=True, freeze_non_lm=False,
                 mode='pretrain', return_val_predictions=False):
        """Initialize a T5Seq2Seq model.

        Warning: the decoder_start_token_id will be initialized as the pad_token_id of a
        tokenizer constructed from args.encoder_name_or_path tokenizer.

        Args:
            args (argparse.Namespace): arguments used to initialize the model.
            encoder (T5Encoder): a T5Encoder model.
            decoder (T5Decoder): a T5Decoder model.
            freeze_lm (bool, optional): whether to freeze the weights of the T5 model. Defaults to True.
            freeze_non_lm (bool, optional): whether to freeze the weights of the added parts. Defaults to False.
            mode (str, optional): the mode of the model, either 'pretrain' or 'finetune'. Defaults to 'pretrain'.
                This influences the validation step: in 'pretrain' mode, the validation step returns the perplexity
                of the validation set; in 'finetune' mode, the validation step returns the F1 score of the validation
                set.
            return_val_predictions (bool, optional): whether to return the predictions of the validation set. Defaults to False.
        """
        assert mode in ['pretrain', 'finetune']
        super().__init__()
        self.validation_step_outputs = []
        self.save_hyperparameters(args)
        self.args = args
        # Freeze node embedding (duplicated but important)
        for n, p in encoder.named_parameters():
            if n.endswith("node_emb.emb.weight"):
                p.requires_grad = False
        # Freeze loaded weights from T5, GNN and XATTN is not frozen
        if freeze_lm:
            encoder.freeze_lm()
            decoder.freeze_lm()
        # Freeze the added parts (GNN & XATTN), T5 is not frozen
        if freeze_non_lm:
            encoder.freeze_non_lm()
            decoder.freeze_non_lm()
        # Construct a encoder-decoder model
        model = T5Seq2Seq(encoder=encoder, decoder=decoder)
        # Keep the original for of loss (batch_size * num_choices)
        model.loss_reduction = 'none'
        self.model = model
        # The tokenizer is used in the validation step
        tokenizer = AutoTokenizer.from_pretrained(args.encoder_name_or_path)
        self.tokenizer = tokenizer
        # The current setting only allows validation in downstream tasks
        self.mode = mode
        self.return_val_predictions = return_val_predictions
        # For validation
        self.evaluator = evaluate.load('accuracy')

    def batch_forward(self, batch):
        input_ids, attention_mask, decoder_labels, _, \
            node_ids, node_type_ids, adj_lengths, \
            edge_index, edge_type = batch

        # flatten the first two dimensions
        input_ids = self.squash_choice_dim(input_ids)
        attention_mask = self.squash_choice_dim(attention_mask)
        decoder_labels = self.squash_choice_dim(decoder_labels)
        node_ids = self.squash_choice_dim(node_ids)
        node_type_ids = self.squash_choice_dim(node_type_ids)
        adj_lengths = self.squash_choice_dim(adj_lengths)
        edge_index = [e for e_list in edge_index for e in e_list]
        edge_type = [e for e_list in edge_type for e in e_list]

        # for debugging
        assert attention_mask.shape == input_ids.shape
        # lm_input_ids as inputs, input_ids as labels, here they share the same attention mask
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            node_ids=node_ids,
            node_type_ids=node_type_ids,
            adj_lengths=adj_lengths,
            edge_index=edge_index,
            edge_type=edge_type,
            output_attentions=True,
            output_hidden_states=True,
            labels=decoder_labels,
            return_dict=True)

        batch_size, num_choices = batch[0].shape[:2]
        # reshape the loss to (batch_size, num_choices)
        losses = output.loss.view(batch_size, num_choices, -1)
        losses = torch.mean(losses, dim=-1)
        output.loss = losses
        return output

    def on_save_checkpoint(self, checkpoint):
        # After moving the embedding to the CPU, this weight should no longer exist
        if 'model.encoder.node_emb.emb.weight' in checkpoint['state_dict']:
            del checkpoint['state_dict']['model.encoder.node_emb.emb.weight']


class LitT5LMForMultipleChoice(LitT5ForMultipleChoice):
    """Pure T5 LM for multiple choice tasks. There is no GNN involved in this model."""
    def __init__(self, args, model):
        super().__init__()
        self.validation_step_outputs = []
        self.args = args
        self.save_hyperparameters({"learning_rate": args.learning_rate, "optimizer": args.optimizer})
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
        self.evaluator = evaluate.load('accuracy')
        self.learning_rate = args.learning_rate
        

    def batch_forward(self, batch):
        input_ids, attention_mask, decoder_labels, _, \
            _, _, _, \
            _, _ = batch

        # flatten the first two dimensions
        input_ids = self.squash_choice_dim(input_ids)
        attention_mask = self.squash_choice_dim(attention_mask)
        decoder_labels = self.squash_choice_dim(decoder_labels)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            labels=decoder_labels,
            return_dict=True)

        lm_logits = output.logits
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), decoder_labels.view(-1))
        output.loss = loss

        batch_size, num_choices = batch[0].shape[:2]
        # reshape the loss to (batch_size, num_choices)
        losses = output.loss.view(batch_size, num_choices, -1)
        losses = torch.mean(losses, dim=-1)
        output.loss = losses
        return output
