import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import AdamW, RMSprop
from transformers import AutoTokenizer
from transformers.optimization import Adafactor
from deepspeed.ops.adam import DeepSpeedCPUAdam

from models.t5 import T5Seq2Seq
from evaluation.squad import compute_score
from evaluation.bleu import compute_bleu


def evaluate(predictions, references):
    score = compute_score(predictions, references)
    bleu = compute_bleu(predictions, references)
    score.update(bleu)
    return score


class LitT5Seq2Seq(pl.LightningModule):
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
        self.model = model
        self.learning_rate = args.learning_rate
        # The tokenizer is used in the validation step
        tokenizer = AutoTokenizer.from_pretrained(args.encoder_name_or_path)
        self.tokenizer = tokenizer
        self.evaluator = evaluate
        # The current setting only allows validation in downstream tasks
        self.mode = mode
        self.return_val_predictions = return_val_predictions

    def batch_forward(self, batch):
        input_ids, attention_mask, decoder_labels, \
            node_ids, node_type_ids, adj_lengths, \
            edge_index, edge_type = batch
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
        return output

    def training_step(self, batch, batch_idx):
        output = self.batch_forward(batch)
        loss = output.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.mode == 'pretrain':
            output = self.batch_forward(batch)
            loss = output.loss
            perplexity = torch.exp(loss)
            self.log('perplexity', perplexity)
            return {'perplexity': perplexity}

        input_ids, attention_mask, answers, \
            node_ids, node_type_ids, adj_lengths, \
            edge_index, edge_type = batch

        tokenizer = self.tokenizer
        gold_answers = answers
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                node_ids=node_ids,
                node_type_ids=node_type_ids,
                adj_lengths=adj_lengths,
                edge_index=edge_index,
                edge_type=edge_type,)
        predictions = tokenizer.batch_decode(generated)
        predictions = [p.replace(tokenizer.pad_token, '').replace(tokenizer.eos_token, '').strip() for p in predictions]
        scores = self.evaluator(predictions, gold_answers)
        self.log_dict(scores)
        if self.return_val_predictions:
            return {
                "predictions": predictions,
                "references": gold_answers,
                **scores
            }
        self.validation_step_outputs.append(scores)
        return scores

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if self.mode == 'pretrain':
            if len(outputs) > 0 and "perplexity" in outputs[0]:
                perplexity = np.mean([o["perplexity"] for o in outputs])
                self.log('val_perplexity', perplexity, on_epoch=True)
                return {'val_perplexity': perplexity}
            return {}
        if len(outputs) > 0:
            mean_keys = outputs[0].keys()
            mean_keys = set(mean_keys) - set(["predictions", "references"])
            scores = {k: np.mean([o[k] for o in outputs]) for k in mean_keys}
            if "predictions" in outputs[0]:
                # concatenate all predictions
                scores["predictions"] = [p for o in outputs for p in o["predictions"]]
            if "references" in outputs[0]:
                # concatenate all references
                scores["references"] = [p for o in outputs for p in o["references"]]
        else:
            scores = {}
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

    def on_save_checkpoint(self, checkpoint):
        # After moving the embedding to the CPU, this weight should no longer exist
        if 'model.encoder.node_emb.emb.weight' in checkpoint['state_dict']:
            del checkpoint['state_dict']['model.encoder.node_emb.emb.weight']
