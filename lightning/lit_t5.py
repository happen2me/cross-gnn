"""This model finetunes the original T5 model without any extra graph information."""
import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import AdamW, RMSprop
from transformers import AutoTokenizer
from transformers.optimization import Adafactor
from deepspeed.ops.adam import DeepSpeedCPUAdam

from evaluation.squad import compute_score
from evaluation.bleu import compute_bleu

def evaluate(predictions, references):
    score = compute_score(predictions, references)
    bleu = compute_bleu(predictions, references)
    score.update(bleu)
    return score


class LitT5(pl.LightningModule):
    def __init__(self, args, model):
        """Initialize a T5Model lightning module.

        Warning: the decoder_start_token_id will be initialized as the pad_token_id of a
        tokenizer constructed from args.encoder_name_or_path tokenizer.

        Args:
            model (T5Model): a T5Model model.
        """
        super().__init__()
        self.validation_step_outputs = []
        self.args = args
        self.save_hyperparameters({"learning_rate": args.learning_rate, "optimizer": args.optimizer})
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
        self.evaluator = evaluate
        self.learning_rate = args.learning_rate

    def batch_forward(self, batch):
        input_ids, attention_mask, decoder_labels, \
            node_ids, node_type_ids, adj_lengths, \
            edge_index, edge_type = batch

        # lm_input_ids as inputs, input_ids as labels, here they share the same attention mask
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
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

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        self.log("learning_rate", self.learning_rate)

    def validation_step(self, batch, batch_idx):

        input_ids, attention_mask, answers, \
            node_ids, node_type_ids, adj_lengths, \
            edge_index, edge_type = batch

        tokenizer = self.tokenizer
        gold_answers = answers
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask)
        predictions = tokenizer.batch_decode(generated)
        predictions = [p.replace(tokenizer.pad_token, '').replace(tokenizer.eos_token, '').strip() for p in predictions]
        scores = self.evaluator(predictions, gold_answers)
        self.log_dict(scores)
        self.validation_step_outputs.append(scores)
        return scores

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
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

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end()

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
