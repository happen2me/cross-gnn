"""
In the transformers T5 implementation, there is not a standalone T5 decoder model.
However, we only want to patch the Flamingo architecture into the decoder. Therefore,
I manually create a T5 decoder model from the transformers' implementation. It is
based on the T5ForConditionalGeneration and BartForCausalLM, but I removed the encoder.
"""
import copy
import logging
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.t5.modeling_t5 import T5PreTrainedModel, T5Stack
from transformers.adapters.mixins.t5 import T5ModelAdaptersMixin, T5ModelWithHeadsAdaptersMixin
from transformers.adapters.context import ForwardContext

logger = logging.getLogger(__name__)


class T5ForCausalLM(T5ModelWithHeadsAdaptersMixin, T5ModelAdaptersMixin, T5PreTrainedModel):
    """Adapted from BartForCausalLM and T5ForConditionalGeneration v4.26.1"""
    _keys_to_ignore_on_load_missing = [
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]

    def __init__(self, config, embed_tokens):
        """
        config: T5Config
        shared_embedding: encoder's embedding
        """
        super().__init__(config)
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        self.shared = embed_tokens
        self.decoder = T5Stack(config, embed_tokens)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.config = config
        self.model_dim = config.d_model

        self._init_adapter_modules()

        self.post_init()
        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def get_output_embeddings(self):
        return self.lm_head

    @ForwardContext.wrap
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,):

        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        else:
            loss = None

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        """Copied from T5ForConditionalGeneration.prepare_inputs_for_generation() v4.26.1"""
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            **kwargs,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        """Copied from T5ForConditionalGeneration._shift_right() v4.26.1"""
        return self._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        """Copied from T5ForConditionalGeneration._reorder_cache() v4.26.1"""
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning(
                "You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(
                        0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + \
                (reordered_layer_past_states,)
        return reordered_decoder_past

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # init weights and load from pretrained
        model = super(cls, T5ForCausalLM).from_pretrained(*args, **kwargs)
        # make sure that the model is correctly initialized
        assert model.config.is_decoder, "If you want to use `T5ForCausalLM` make sure that `config.is_decoder=True` for "
        # Patch: we noticed that the lm_head is not correctly initialized, therefore,
        # we mannually initialize a T5ForConditionalGeneration model and copy the weights
        if 'embed_tokens' in kwargs:
            del kwargs['embed_tokens']
        t5_generation_model = T5ForConditionalGeneration.from_pretrained(*args, **kwargs)
        with torch.no_grad():
            model.lm_head.weight.copy_(t5_generation_model.lm_head.weight)
        return model
