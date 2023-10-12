import contextlib
import dataclasses
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from .gated_xattn import HijackedLMBlock


@contextlib.contextmanager
def suppress_model_loading_warnings(suppress: bool = True):
    if suppress:
        logger = logging.getLogger('transformers.modeling_utils')
        level = logger.level
        logger.setLevel(logging.CRITICAL)
        yield
        logger.setLevel(level)
    else:
        yield


class FlamingoConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `FlamingoT5` model."""
    d_model: int            # dimension of the token embedding
    dim_media: int          # dimension of the media embedding
    lm_name_or_path: str    # name or path of the pretrained LM model
    xattn_heads: int        # number of cross attention heads
    xattn_dim_head: int     # dimension of the cross attention heads
    xattn_every: int = 1    # how often to insert a gated cross attention layer
    xattn_ff_mult: int = 4  # multiplier for the feedforward layer in the gated cross attention


@dataclass
class FlamingoCausalLMOutputWithCrossAttentions(CausalLMOutputWithCrossAttentions):
    """Add cross attentions past key values to the base class."""
    xattn_past_key_values: Tuple[torch.FloatTensor] = None


class FlamingoDecoderBaseModel(ABC, PreTrainedModel):
    """
    abstract class, which is inherited by FlamingoT5.
    Different from dhansmair's implementation, this class assumes the lm model is a decoder
    with a lm head. The labels are processed and the loss is calculated in lm's forward() pass.
    This class provides the core functionalities of Flamingo: the forward() function,
    setting up the resampler and hijacking the LM layers with GatedXAttn layers.

    Modification upon dhansmair's implementation:
    - removed vision encoder
    - removed resampler
    - freeze token embedding of the language model
    - remove lm head
    - removed pixels
    - media and media_mask are nullable. If so, the cross attention on them will be skipped.
    """

    config: FlamingoConfig
    lm: PreTrainedModel

    config_class = FlamingoConfig

    def __init__(self, config: FlamingoConfig, suppress_warnings=True):
        assert isinstance(config, FlamingoConfig)
        super().__init__(config)

    def _init_layers(self, lm_layers: nn.ModuleList):
        """
        call during init of the subclass.
        careful, this method will modify the LM layers!
        """
        for i, lm_layer in enumerate(lm_layers):
            if i % self.config.xattn_every != 0:
                continue

            lm_layers[i] = HijackedLMBlock(
                lm_layer,
                dim=self.config.d_model,
                dim_media=self.config.dim_media,
                dim_head=self.config.xattn_dim_head,
                heads=self.config.xattn_heads,
                ff_mult=self.config.xattn_ff_mult,
            )

    @abstractmethod
    def get_modified_layers(self) -> List[HijackedLMBlock]:
        raise NotImplementedError

    def freeze_lm(self):
        """Freeze weights of the language model.
        It unfreeze modified layers and adapter layers.
        """
        for param in self.lm.parameters():
            param.requires_grad = False

        n_xattn = 0
        for xattn in self.get_modified_layers():
            for param in xattn.xattn_block.parameters():
                param.requires_grad = True
                n_xattn += param.numel()
        print(f'Number of unfrozen xattn parameters: {n_xattn}')

        n_adapter = 0
        for key, param in self.lm.named_parameters():
            if 'adapter' in key:
                param.requires_grad = True
                n_adapter += param.numel()
        print(f'Number of unfrozen adapter parameters: {n_adapter}')

    def unfreeze_lm(self):
        """Unfreeze all weights of the language model.
        """
        for param in self.lm.parameters():
            param.requires_grad = True

    def freeze_non_lm(self):
        """Freeze all weights except the language model.
        This is mainly to check whether the LM works as in its original implementation.
        """
        self.unfreeze_lm()
        for xattn in self.get_modified_layers():
            for param in xattn.xattn_block.parameters():
                param.requires_grad = False

    def state_dict_trainable(self) -> Dict[str, torch.Tensor]:
        """ include weights in the state dict if they have requires_grad = True"""

        trainable_param_names = [
            w for w, t in self.named_parameters() if t.requires_grad]
        return {k: v for k, v in self.state_dict().items() if k in trainable_param_names}

    def parameters_trainable(self):
        """Access the trainable parameters, e.g. useful for the optimizer and gradient clipping. 

        example: optimizer = AdamW(model.parameters_trainable(), lr=args.lr)
        make sure to call freeze_lm() first!
        """
        return filter(lambda p: p.requires_grad, self.parameters())

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        media: Optional[torch.Tensor] = None,
        media_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple] = None,
        xattn_past_key_values: Optional[tuple] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        return_dict: bool = True,
        **kwargs
    ) -> FlamingoCausalLMOutputWithCrossAttentions:
        """Flamingo forward pass

        The forward pass is closely coupled to the T5ForCaualLM's forward pass (implemented within this repo).

        Args:
            input_ids (Tensor | None):         shape (n_batch, n_tokens). the tokenized input text
            attention_mask (Tensor | None):    shape (n_batch, n_tokens). 
                Mask as produced by the tokenizer. Required when a batch of input strings are tokenized and thus padded at the end.
                Then this will indicate the locations of 'real' tokens vs. the location of 'pad' tokens.
            inputs_embeds (Tensor | None):   shape (n_batch, seq_len, hidden_size)
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
                is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
                model's internal embedding lookup matrix.
            media (Tensor | None):         shape (b N q d). Optional.
                If pixel_values already have been passed through encode_resample_visuals(), 
                you can pass the resampled visual embeddings via this parameter.
                If provided, pixel_values will be ignored
            media_mask (Tensor | None):   shape (n_batch, n_latents).
                Mask for the media tokens. If None, all media tokens will be regarded valid.
            head_mask (Tensor | None):  shape (n_heads,) or (num_layers, n_heads)
                Mask to nullify selected heads of the self-attention modules in the encoder.
            past_key_values (tuple):
                past_key_values of the language model
            xattn_past_key_values (tuple):
                past_key_values of the cross attention on media
            labels (Tensor):
                It is possible to pass the exact value as input_ids also as labels. If present, the output will contain a CE loss of the next token prediction.
                optional, defaults to None
            use_cache (bool): whether to return the inner keys and values. Used to speed up text generation at inference. defaults to False
            return_dict (bool): Whether to return a dictionary. Right now, only dicts are supported, so this must be set to True. Defaults to True.
            **kwargs

        Returns:
            (FlamingoCausalLMOutputWithCrossAttentions): an object containing all the useful stuff. Refer to hf documentation.
        """

        # sanity check
        assert return_dict, "can only use return_dict=True at the moment!"
        assert (input_ids is None) != (inputs_embeds is None), "you must pass either input_ids or inputs_embeds!"

        if media_mask is None and media is not None:
            # media is of shape (batch_size, n_latents, d_media)
            batch_size, n_latents = media.shape[:2]
            media_mask = torch.ones(size=(batch_size, n_latents), dtype=torch.long, device=media.device)

        # condition xattn layers
        for i, xattn in enumerate(self.get_modified_layers()):
            layer_past = None if xattn_past_key_values is None else xattn_past_key_values[i]
            xattn.condition(media, media_mask, layer_past)

        # pass through LM
        out: CausalLMOutputWithCrossAttentions = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            labels=labels,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=True,
            **kwargs
        )

        # collect the past_key_values from the xattn layers
        if use_cache:
            xattn_past_key_values = []
            for modified_layer in self.get_modified_layers():
                xattn_past_key_values.append(modified_layer.kv_output)

        # failed: out = FlamingoCausalLMOutputWithCrossAttentions(**dataclasses.asdict(out))
        # according to python issue43905, dataclasses.asdict() will inevitably deepcopy the data,
        # this is however not supported by optimizer.backward().
        # the following is a workaround
        out_with_xattn = FlamingoCausalLMOutputWithCrossAttentions()
        for field in dataclasses.fields(out):
            key = field.name
            val = getattr(out, key)
            setattr(out_with_xattn, key, val)
        out_with_xattn.xattn_past_key_values = tuple(xattn_past_key_values) if use_cache else None

        return out_with_xattn
