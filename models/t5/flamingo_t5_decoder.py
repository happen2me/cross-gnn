"""
This file patches the flamingo gated cross attention into the T5 decoder.
"""
from typing import List

from ..gated_xattn import HijackedLMBlock
from ..flamingo import FlamingoDecoderBaseModel, FlamingoConfig
from .t5_decoder import T5ForCausalLM


class FlamingoT5Decoder(FlamingoDecoderBaseModel):
    """T5 decoder model patched with Flamingo cross attention layers"""
    config: FlamingoConfig
    _keys_to_ignore_on_load_unexpected = ["encoder*"]

    def __init__(self, config: FlamingoConfig, embed_tokens, **kwargs):
        super().__init__(config, **kwargs)
        assert config.lm_name_or_path is not None and "t5" in config.lm_name_or_path, f"Unexpected lm {config.lm_name_or_path}"
        self.lm = T5ForCausalLM.from_pretrained(config.lm_name_or_path, embed_tokens=embed_tokens)
        assert self.lm.config.d_model == config.d_model, \
            f"LM and Flamingo model must have the same token embedding dimension. Got {self.lm.config.d_model} and {config.d_model}"
        # update config with only Flamingo-specific parameters
        self.lm.config.update(config.to_diff_dict())
        self.config = self.lm.config
        # hijack lm layers (t5 stack: list[T5Block])
        self._init_layers(self.lm.decoder.block)

    def get_modified_layers(self) -> List[HijackedLMBlock]:
        if self.config.xattn_every == 1:
            return self.lm.decoder.block
        return filter(lambda layer: isinstance(layer, HijackedLMBlock), self.lm.decoder.block)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.lm.prepare_inputs_for_generation(*args, **kwargs)

    def _reorder_cache(self, past, beam_idx):
        return self.lm._reorder_cache(past, beam_idx)
