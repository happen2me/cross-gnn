"""
The gated cross attention is built from lucidrains and dhansmair's implementation.
The 'media' in the code refers to the other modality, it can be knowledge graph, passage
embedding, image etc.
"""
from typing import Tuple, Optional

import torch
from einops import rearrange
from einops_exts import rearrange_many
from torch import einsum, nn


def exists(val):
    return val is not None


def FeedForward(dim, mult = 4):
    """Feedforward layer with GELU activation."""
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )


# gated cross attention

class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim_q,
        dim_kv,
        dim_head = 64,
        heads = 8,
    ):
        """
        Args:
            dim_q (int): dimension of the input query embedding
            dim_kv (int): dimension of the input key, value embedding
            dim_head (int, optional): dimension of the q, k, v inside the attention head. Defaults to 64.
            heads (int, optional): number of attention heads. Defaults to 8.
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim_q)

        self.to_q = nn.Linear(dim_q, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_kv, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim_q, bias = False)

    def forward(
        self,
        q: torch.FloatTensor,
        kv: torch.FloatTensor,
        kv_mask: torch.BoolTensor,
        previous_kv: tuple = None,
        output_kv: bool = False
    ):
        """This has the same inputs as the GatedCrossAttentionBlock
        Args:
            q (FloatTensor):
                language features (n_batch, n_token, d_token)
            kv (FloatTensor, optional):
                media features, represents information from other modality, e.g. encoded by perceiver resample
                (n_batch, n_latents, d_media). Defaults to None.
            kv_mask (LongTensor | BoolTensor, optional):
                mask for key, value features (n_batch, n_latents). Defaults to None.
            previous_kv (tuple, optional):
                tuple of previous keys and values. Passed when caching is used during text generation.
                Defaults to None.
            output_kv (bool, optional):
                whether to return the keys and values. Defaults to False.
        Returns:
            FloatTensor: Tensor (n_batch, n_token, d_token)
        """
        h = self.heads

        q = self.norm(q)

        # d_inner = d_head * n_head
        # (batch, n_token, d_token) -> (batch, n_token, d_inner)
        q = self.to_q(q)
        q = q * self.scale

        if previous_kv is None:
            # (batch, n_latents, d_media) -> (batch, n_latents, d_inner) for k, v
            k, v = self.to_kv(kv).chunk(2, dim=-1)
            q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=h)
        else:
            # media can be ignored, k, v already computed
            k, v = previous_kv
            q = rearrange(q, 'b n (h d) -> b h n d', h=h)

        # compute the attention scores from the queries and keys
        sim = einsum('... i d, ... j d -> ... i j', q, k)  # (batch, n_heads, n_token, n_latents)

        if kv_mask is not None:
            # broadcast the mask in the head and token dimension
            kv_mask = rearrange(kv_mask, 'b n -> b 1 1 n')
            sim = sim.masked_fill(kv_mask.logical_not(), float('-inf'))

        # What is this for? For numerical stability?
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        # compute the context vectors from the attention scores and values
        out = einsum('... i j, ... j d -> ... i d', attn, v)  # (batch, n_token, d_inner)
        out = rearrange(out, 'b h n d -> b n (h d)')
        if output_kv:
            return self.to_out(out), (k, v)
        return self.to_out(out), None  # (batch, n_token, d_token)


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_media,
        dim_head = 64,
        heads = 8,
        ff_mult = 4
    ):
        """
        Args:
            dim (int): dimension of the input language token embedding
            dim_media (int): dimension of the input media token embedding
            dim_head (int, optional): dimension of the q, k, v inside the attention head. Defaults to 64.
            heads (int, optional): number of attention heads. Defaults to 8.
            ff_mult (int, optional): multiplier for the hidden dimension of the feedforward layer. Defaults to 4.
        """
        super().__init__()
        self.attn = MaskedCrossAttention(dim_q=dim, dim_kv=dim_media, dim_head=dim_head, heads=heads)
        self.attn_gate = nn.Parameter(torch.tensor([0.]))

        self.ff = FeedForward(dim, mult = ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.]))

    def forward(
        self,
        x: torch.FloatTensor,
        media: torch.FloatStorage,
        media_mask: torch.BoolTensor,
        previous_kv: tuple = None,
        output_kv: bool = False
    ):
        """
        Args:
            x (FloatTensor): language features (n_batch, n_token, d_token)
            media (FloatTensor, optional): media features, e.g. encoded by perceiver resample (n_batch, n_latents, d_media).
            media_mask (LongTensor | BoolTensor, optional): mask for media features (n_batch, n_latents).
        """
        if media is None:
            kv = None
        else:
            attn_out, kv = self.attn(x, media, media_mask, previous_kv=previous_kv, output_kv=output_kv)
            x = x + self.attn_gate.tanh() * attn_out
            x = x + self.ff_gate.tanh()* self.ff(x)
        return x, kv


class HijackedLMBlock(nn.Module):
    """
    A block that wraps a gated cross-attention layer, followed by a LM layer.
    We replace the original layers in the LM with these at a certain frequency
    to introduce the xattn layer. This layer mimics the functionality and behavior
    of the underlying LM block. This way, the LM can be used in the same way as before,
    and we can do the conditioning without altering the LM implementation.

    One drawback of this approach is that we cannot pass the visual features to forward()
    directly, but instead we need to pass them before the actual forward pass, via a
    side-channel, which is the condition() method. In addition, when use_cache is used,
    the cached keys and values for the xattn layers need to be retrieved separately from
    the kv_output property.
    """

    def __init__(self, lm_block, **kwargs):
        super().__init__()

        self.xattn_block = GatedCrossAttentionBlock(**kwargs)
        self.lm_block = lm_block
        self.media = None
        self.media_mask = None
        self.xattn_layer_past = None
        self.kv_output = None

    def condition(self, media: torch.Tensor, media_mask: torch.Tensor, xattn_layer_past=None) -> None:
        """
        conditioning. Called from outside of the LM before passing the text input to the LM.
        This way, the gated cross-attention layers get informed about the media input
        without the need to pipe the media input through the LM forward() function.

        xattn_layer_past can contain the cached cross-attention keys and values (computed
        from the media input). Passing them is useful to speed up the autoregressive text
        generation where the keys and values will be the same for every word, since the
        media input doesn't change.
        If both media and xattn_layer past are passed, media will be ignored in the xattn layers.
        """
        self.media = media
        self.media_mask = media_mask
        self.xattn_layer_past = xattn_layer_past

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.Tensor]],
        use_cache: Optional[bool] = False,
        **kwargs
    ):
        """
        This forward function mimics forward() of T5Block, so it has the same input and output.
        """

        # pass through xattn
        hidden_states, kv = self.xattn_block(
            x=hidden_states,
            media=self.media,
            media_mask=self.media_mask,
            previous_kv=self.xattn_layer_past,
            output_kv=use_cache
        )
        self.kv_output = kv

        # pass through original LM layer
        return self.lm_block(hidden_states, use_cache=use_cache, **kwargs)
