"""
This module contains util layers for T5GNN  model. It is copied from michiyasunaga/dragon/utils/layers.py (or INK-USC/SalKG/layers.py)
"""

import numpy as np
import torch
import torch.nn as nn


class CustomizedEmbedding(nn.Module):
    """This embedding layer keeps the embedding on CPU"""
    def __init__(self, node_num, node_in_dim, node_out_dim, use_contextualized=False,
                 pretrained_node_emb=None, freeze_ent_emb=True, scale=1.0, init_range=0.02):
        super().__init__()
        self.scale = scale
        self.use_contextualized = use_contextualized
        if not use_contextualized:
            # Wrapping embedding with brackets makes the model not aware of the layer
            self.emb = [nn.Embedding(node_num + 2, node_in_dim)]
            if pretrained_node_emb is not None:
                self.emb[0].weight.data.fill_(1)
                self.emb[0].weight.data[:node_num].copy_(pretrained_node_emb)
            else:
                self.emb[0].weight.data.normal_(mean=0.0, std=init_range)
            if freeze_ent_emb:
                for p in self.emb[0].parameters():
                    p.requires_grad = False

        if node_in_dim != node_out_dim:
            self.cpt_transform = nn.Linear(node_in_dim, node_out_dim)
            self.activation = nn.GELU()

    def forward(self, index, contextualized_emb=None):
        """
        index: size (bz, a)
        contextualized_emb: size (bz, b, emb_size) (optional)
        """
        if contextualized_emb is not None:
            assert index.size(0) == contextualized_emb.size(0)
            if hasattr(self, 'cpt_transform'):
                contextualized_emb = self.activation(self.cpt_transform(contextualized_emb * self.scale))
            else:
                contextualized_emb = contextualized_emb * self.scale
            emb_dim = contextualized_emb.size(-1)
            return contextualized_emb.gather(1, index.unsqueeze(-1).expand(-1, -1, emb_dim))
        else:
            if hasattr(self, 'cpt_transform'):
                return self.activation(self.cpt_transform(self.emb[0](index.cpu()).to(index.device) * self.scale))
            else:
                return self.emb[0](index.cpu()).to(index.device) * self.scale


class MLP(nn.Module):
    """
    Multi-layer perceptron

    Parameters
    ----------
    num_layers: number of hidden layers
    """
    activation_classes = {'gelu': nn.GELU, 'relu': nn.ReLU, 'tanh': nn.Tanh}

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, batch_norm=False,
                 init_last_layer_bias_to_zero=False, layer_norm=False, activation='gelu'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        assert not (self.batch_norm and self.layer_norm)

        self.layers = nn.Sequential()
        for i in range(self.num_layers + 1):
            n_in = self.input_size if i == 0 else self.hidden_size
            n_out = self.hidden_size if i < self.num_layers else self.output_size
            self.layers.add_module(f'{i}-Linear', nn.Linear(n_in, n_out))
            if i < self.num_layers:
                self.layers.add_module(f'{i}-Dropout', nn.Dropout(self.dropout))
                if self.batch_norm:
                    self.layers.add_module(f'{i}-BatchNorm1d', nn.BatchNorm1d(self.hidden_size))
                if self.layer_norm:
                    self.layers.add_module(f'{i}-LayerNorm', nn.LayerNorm(self.hidden_size))
                self.layers.add_module(f'{i}-{activation}', self.activation_classes[activation.lower()]())
        if init_last_layer_bias_to_zero:
            self.layers[-1].bias.data.fill_(0)

    def forward(self, input):
        return self.layers(input)


class MatrixVectorScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask=None):
        """
        q: tensor of shape (n*b, d_k)
        k: tensor of shape (n*b, l, d_k)
        v: tensor of shape (n*b, l, d_v)

        returns: tensor of shape (n*b, d_v), tensor of shape(n*b, l)
        """
        #V0
        # attn = ((q.float().unsqueeze(1) / self.temperature) * k.float()).sum(2)  # (n*b, l)

        #V1
        # attn = (q.float().unsqueeze(1) * (k.float() / self.temperature)).sum(2)  # (n*b, l)

        #V2
        # attn = (q.float().unsqueeze(1) * k.float()).sum(2)  # (n*b, l)
        # attn = attn / self.temperature

        #V3: seems to work the best (CSQA, OBQA)
        Qmax = torch.abs(q).max().detach().item()
        Kmax = torch.abs(k).max().detach().item()
        if Qmax > Kmax:
            attn = ((q.float().unsqueeze(1) / self.temperature) * k.float()).sum(2)  # (n*b, l)
        else:
            attn = (q.float().unsqueeze(1) * (k.float() / self.temperature)).sum(2)  # (n*b, l)

        #V4
        # Qmax = torch.abs(q).max().detach().item()
        # Kmax = torch.abs(k).max().detach().item()
        # if Qmax < 0.5 and Kmax < 0.5:
        #     attn = (q.float().unsqueeze(1) * k.float()).sum(2) / self.temperature # (n*b, l)
        # else:
        #     if Qmax > Kmax:
        #         attn = ((q.float().unsqueeze(1) / self.temperature) * k.float()).sum(2)  # (n*b, l)
        #     else:
        #         attn = (q.float().unsqueeze(1) * (k.float() / self.temperature)).sum(2)  # (n*b, l)

        # attn = attn.to(dtype=v.dtype)
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = (attn.unsqueeze(2) * v).sum(1)
        return output, attn


class MultiheadAttPoolLayer(nn.Module):

    def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1):
        super().__init__()
        assert d_k_original % n_head == 0  # make sure the outpute dimension equals to d_k_origin
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = nn.Linear(d_k_original, n_head * self.d_k)
        self.w_vs = nn.Linear(d_k_original, n_head * self.d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):
        """
        q: tensor of shape (b, d_q_original)
        k: tensor of shape (b, l, d_k_original)
        mask: tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, n*d_v)
        """
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        bs, _ = q.size()
        bs, len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
        ks = self.w_ks(k).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
        vs = self.w_vs(k).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1)
        output, attn = self.attention(qs, ks, vs, mask=mask)

        output = output.view(n_head, bs, d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
        output = self.dropout(output)
        return output, attn
