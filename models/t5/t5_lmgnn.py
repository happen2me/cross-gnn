"""
This is built on top of LMGNN and DRAGON model.
This module patches GNN into the encoder of T5 model.
"""
import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5.modeling_t5 import T5EncoderModel

from ..gnn import GATConvE,  make_one_hot
from ..gated_xattn import MaskedCrossAttention
from utils.layers import MLP, CustomizedEmbedding

logger = logging.getLogger(__name__)


@dataclass
class T5GNNEncoderOutput(BaseModelOutputWithPastAndCrossAttentions):
    gnn_hidden_states: Optional[torch.FloatTensor] = None


class T5GNNConfig(PretrainedConfig):
    encoder_name_or_path: str
    gnn_dim: int            # dimension of the GNN layers, is gnn_dim in the config file
    ie_dim: int             # hidden dim of information exchange layers
    node_in_dim: int        # node embedding dim of the input node embedding, i.e. wikidata5m is 512
    n_ntype: int            # number of node types
    n_etype: int            # number of edge types
    num_ie_layer: int       # number of information exchange hidden layers, ie_layer_num in the config file
    num_gnn_layers: int
    num_entity: int         # total number of enitities in the entity embedding
    dropout_gnn: float      # node feature dropout rate, is dropoutg in the config file
    dropout_emb: float      # dropout for the embedding layer, is dropouti in the config file


class T5GNNEncoder(PreTrainedModel):
    """We can acquire all intermediate hidden states without hijack the language model,
    by using the returned all_hidden_states.
    """
    def __init__(self, config: T5GNNConfig, pretrained_node_emb):
        super().__init__(config)
        # Load language model
        self.lm = T5EncoderModel.from_pretrained(config.encoder_name_or_path)
        if config.num_gnn_layers > self.lm.config.num_layers:
            raise ValueError(f"num_gnn_layers {config.num_gnn_layers} should be no greater than"
                             f"language encoder layers {self.lm.config.num_layers}")

        # T5GNN
        self.t5gnn_activation = nn.GELU()
        # the pretrained node embedding is kept on the CPU with this customized embedding layer
        self.node_emb = CustomizedEmbedding(node_num=config.num_entity, node_out_dim=config.gnn_dim,
                                            use_contextualized=False, node_in_dim=config.node_in_dim,
                                            pretrained_node_emb=pretrained_node_emb,
                                            freeze_ent_emb=True)
        self.dropout_embed = nn.Dropout(config.dropout_emb)

        # Texe message passing
        self.emb_node_type = nn.Linear(config.n_ntype, config.gnn_dim // 2)
        self.Vh = nn.Linear(config.gnn_dim, config.gnn_dim)
        self.Vx = nn.Linear(config.gnn_dim, config.gnn_dim)
        self.activation_node_type = nn.GELU()
        self.activation_residual = nn.Sequential(
            nn.GELU(),
            nn.Dropout(config.dropout_gnn)
        )
        # shared edge encoder
        edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(config.n_etype + 1 + config.n_ntype * 2, config.gnn_dim),
            torch.nn.BatchNorm1d(config.gnn_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(config.gnn_dim, config.gnn_dim))
        self.gnn_blocks = nn.ModuleList(
            [GNNBlock(edge_encoder,
                      config.n_ntype,
                      config.n_etype,
                      sent_dim=self.lm.config.d_model,
                      node_dim=config.gnn_dim,
                      ie_dim=config.ie_dim,
                      ie_layers=config.num_ie_layer)
             for _ in range(config.num_gnn_layers)])

        # T5GAT
        self.activation_gat = nn.Sequential(
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # config is updated with lm's config
        self.lm.config.update(config.to_diff_dict())
        config = self.lm.config
        self.config = config

    def batch_graph(self, edge_index_init, edge_type_init,  n_nodes):
        """
        edge_index_init:  list of (n_examples, ). each entry is torch.tensor(2, E?)    ==> [2, total_E]
        edge_type_init:   list of (n_examples, ). each entry is torch.tensor(E?, )     ==> [total_E, ]
        """
        n_examples = len(edge_index_init)
        edge_index = [edge_index_init[i] + i * n_nodes for i in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1) #[2, total_E]
        edge_type = torch.cat(edge_type_init, dim=0) #[total_E, ]
        return edge_index, edge_type

    def forward(
        self,
        node_ids,
        node_type_ids,
        adj_lengths,
        edge_index,
        edge_type,
        **kwargs
        ):
        """
        Args:
            node_ids: (batch_size, n_nodes)
            node_type_ids: (batch_size, n_nodes)
            adj_lengths: (batch_size, ) means the "actual" number of nodes
            adj -> edge_index, edge_type
                edge_index: list of (batch_size, );
                    each entry is torch.tensor(2, E(variable)) -> (2, total E)
                edge_type: list of (batch_size, );
                    each entry is torch.tensor(E(variable), ) -> (total E, )
            kwargs: other inputs for language model
        """
        # save the original config
        output_hidden_states = kwargs.get('output_hidden_states', False)
        return_dict = kwargs.get('return_dict', False)
        # capture all intermediate hidden states
        kwargs['output_hidden_states'] = True
        kwargs['return_dict'] = True
        lm_outputs: BaseModelOutputWithPastAndCrossAttentions = self.lm(**kwargs)
        lm_all_hidden_states = lm_outputs.hidden_states  # tuple of #layers of hidden states

        # Originally in T5DragonEncoder
        edge_index, edge_type = self.batch_graph(edge_index, edge_type, node_ids.size(1))
        adj = (edge_index, edge_type)

        # Originally in T5GNN
        # Embed node
        node_ids[node_ids == 0] = self.config.num_entity + 2
        gnn_input = self.node_emb(node_ids - 1)
        gnn_input[:, 0] = 0
        gnn_input = self.dropout_embed(gnn_input)

        # Originally in TextMessagePassing
        # embed type
        batch_size, n_nodes = node_type_ids.shape[:2]
        T = make_one_hot(node_type_ids.view(-1).contiguous(), self.config.n_ntype).view(batch_size, n_nodes, self.config.n_ntype)
        node_type_emb = self.activation_node_type(self.emb_node_type(T)) #[batch_size, n_node, dim/2]

        X = gnn_input
        edge_index, edge_type = adj #edge_index: [2, total_E]   edge_type: [total_E, ]  where total_E is for the batched graph
        node_type_flatten = node_type_ids.view(-1).contiguous() #[`total_n_nodes`, ]
        node_feature_extra = torch.cat([node_type_emb, node_type_emb], dim=2).view(node_type_flatten.size(0), -1).contiguous() #[`total_n_nodes`, dim]

        # Design choice: use lask K hidden_states to query GNN
        # or to use every x hidden_states to query GNN
        # Currently, I chose the former. As deeper LM features contain more semantic information,
        # while the shallower LM features contain more linguistic information
        lm_attention_mask = kwargs.get('attention_mask', None)
        for gnn_block, lm_hidden_states in zip(self.gnn_blocks, lm_all_hidden_states[-len(self.gnn_blocks):]):
            X = gnn_block(
                X=X,
                hidden_states=lm_hidden_states,
                hidden_states_mask=lm_attention_mask,
                edge_index=edge_index,
                edge_type=edge_type,
                node_type=node_type_flatten,
                node_feature_extra=node_feature_extra
            )
        gnn_output = self.activation_gat(X)
        
        # Originally in TextMessagePassing
        gnn_output = self.activation_residual(self.Vh(gnn_input) + self.Vx(gnn_output))
        
        # Originally in T5GNN
        node_mask = torch.arange(node_type_ids.size(1), device=node_type_ids.device) >= adj_lengths.unsqueeze(1) #[bs, nodes] 1 means masked out
        gnn_output = gnn_output * (~node_mask).float().unsqueeze(2)

        if not return_dict:
            outputs = (lm_outputs.last_hidden_state,)
            if output_hidden_states:
                outputs = outputs + (lm_outputs.hidden_states,)
            if kwargs.get('output_attentions', False):
                outputs = outputs + (lm_outputs.attentions,)
            outputs += (gnn_output,)
            return outputs
        outputs = T5GNNEncoderOutput(
            last_hidden_state=lm_outputs.last_hidden_state,
            hidden_states=lm_outputs.hidden_states,
            attentions=lm_outputs.attentions,
            gnn_hidden_states=gnn_output
        )
        return outputs

    def get_input_embeddings(self):
        return self.lm.get_input_embeddings()

    def freeze_lm(self):
        for p in self.lm.parameters():
            p.requires_grad = False

    def freeze_non_lm(self):
        """Freeze all parameters except the language model."""
        for p in self.parameters():
            p.requires_grad = False
        for p in self.lm.parameters():
            p.requires_grad = True


class GNNBlock(nn.Module):
    """Layerwise interaction with language hidden states.
    """
    def __init__(self, edge_encoder, n_ntype, n_etype, sent_dim, node_dim, ie_dim, ie_layers, dropout=0.2):
        """
        sent_dim: should bet set as config.hidden_size
        ie_layer: there is a design choice, ie_layer can be shared, or exists one in every layers,
            if create inside, then ie_dim (information exchange dim) is required.
            To maximize the learning ability, and also because of the changing lm representation,
            I use different info_exchange layer in every gnn block
        """
        super().__init__()
        self.gnn_layer = GATConvE(node_dim, n_ntype, n_etype, edge_encoder)
        self.activation = nn.GELU()
        self.node_dropout = nn.Dropout(dropout)
        # information exchange layer
        self.lm2graph_layer = MLP(node_dim * 2, ie_dim, node_dim, ie_layers, 0.1)
        self.lm_pooler = MaskedCrossAttention(dim_q=node_dim, dim_kv=sent_dim)

    def forward(
        self,
        X,
        hidden_states,
        hidden_states_mask,
        edge_index=None,
        edge_type=None,
        node_type=None,
        node_feature_extra=None,):
        """
        hidden_states: [bs, seq_len, sent_dim]
        X: [batch_size, n_node, d_node]
        edge_index: [2, n_edges]
        edge_type: [n_edges]
        _node_type: [bs * n_nodes]
        _node_feature_extra: [bs * n_nodes, node_dim]
        """
        batch_size = hidden_states.size(0)
        node_dim = X.size(-1)
        # X_flatten: [total_n_nodes, node_dim] where `total_n_nodes` = b_size * n_node
        X_flatten = X.view(-1, node_dim)
        X_flatten = self.gnn_layer(X_flatten, edge_index, edge_type, node_type, node_feature_extra)
        X_flatten = self.activation(X_flatten)
        X_flatten = self.node_dropout(X_flatten)

        # Propagate info from LM to GNN hidden states (Modality interaction)
        X = X_flatten.view(batch_size, -1, node_dim) # [bs, max_num_nodes, node_dim]
        # Implement better pooling over the sentence embedding
        # Design choices:
        # - use max | mean pooling
        # - use multihead self attention as pooling
        # ✓ use multihead cross attention pooling (where the query is the gnn context node)
        context_node_gnn_feats = X[:, 0, :].clone() # [bs, node_dim]
        context_node_lm_feats, _ = self.lm_pooler(
            q=context_node_gnn_feats.unsqueeze(1), # [bs, 1, node_dim]
            kv=hidden_states, # [bs, seq_len, sent_dim]
            kv_mask=hidden_states_mask, # [bs, seq_len]
        )   # [bs, 1, node_dim]
        context_node_lm_feats = context_node_lm_feats.squeeze(1) # [bs, node_dim]
        context_node_feats = torch.cat([context_node_lm_feats, context_node_gnn_feats], dim=1)
        context_node_feats = self.lm2graph_layer(context_node_feats)
        # residual link
        context_node_feats = context_node_feats + context_node_gnn_feats
        X[:, 0, :] = context_node_feats
        return X
