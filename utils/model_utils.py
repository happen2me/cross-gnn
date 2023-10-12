import torch
import numpy as np

from models.t5 import T5GNNConfig, T5GNNEncoder


# Utils for the encoder
def get_tweaked_num_relations(num_relations, cxt_node_connects_all):
    """
    Args:
        num_relations: the number of relations in the dataset, i.e. len(id2relation)
        ctx_node_connects_all: whether every other node is connected by a ctx node
    """
    # cxt2qlinked_rel and cxt2alinked_rel
    tweaked_num_relations = num_relations + 2
    if cxt_node_connects_all:
        tweaked_num_relations += 1
    return tweaked_num_relations


def construct_encoder(args):
    """
    num_relation: the number of relations in the original KG: len(id2relation)
    final_num_relation: the number of relations in the final KG, e.g. len(id2relation) + 2
    model_type: 'dragon' or 'dragon_encoder'. 'dragon' is the model for pretraining;
        'dragon_encoder' is the model for downstream tasks, it is compatible with the
        EncoderDecoderModel in transformers.
    """

    num_relation = args.num_relations
    final_num_relation = get_tweaked_num_relations(
        num_relation, args.cxt_node_connects_all)

    # Load pretrained entity embeddings
    entity_emb = np.load(args.ent_emb_paths)
    print(f"Entity embedding (shape: {entity_emb.shape}) loaded from {args.ent_emb_paths}.")
    entity_emb = torch.tensor(entity_emb, dtype=torch.float)
    enity_num, entity_in_dim = entity_emb.size(0), entity_emb.size(1)
    print(f"| num_entities: {enity_num} |")

    n_ntype = 4
    n_etype = final_num_relation * 2
    print(f"| final_num_relation: {final_num_relation}, len(id2relation): {num_relation} |")
    print(f"| n_ntype: {n_ntype}, n_etype {n_etype} |")

    config = T5GNNConfig(
        encoder_name_or_path=args.encoder_name_or_path,
        gnn_dim=args.gnn_dim,
        ie_dim=args.ie_dim,
        node_in_dim=entity_in_dim,
        n_ntype=n_ntype,
        n_etype=n_etype,
        num_ie_layer=args.ie_layer_num,
        num_gnn_layers=args.k,
        num_entity=enity_num,
        dropout_gnn=args.dropoutg,
        dropout_emb=args.dropoutg,
    )

    model = T5GNNEncoder(config, pretrained_node_emb=entity_emb)
    return model


def sep_params(model, loading_info, prefix=""):
    """Separate the parameters into loaded and not loaded.

    Usage:
        model = construct_model(args)
        loading_info = model.loading_info
        prefix = "lmgnn."
        loaded_params, not_loaded_params = sep_params(model, loading_info, prefix)

    Returns:
        loaded_params: the parameters that are loaded from the pretrained model
            it is also params_to_freeze, small_lr_params
        not_loaded_params: the parameters that are not loaded from the pretrained model
            it is also large_lr_params
    """
    # WHY is this needed: the keys in the current model have a LMGNN prefix, but the keys
    # in the loaded model do not have the prefix. So we need to add the prefix to the keys 
    # Discriminate the parameters into different groups
    # "all_keys" is already removed from Model.from_pretrained
    # loaded_roberta_keys = [prefix + k for k in loading_info["all_keys"]]
    missing_keys = [prefix + k for k in loading_info["missing_keys"]]

    loaded_params = {}
    not_loaded_params = {}
    for n, p in model.named_parameters():
        if n in missing_keys:
            not_loaded_params[n] = p
        else:
            loaded_params[n] = p
    return loaded_params, not_loaded_params
