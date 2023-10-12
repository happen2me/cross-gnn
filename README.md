# CrossGNN: Graph-Enhanced Cross-Modal Transformer for Text Generation

CrossGNN enables a transformer model to comprehend graph modality by integrating graph-specific layers to the language model.

## Dependencies
`deepspeed` does not support gxx later than 10. Installing `gxx_linux-64=9.3.0` in advance avoids reconfiguring the whole environment.
```bash
conda install gxx_linux-64=9.3.0
pip install -r requirements.txt
conda install pyg pytorch-scatter -c pyg
```

## Architecture

Upon an encoder-decoder transformer-architectured language backbone:

- On the encoder side, an extra graph neural network (GNN) is used to contextualize the graph nodes and edges with the language hidden states. 
- On the decoder side, the GNN hidden states are attended with inserted cross attention layers to provide knowledge for language decoding.

Two concurrent data streams exist:

- A T5 model handles language encoding and decoding, remaining unchanged during training.
- The graph first encoded by a GNN model, influenced by language hidden states, then attended and merged into the language decoder.


These streams intersect through two cross-attentions:

- During encoding, the graph's context node attends to language hidden states, allowing a one-way flow from language to graph.
- In decoding, language hidden states cross attend the graph hidden states, drawing knowledge from the graph back to the language.

**TL;DR**:

1. Language Encoder --inform--> Graph Encoder (❄)
2. Encoded Graph --condition--> Language Decoder (❄)

## Usage

Specify the configurations for model, data, and training in a yaml file first, following examples in `configs/`. Then run the following commands. Then assign the config file path and profile to training or evaluation scripts. Check their respective help messages for more details.

- Training
```bash
python train.py --help
```

- Evaluation
```bash
python eval.py --help
```

## Acknowledgement

The model is built upon [dragon](https://github.com/michiyasunaga/dragon), [flamingo](https://arxiv.org/abs/2204.14198), and a [pytorch implementation](https://github.com/dhansmair/flamingo-mini) of flamingo.
