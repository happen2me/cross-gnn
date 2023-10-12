"""
LMGNN, but with choices as inputs

train dataset:
    lm inputs: input_ids, attention_mask, decoder_labels
    graph inputs: node_ids, node_type_ids, adj_lengths, edge_index, edge_type

validation dataset:
    lm inputs: input_ids, attention_mask, decoder_labels
    graph inputs: node_ids, node_type_ids, adj_lengths, edge_index, edge_type

test dataset:
    lm inputs: input_ids, attention_mask, answers(a list of strings)
    graph inputs: node_ids, node_type_ids, adj_lengths, edge_index, edge_type

In case of multiple choices, we read key "answers" as choices from the statements,
and use "label" to indicate the index of the correct answer. Besides, each __getitem__
will return K examples, where K is the number of choices.
"""
import json
import math
import os
import pickle
from abc import ABC, abstractmethod
from collections import namedtuple
from functools import lru_cache
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, T5Tokenizer
from transformers.data import DataCollatorForSeq2Seq
from transformers import BatchEncoding, PreTrainedTokenizerBase

from utils.model_utils import get_tweaked_num_relations

InputExample = namedtuple('InputExample', 'example_id contexts question endings label')
InputFeatures = namedtuple('InputFeatures', 'input_ids attention_mask decoder_labels choice_label node_ids node_type_ids adj_length edge_index edge_type')


class LMGNNChoiceDataset(Dataset):
    """This dataset additionally outputs decoder inputs on top of DragonDataset with 
    dragond_enc_collate_fn's output. It removes the annoying choices dimension. Besides,
    it doesn't corrupt the graph and the text for the link prediction task and masked
    language modeling task.
    """
    def __init__(self, statement_path, num_relations, adj_path, max_seq_length=256,
                 model_name='t5-base', max_node_num=200, cxt_node_connects_all=True,
                 kg_only_use_qa_nodes=False, truncation_side='right',
                 encoder_input='question', decoder_label='answer', prefix_ratio=0.2,
                 num_choices=4, has_choice_graph=False):
        """
        Valid pairs of encoder_input and decoder_label:
        - Pretraining (Denoise): encoder_input = 'context', decoder_label = 'context'
        - Pretraining (PrefixLM): encoder_input = 'context_prefix', decoder_label = 'context_suffix'
        - Finetuning: encoder_input = 'question' | 'retrieval_augmented_question' | 'contextualized_question',
          decoder_label = 'answer' | 'choices'
        - Test: encoder_input = 'question' | 'retrieval_augmented_question',
          decoder_label = 'raw_answers' | 'choices'
    
        Args:
            dataset_name: the name of the dataset. Currently only 'squad_v2' is supported
            adj_path: the path to a monilithic adj pickle (legacy) or path to a folder containing adj pickle files
            legacy_mode: if True, use the monolithic adj pickle file, else the adj_path should be a folder
            encoder_input: the input to the encoder. Can be 'question', 'context', or 'contextualized_question'
            has_choice_graph: if True, each choice will have its own graph. Otherwise, the graph for the question is
                duplicated for each choice
        """
        assert encoder_input in ['question', 'context', 'context_prefix', 'retrieval_augmented_question', 'contextualized_question']
        assert decoder_label  in ['answer', 'context', 'context_suffix', 'raw_answers', 'choices']
        if encoder_input == 'context_prefix' or decoder_label == 'context_suffix':
            assert encoder_input == 'context_prefix' and decoder_label == 'context_suffix', \
                "'context_prefix' and 'context_suffix' must be used together, " \
                f"got {encoder_input} and {decoder_label}"
        assert os.path.isdir(adj_path), "adj_path should be a folder in non-legacy mode"
        super(Dataset).__init__()
        # For text data
        self.max_seq_length = max_seq_length
        # truncation_side is only available in newer versions of transformers
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, truncation_side=truncation_side)
        # Read statements
        self.examples = self.read_statements(statement_path)
        if len(self.examples) == 0:
            raise ValueError("No examples found in the dataset")

        self.encoder_input = encoder_input
        self.decoder_label = decoder_label

        # For graph data
        self.adj_path = adj_path
        self.cxt_node_connects_all = cxt_node_connects_all
        self.num_relations = num_relations
        self.max_node_num = max_node_num
        self.kg_only_use_qa_nodes = kg_only_use_qa_nodes
        self.num_choices = num_choices
        self.has_choice_graph = has_choice_graph

        # For retrieval
        if encoder_input == 'retrieval_augmented_question':
            raise NotImplementedError("Retrieval augmentation is not implemented yet")

        # For prefix completion
        self.prefix_ratio = prefix_ratio

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # 1. Load text data
        # self.train_qids, self.train_labels, self.train_encoder_data, train_nodes_by_sents_list
        example = self.examples[idx]
        # MLM is also done in postprocess_text
        encoder_inputs, decoder_labels = self.postprocess_text(example)
        input_ids, attention_mask = encoder_inputs["input_ids"], encoder_inputs["attention_mask"]
        choice_label = example.label

        # 2. Load graph data, post processing is done in the _load_graph_from_index function
        node_ids, node_type_ids, adj_length, edge_index, \
            edge_type = self._load_graph_from_index(idx)

        return InputFeatures(input_ids, attention_mask, decoder_labels, choice_label,
            node_ids, node_type_ids, adj_length,
            edge_index, edge_type)

    def _load_graph_from_index(self, idx):
        """adapted from load_sparse_adj_data_with_contextnode in utils/data_utils.py L653"""
        example = self.examples[idx]
        graph_id = example.example_id
        if self.has_choice_graph:
            decoder_data_list = []
            adj_data_list = []
            for i in range(self.num_choices):
                choice_graph_id = graph_id + f"_{i}"
                graph = self.load_graph(choice_graph_id)
                *decoder_data, adj_data = self.preprocess_graph(graph)
                adj_data_list.append(adj_data)
                decoder_data_list.append(decoder_data)
        else:
            graph = self.load_graph(graph_id)
            *decoder_data, adj_data = self.preprocess_graph(graph)
            decoder_data_list = [decoder_data] * self.num_choices
            adj_data_list = [adj_data] * self.num_choices
        node_ids, node_type_ids, node_scores, adj_length, special_nodes_mask = zip(*decoder_data_list)
        edge_index, edge_type = zip(*adj_data_list)

        node_ids = torch.stack(node_ids, dim=0)
        node_type_ids = torch.stack(node_type_ids, dim=0)
        adj_length = torch.stack(adj_length, dim=0)

        return node_ids, node_type_ids, adj_length,\
            edge_index, edge_type

    def load_graph(self, example_id):
        """
        Load subgraph from adj_folder/example_id.pkl 
        """
        graph_path = os.path.join(self.adj_path, example_id + ".pkl")
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
        return graph

    def preprocess_graph(self, graph):
        """
        Adapted from load_sparse_adj_data_with_contextnode in utils/data_utils.py L653
        An example is a tuple of (adj, nodes, qmask, amask, statement_id)
        Returns:
            node_ids: [max_node_num]
            node_type_ids: [max_node_num]
            node_scores: [max_node_num, 1]
            adj_length: (1,)
            special_nodes_mask: [max_node_num]
            (edge_index, edge_type): [2, max_edge_num], [max_edge_num]
        """
        # Define special nodes and links
        context_node = 0
        n_special_nodes = 1
        cxt2qlinked_rel = 0
        cxt2alinked_rel = 1
        half_n_rel = get_tweaked_num_relations(self.num_relations, self.cxt_node_connects_all)
        if self.cxt_node_connects_all:
            cxt2other_rel = half_n_rel - 1

        node_ids = torch.full((self.max_node_num,), 1, dtype=torch.long)
        node_type_ids = torch.full((self.max_node_num,), 2, dtype=torch.long) #default 2: "other node"
        node_scores = torch.zeros((self.max_node_num, 1), dtype=torch.float)
        special_nodes_mask = torch.zeros((self.max_node_num,), dtype=torch.bool)

        # adj: e.g. <4233x249 (n_nodes*half_n_rels x n_nodes) sparse matrix of type '<class 'numpy.bool'>' with 2905 stored elements in COOrdinate format>
        # nodes: np.array(num_nodes, ), where entry is node id
        # qm: np.array(num_nodes, ), where entry is True/False
        # am: np.array(num_nodes, ), where entry is True/False
        adj, nodes, qmask, amask = graph[:4]
        assert len(nodes) == len(set(nodes))
        assert len(nodes) == len(qmask), "The number of nodes should be the same as the number of qmask"

        qamask = qmask | amask
        # Sanity check: should be T,..,T,F,F,..F
        if len(nodes) == 0:
            pass
        else:
            # qamask[0] is np.bool_ type, is operator doesn't work
            assert qamask[0], "The existing nodes should not be masked"
        f_start = False
        for tf in qamask:
            if tf is False:
                f_start = True
            else:
                assert f_start is False

        assert n_special_nodes <= self.max_node_num
        special_nodes_mask[:n_special_nodes] = 1
        if self.kg_only_use_qa_nodes:
            actual_max_node_num = torch.tensor(qamask).long().sum().item()
        else:
            actual_max_node_num = self.max_node_num
        num_node = min(len(nodes) + n_special_nodes, actual_max_node_num) # this is the final number of nodes including contextnode but excluding PAD
        adj_length = torch.tensor(num_node)

        # Prepare nodes
        nodes = nodes[:num_node - n_special_nodes]
        node_ids[n_special_nodes:num_node] = torch.tensor(nodes) + 1  # To accomodate contextnode, original node_ids incremented by 1
        node_ids[0] = context_node # this is the "node_id" for contextnode

        # Prepare node types
        node_type_ids[0] = 3 # context node
        node_type_ids[1:n_special_nodes] = 4 # sent nodes
        node_type_ids[n_special_nodes:num_node][torch.tensor(qmask, dtype=torch.bool)[:num_node - n_special_nodes]] = 0
        node_type_ids[n_special_nodes:num_node][torch.tensor(amask, dtype=torch.bool)[:num_node - n_special_nodes]] = 1

        #Load adj
        ij = torch.tensor(adj.row, dtype=torch.int64) #(num_matrix_entries, ), where each entry is coordinate
        k = torch.tensor(adj.col, dtype=torch.int64)  #(num_matrix_entries, ), where each entry is coordinate
        n_node = adj.shape[1]
        if n_node > 0:
            assert self.num_relations == adj.shape[0] // n_node
            i, j = torch.div(ij, n_node, rounding_mode='floor'), ij % n_node
        else:
            i, j = ij, ij

        #Prepare edges
        # **** increment coordinate by 1, rel_id by 2 ****
        i += 2
        j += 1
        k += 1
        extra_i, extra_j, extra_k = [], [], []
        for _coord, q_tf in enumerate(qmask):
            _new_coord = _coord + n_special_nodes
            if _new_coord > num_node:
                break
            if q_tf:
                extra_i.append(cxt2qlinked_rel) #rel from contextnode to question node
                extra_j.append(0) #contextnode coordinate
                extra_k.append(_new_coord) #question node coordinate
            elif self.cxt_node_connects_all:
                extra_i.append(cxt2other_rel) #rel from contextnode to other node
                extra_j.append(0) #contextnode coordinate
                extra_k.append(_new_coord) #other node coordinate
        for _coord, a_tf in enumerate(amask):
            _new_coord = _coord + n_special_nodes
            if _new_coord > num_node:
                break
            if a_tf:
                extra_i.append(cxt2alinked_rel) #rel from contextnode to answer node
                extra_j.append(0) #contextnode coordinate
                extra_k.append(_new_coord) #answer node coordinate
            elif self.cxt_node_connects_all:
                extra_i.append(cxt2other_rel) #rel from contextnode to other node
                extra_j.append(0) #contextnode coordinate
                extra_k.append(_new_coord) #other node coordinate

        # half_n_rel += 2 #should be 19 now
        if len(extra_i) > 0:
            i = torch.cat([i, torch.tensor(extra_i)], dim=0)
            j = torch.cat([j, torch.tensor(extra_j)], dim=0)
            k = torch.cat([k, torch.tensor(extra_k)], dim=0)

        ########################

        mask = (j < actual_max_node_num) & (k < actual_max_node_num)
        i, j, k = i[mask], j[mask], k[mask]
        i, j, k = torch.cat((i, i + half_n_rel), 0), torch.cat((j, k), 0), torch.cat((k, j), 0)  # add inverse relations
        edge_index = torch.stack([j,k], dim=0) # each entry is [2, E]
        edge_type = i # each entry is [E, ]

        # node_ids:  [max_node_num]
        # node_type_ids: [max_node_num]
        # node_scores: [max_node_num]
        # adj_lengths: [1]
        # edge_index: [2, E]
        # edge_type: [E, ]
        return node_ids, node_type_ids, node_scores, adj_length, special_nodes_mask, (edge_index, edge_type)

    def read_statements(self, input_file):
        """
        Retruns:
            example_id (str): id
            contexts (str): context if exists, otherwise ""
            question (str): question
            endings (str): answer
            label (int): label, -100 in our case
        """
        with open(input_file, "r", encoding="utf-8") as f:
            examples = []
            for line in tqdm(f.readlines()):
                item = json.loads(line)
                label = item["label"] if "label" in item else -100
                context = item["context"] if "context" in item else ""
                examples.append(
                    InputExample(
                        example_id=item["id"],
                        contexts=context,
                        question=item["question"],
                        endings=item["answers"],
                        label=label
                    ))
        return examples

    def inject_choices(self, question, choices):
        """Inject candidates into the question
        Args:
            question (str): question
            choices (list): list of strings
        Returns:
            question (str): question with candidates injected
        """
        # Inject candidates into the question
        # We use the following format: <question> \n (A) <choice1> (B) <choice2> ...
        # This is the same format as the one used in the original code
        question = question + ' \n Candidates:'
        for i, choice in enumerate(choices):
            # it supports up to 7 choices (A to G)
            question = question + f" ({'ABCDEFG'[i]}) {choice}"
        return question

    def postprocess_text(self, example):
        """Adapted from load_input_tensors in utils/data_utils.py L584
        Args:
            example: an InputExample object
        Returns:
            encoder_inputs (dict): a dictionary containing input_ids and attention_mask
            decoder_labels (tensor | list): input_ids for decoder, or a list of raw strings as answers
        """
        # There is no choice in out current setting, this is purely for compatibility with the original code

        context = example.contexts
        assert isinstance(context, str), "Currently contexts must be a string"
        question = example.question
        answers = example.endings
        assert isinstance(answers, list), "Currently answers must be a list of strings"
        # Here we separately encode the question and answer
        # Warning: we assume the encoder and the decoder share the same tokenizer
        # but this is not necessarily true!
        # Padding is removed here, it's done in collate function
        if self.encoder_input == 'context':
            encoder_input = context
        elif self.encoder_input == 'question':
            encoder_input = 'question: ' + question
            if self.decoder_label == 'choices' and self.has_choice_graph:
                encoder_input = self.inject_choices(encoder_input, answers)
        elif self.encoder_input == 'contextualized_question':
            encoder_input = 'question: ' + question + ' context: ' + context
        elif self.encoder_input == 'context_prefix':
            context_splited = context.split()
            prefix_length = math.floor(len(context_splited) * self.prefix_ratio)
            encoder_input = "complete: " +  " ".join(context_splited[:prefix_length])
        else: # self.encoder_input == 'retrieval_augmented_question':
            raise NotImplementedError(f"Encoder input {self.encoder_input} is not implemented")
            # TODO: integrate retrieval
            retrieved_doc = self.retriever(question)
            encoder_input = question + self.tokenizer.sep_token + retrieved_doc
        encoder_inputs = self.tokenizer(encoder_input, truncation=True,
                                        max_length=self.max_seq_length,
                                        return_tensors='pt')
        # decoder label:
        # - context or context_label are for pretraining, where context returns original text, context_label returns
        #   mlm labels of the context
        # - answer is for downstream tasks
        if self.decoder_label == self.encoder_input:
            decoder_labels = encoder_inputs['input_ids']
        elif self.decoder_label == 'raw_answers':
            decoder_labels = answers
        else:
            if self.decoder_label == 'context':
                decoder_input = context
            elif self.decoder_label == 'answer':
                answer = answers[0]
                decoder_input = answer
            elif self.decoder_label == 'context_suffix':
                # As context_prefix and context_suffix are always used together, we should
                # be able to access the context_splited and prefix_length defined earlier
                decoder_input = " ".join(context_splited[prefix_length:])
            elif self.decoder_label == 'choices':
                decoder_input = answers
            else:
                raise NotImplementedError(f"decoder_input {self.decoder_label} is not implemented")
            decoder_inputs = self.tokenizer(decoder_input, truncation=True,
                                            max_length=self.max_seq_length,
                                            return_tensors='pt', padding=True)
            decoder_labels = decoder_inputs['input_ids']
        return encoder_inputs, decoder_labels


def shift_tokens_right(input_ids: np.array, pad_token_id: int, decoder_start_token_id: int) -> np.ndarray:
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids

@dataclass
class DataCollatorForT5MLM:
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """
    tokenizer: PreTrainedTokenizerBase
    noise_density: float
    mean_noise_span_length: float
    pad_token_id: int
    decoder_start_token_id: int

    def __call__(self, batch) -> BatchEncoding:
        # convert list to dict and tensorize input
        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = np.asarray([self.random_spans_noise_mask(expandend_input_length) for i in range(batch_size)])
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        # to check that tokens are correctly preprocessed, one can run `self.tokenizer.batch_decode(input_ids)` and `self.tokenizer.batch_decode(labels)` here...
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.pad_token_id, self.decoder_start_token_id
        )
        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.
        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number
        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]


class Retriever(ABC):
    """Retriever base class.
    """
    @abstractmethod
    def __init__(self, n_docs=1):
        self.n_docs = n_docs

    @abstractmethod
    @lru_cache(maxsize=None)
    def retrieve_contexts(self, question):
        """Retrieve contexts for a given question.

        Args:
            question (str): The question to retrieve contexts for.
            aggregate_docs_fn (callable(list -> str), optional): Function to aggregate the retrieved documents.

        Retruns:
            contexts (list): The retrieved contexts.
        """
        pass

    def __call__(self, question):
       return self.retrieve_contexts(question)


def create_dummy_graph(batch_size):
    node_ids = torch.ones((batch_size, 1, 1), dtype=torch.long)
    node_type_ids = torch.full_like(node_ids, 3)
    adj_lengths = torch.ones((batch_size, 1), dtype=torch.long)
    edge_index = [[torch.zeros((2, 0), dtype=torch.long)] for _ in range(batch_size)]
    edge_type = [[torch.zeros((0,), dtype=torch.long)] for _ in range(batch_size)]
    # TODO: do this in one step
    node_ids = node_ids.squeeze(1)
    node_type_ids = node_type_ids.squeeze(1)
    adj_lengths = adj_lengths.squeeze(1)
    edge_index = sum(edge_index,[])
    edge_type = sum(edge_type,[])
    return node_ids, node_type_ids, adj_lengths, edge_index, edge_type


def seq2seq_collate_texts(input_ids, attention_mask, decoder_labels=None, *, tokenizer):
    """
    The inputs are assumed to have the shape of batch_size * [num_choices, seq_len]
    The outputs flatten the choice dimension, and pad the sequences to the same length.
    Returns:
        a dictionary with keys:
            - input_ids: [batch_size, num_choices, seq_len]
            - attention_mask: [batch_size, num_choices, seq_len]
            - labels: [batch_size, num_choices, seq_len]
    """
    collator = DataCollatorForSeq2Seq(tokenizer)
    # Squeeze the choice dimension
    batch_size = len(input_ids)
    inputs = [input_ids, attention_mask, decoder_labels]
    for i, sublists in enumerate(inputs):
        # flatten the choice dimension:convert to a list of lenght batch_size * choice_dim
        if sublists is not None:
            inputs[i] = [element for sublist in sublists for element in sublist]
    input_ids, attention_mask, decoder_labels = inputs
    features = [
        {
            'input_ids': ii,
            'attention_mask': am,
        }
        for ii, am in zip(input_ids, attention_mask)
    ]
    if decoder_labels is not None:
        for i, dl in enumerate(decoder_labels):
            features[i]['labels'] = dl
    features = collator(features)
    features = {k: v.view(batch_size, -1, *v.shape[1:]) for k, v in features.items()}
    return features


def corrupt_texts(features, *, tokenizer, mlm_probability=0.2, mean_noise_span_length=3,
                  pad_token_id=0, decoder_start_token_id=0):
    assert "labels" in features, "labels are necessary for corrupting texts"
    collator = DataCollatorForT5MLM(
        tokenizer=tokenizer,
        noise_density=mlm_probability,
        mean_noise_span_length=mean_noise_span_length,
        pad_token_id=pad_token_id,
        decoder_start_token_id=decoder_start_token_id
    )
    features = collator(features)
    return features


@dataclass
class T5GNNMultipleChoiceDataCollator:
    """For DragonDataset of encoder-decoder architecture.
    """
    tokenizer: PreTrainedTokenizer
    dummy_graph: bool = False
    corrupt_text: bool = False
    return_raw_answers: bool = False
    num_choices: int = 1

    def __call__(self, examples: List[InputFeatures]):
        """Collate examples into a batch.

        Returns:
            input_ids, attention_mask, decoder_labels: [batch_size, num_choices, max_seq_length]
            choice_labels: [batch_size, 1]
            node_id, node_type_id: batch_size * [max_node_num] -> [batch_size, num_choices, max_node_num]
            adj_lengths: batch_size * [1] -> [batch_size, num_choices, 1]
            edge_index: batch_size * [2, E] -> batch_size * [num_choices 2, E]
            edge_type: batch_size * [E, ] -> batch_size * [num_choices, E]
        """
        # Tensors of shape [batch_size, num_choices, max_seq_length] (written separately for clarity)
        decoder_labels = [example.decoder_labels for example in examples] # individual labels: [k, seq_len]
        input_ids = [example.input_ids for example in examples]  # individual input ids: [1, seq_len]
        attention_mask = [example.attention_mask for example in examples] # individual attention mask: [1, seq_len]
        # Expand question input ids and attention mask to match the number of choices
        input_ids = [input_id.expand(self.num_choices, -1) for input_id in input_ids]
        attention_mask = [attention_mask.expand(self.num_choices, -1) for attention_mask in attention_mask]
        assert len(decoder_labels[0]) == self.num_choices, "number of choices does not match"
        # The texts were tokenized with return_tensors='pt', so they have an extra first dimension
        features = seq2seq_collate_texts(
            input_ids, attention_mask,
            decoder_labels=None if self.return_raw_answers else decoder_labels,
            tokenizer=self.tokenizer)
        if self.corrupt_text:
            features = corrupt_texts(features, tokenizer=self.tokenizer)

        input_ids, attention_mask = features['input_ids'], features['attention_mask']
        if not self.return_raw_answers:
            decoder_labels = features['labels']
        choice_labels = torch.tensor([example.choice_label for example in examples]).view(-1, 1)

        if self.dummy_graph:
            batch_size = input_ids.size(0)
            node_ids, node_type_ids, adj_lengths, edge_index,\
                edge_type = create_dummy_graph(batch_size)
        else:
            # batch_size * [num_choices, max_nodes] => [batch_size, num_choices, max_nodes]
            node_ids = torch.stack([example.node_ids for example in examples], dim=0)
            node_type_ids = torch.stack([example.node_type_ids for example in examples], dim=0)
            # batch_size * [num_choices] => [batch_size, num_choices]
            adj_lengths = torch.stack([example.adj_length for example in examples], dim=0)
            # batch_size * num_choices * [2, E] => [batch_size, [num_choices, [2, E]]], E varies
            edge_index = [example.edge_index for example in examples]
            # batch_size * [num_choices, [E, ]] => [batch_size, [num_choices, [E,]]], E varies
            edge_type = [example.edge_type for example in examples]
        return input_ids, attention_mask, decoder_labels, choice_labels, \
            node_ids, node_type_ids, adj_lengths, \
            edge_index, edge_type


def load_data(args, corrupt=False, dummy_graph=False, num_workers=1, num_choices=5, has_choice_graph=False,
              train_kwargs={'encoder_input': 'contextualized_question', 'decoder_label': 'answer'},
              val_kwargs={'encoder_input': 'contextualized_question', 'decoder_label': 'raw_answers'}):
    """Construct the dataset and return dataloaders

    Args:
        dataset_cls (class): DragonDataset or DragonEncDecDataset
        collate_fn (callable): dragond_collate_fn, dragond_adapt2enc_collate_fn or dragond_encdec_collate_fn
        corrupt (bool): whether to corrupt the graph and text
        num_workers (int): number of workers for dataloader
        prefix_ratio (float): ratio of prefix in prefix-LM
    Returns:
        train_dataloader, validation_dataloader
    """
    num_relations = args.num_relations
    model_name = args.encoder_name_or_path
    max_seq_length = args.max_seq_len
    prefix_ratio = float(args.prefix_ratio)
    train_dataset = LMGNNChoiceDataset(
        statement_path=args.train_statements,
        num_relations=num_relations,
        adj_path=args.train_adj,
        model_name=model_name,
        max_seq_length=max_seq_length,
        prefix_ratio=prefix_ratio,
        has_choice_graph=has_choice_graph,
        **train_kwargs)
    validation_dataset = LMGNNChoiceDataset(
        statement_path=args.dev_statements,
        adj_path=args.dev_adj,
        num_relations=num_relations,
        model_name=model_name,
        max_seq_length=max_seq_length,
        prefix_ratio=prefix_ratio,
        has_choice_graph=has_choice_graph,
        **val_kwargs)
    # get tokenizer
    train_collator = T5GNNMultipleChoiceDataCollator(
        tokenizer=train_dataset.tokenizer,
        corrupt_text=corrupt,
        dummy_graph=dummy_graph,
        return_raw_answers=False,
        num_choices=num_choices)
    val_collator = T5GNNMultipleChoiceDataCollator(
        tokenizer=validation_dataset.tokenizer,
        corrupt_text=False,
        dummy_graph=dummy_graph,
        return_raw_answers=val_kwargs['decoder_label'] == 'raw_answers',
        num_choices=num_choices)
    train_dataloader = DataLoader(train_dataset, collate_fn=train_collator, batch_size=args.batch_size,num_workers=num_workers)
    validation_dataloader = DataLoader(validation_dataset, collate_fn=val_collator, batch_size=args.batch_size, num_workers=num_workers)
    return train_dataloader, validation_dataloader
