from typing import Any, Callable, Dict
from transformers import BertConfig
from retriever import FlaxBertModel
from EasyLM.models.llama.llama_model import LLaMAConfig, FlaxLLaMAForCausalLMModule

from flax.training.train_state import TrainState
from flax import linen as nn
from flax import core
from flax import struct

import optax
import jax
import jax.numpy as jnp


class RALMTrainState(TrainState):
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState = struct.field(pytree_node=True)
    kb_index: jnp.ndarray = None
    kb_docs: Dict = None
    kb_tokens: Dict = None

    @classmethod
    def create(cls, *, apply_fn, params, tx, kb_index=None, kb_docs=None, kb_tokens=None, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            kb_index=kb_index,
            kb_docs=kb_docs,
            kb_tokens=kb_tokens,
            **kwargs,
        )
        

class RALM(nn.Module):
    k: int
    bert_config: BertConfig
    llama_config: LLaMAConfig
    reader_tokenizer: Any

    def setup(self):
        self.retriever = FlaxBertModel.module_class(self.bert_config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
        self.reader = FlaxLLaMAForCausalLMModule(self.llama_config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)

    def __call__(self, input_ids, attention_mask):
        self.retriever(input_ids, attention_mask)
        self.reader(input_ids, attention_mask)

    def __ralm_call__(self, retriever_input_ids, retriever_attention_mask, reader_input_ids, reader_attention_mask, reader_loss_mask, kb_index, kb_tokens):
        ''' Note!
            1. retriever_input_ids/retriever_attention_mask are padded and collated as jnp.Array.
            2. reader_input_ids/reader_attention_mask are not padded. They are a list of {
                "input_ids": List of input ids, "attention_mask": List of attention masks.
            }, the collating process is completed in the forward pass (this function). Also, read_input_ids is a concatenation of input_ids and output_ids.
        '''
        # retrieve top-k docs
        scores, indices = self.retrieve_docs(retriever_input_ids, retriever_attention_mask, kb_index)
        # collecting retrieved docs
        doc_tokens = self.collect_docs(indices, kb_tokens)
        # generate a new input/output pair for reader (with retrieved tokens concatenated)
        pass



        reader_output = self.reader(input_ids, attention_mask)
    
    def retrieve_docs(self, retriever_input_ids, retriever_attention_mask, search_index):
        input_reprs = self.retriever(
            input_ids=retriever_input_ids,
            attention_mask=retriever_attention_mask
        ).last_hidden_state[:, 0, :] # (batch_size, dim)

        # search_index: (KB_size, dim)
        # use MIPS to retrieve top-k docs (approximate search for now)
        scores = jax.lax.dot(input_reprs, search_index.transpose()) # (batch_size, KB_size)
        topk_scores, topk_indices = jax.lax.approx_max_k(scores, self.k) # (batch_size, k)
        return topk_scores, topk_indices
    
    def collect_docs(self, indices, kb_tokens):
        retrieved_docs = []
        for index in indices.tolist():
            index_docs = []
            for doc_idx in index:
                index_docs.append(kb_tokens[doc_idx])
            retrieved_docs.append(index_docs)
        return retrieved_docs
    
    def prepare_input(self, reader_input_ids, reader_attention_mask, reader_loss_mask, doc_tokens):
        # doc_tokens: list of (list of {input_ids:, attention_mask:} of size k) of size batch_size
        for batch_idx in range(len(reader_input_ids)):
            final_input_ids = reader_input_ids + [self.reader_tokenizer.sep_token_id]
    
    def compute_index(self, input_ids, attention_mask):
        retriever_output = self.retriever(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        batch_index = retriever_output.last_hidden_state[:, 0, :]
        return batch_index