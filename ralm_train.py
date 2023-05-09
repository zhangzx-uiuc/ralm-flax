from EasyLM.jax_utils import (
    JaxRNG, get_jax_mp_mesh, next_rng, match_partition_rules,
    cross_entropy_loss_and_accuracy, named_tree_map, global_norm,
    set_random_seed, average_metrics, get_weight_decay_mask,
    make_shard_and_gather_fns, tree_apply
)
from EasyLM.checkpoint import StreamingCheckpointer
from jax.experimental.pjit import pjit, with_sharding_constraint
from jax.experimental import PartitionSpec as PS
# from flax.training.train_state import TrainState
from ralm_model import RALMTrainState
from transformers import BertConfig, BertTokenizerFast
from EasyLM.models.llama.llama_model import LLaMAConfig
from torch.utils.data import DataLoader

from ralm_model import RALM
from ralm_data import KBDataset

from jax import numpy as jnp
from jax import random

import json
import pickle
import optax
import pickle
import mlxu
import jax


''' Training script for RALM model. '''
# Set global random seed
# k = jax.random.PRNGKey(4222)
print("Setting random seed...")
set_random_seed(4222)

# Load config files
llama_config = LLaMAConfig.load_config('7b')
bert_config = BertConfig.from_pretrained("facebook/contriever")

# load Bert and LLaMA tokenizers
llama_tokenizer_config = llama_config.get_tokenizer_config()
llama_tokenizer_config.vocab_file = "./ralm_ckpt/tokenizer.llama"
llama_tokenizer = LLaMAConfig.get_tokenizer(llama_tokenizer_config)
bert_tokenizer = BertTokenizerFast.from_pretrained("facebook/contriever")

model = RALM(10, bert_config, llama_config, llama_tokenizer)

# Load optimizer
optimizer = optax.adamw(learning_rate=1e-4)

# Load partition rules
with open("./ralm_ckpt/partition.rules", 'rb') as f:
    new_rules = pickle.load(f)

# Generate a pytree of parameter shapes
def init_fn(rng):
    rng_generator = JaxRNG(rng)
    params = model.init(
        input_ids=jnp.zeros((4, 512), dtype=jnp.int32),
        attention_mask=jnp.ones((4, 512), dtype=jnp.int32),
        rngs=rng_generator()
    )
    return RALMTrainState.create(params=params, tx=optimizer, apply_fn=None)

train_state_shapes = jax.eval_shape(init_fn, next_rng())

with open("shapes.txt", 'w') as f:
    print(train_state_shapes, file=f)
   
# Generate a pytree of parameter PartitionSpecs
train_state_partition = match_partition_rules(new_rules, train_state_shapes)

with open("train_state_partition.txt", 'w') as f:
    print(train_state_partition, file=f)

# Generate a pytree of sharding and gathering functions
shard_fns, gather_fns = make_shard_and_gather_fns(train_state_partition, train_state_shapes)
with open("shard_fns.txt", 'w') as f:
    print(shard_fns, file=f)

# Initialize checkpointer
config = StreamingCheckpointer.get_default_config()
new_ckpt = StreamingCheckpointer(config, './ralm_ckpt', enable=True)

# Define TrainState creation function, and shard it.
def create_trainstate_from_params(params):
    return RALMTrainState.create(params=params, tx=optimizer, apply_fn=None)

sharded_create_trainstate_from_params = pjit(
    create_trainstate_from_params,
    in_shardings=(train_state_partition.params,),
    out_shardings=train_state_partition,
    donate_argnums=(0, ),
)

# Create the dataloader for index building.
kb_file_path = "./data/example_kb.jsonl"
kb_dataset = KBDataset(bert_tokenizer, kb_file_path)
kb_dataloader = DataLoader(kb_dataset, batch_size=8, collate_fn=kb_dataset.collate_fn, shuffle=False)

# Define the index building function, and shard it.
def build_index(train_state, rng, batch):
    rng_generator = JaxRNG(rng)
    input_ids = with_sharding_constraint(batch['input_ids'], PS(('mp1', 'mp2'), None))
    attn_mask = with_sharding_constraint(batch['attention_mask'], PS(('mp1', 'mp2'), None))
    batch_embedding = model.apply(
        train_state.params,
        input_ids=input_ids,
        attention_mask=attn_mask,
        rngs={"params": rng_generator(), "dropout": rng_generator()},
        method=model.compute_index
    )
    return batch_embedding

sharded_build_index = pjit(
    build_index,
    in_shardings=(train_state_partition, PS(), PS()),
    out_shardings=PS(('mp1', 'mp2'), None),
    donate_argnums=(1, ),
)

# Define the tokenization function.
def reader_tokenize(reader_tokenizer, kb_docs):
    kb_tokens = {i:reader_tokenizer(document) for i,document in kb_docs.items()}
    return kb_tokens

# Define the device mesh: distributed training is disabled.
mesh = get_jax_mp_mesh((8, 1), mp_axis_prefix='mp', dp_axis_name='dp')
print(mesh)

# Load the model parameters from the checkpoint, and shard them.
with mesh:
    _, llama_params = new_ckpt.load_trainstate_checkpoint('params::./ralm_ckpt/ralm.model', train_state_shapes, shard_fns)
    new_state = sharded_create_trainstate_from_params(llama_params)

    # Build the index. (if it is not loaded from the checkpoint)
    kb_index = jnp.zeros((0, 768), dtype=jnp.bfloat16)
    for batch in kb_dataloader:
        batch_embedding = sharded_build_index(new_state, next_rng(), batch)
        kb_index = jnp.concatenate((kb_index, batch_embedding), axis=0)

    jax.debug.visualize_array_sharding(kb_index)
    kb_docs = {i:example["document"] for i,example in enumerate(kb_dataset.examples)}
    kb_tokens = reader_tokenize(llama_tokenizer, kb_docs)
    new_state = new_state.replace(kb_index=kb_index, kb_docs=kb_docs, kb_tokens=kb_tokens)

    # Update the partition rules after computing the index (add rules for index)
    train_state_partition = train_state_partition.replace(kb_index=PS(('mp1', 'mp2'), None))

