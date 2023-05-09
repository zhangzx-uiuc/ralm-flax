import dataclasses
import pprint
from functools import partial
import re
import os
from threading import Lock
from tqdm import tqdm


from tqdm import tqdm, trange
import numpy as np
import mlxu

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit, with_sharding_constraint
from jax.experimental import PartitionSpec as PS
import flax
from flax import linen as nn
from flax.jax_utils import prefetch_to_device
from flax.training.train_state import TrainState
import optax
from transformers import GenerationConfig, FlaxLogitsProcessorList

from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.serving import LMServer
from EasyLM.jax_utils import (
    JaxRNG, get_jax_mp_mesh, next_rng, match_partition_rules, tree_apply,
    set_random_seed, get_float_dtype_by_name, make_shard_and_gather_fns,
    FlaxTemperatureLogitsWarper
)
from EasyLM.models.llama.llama_model import LLaMAConfig, FlaxLLaMAForCausalLM

import json


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    initialize_jax_distributed=False,
    mp_mesh_dim='-1,1',
    dtype='bf16',
    input_length=512,
    seq_length=1024,
    top_k=50,
    temperature=1.0,
    top_p=1.0,
    do_sample=True,
    num_beams=1,
    loglikelihood_add_bos_token=True,
    load_llama_config='',
    load_checkpoint='',
    prediction_input_files='',
    prediction_input_field_mappings='',
    prediction_output_file='',
    prediction_output_field='',
    prediction_batch_size=1,
    template_index=0,
    tokenizer=LLaMAConfig.get_tokenizer_config(),
    lm_server=LMServer.get_default_config(),
)


templates = [
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCan you generate some questions that can be answered with the following information?\n\n{instruction}\n\n### Response:\n",
    "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n",
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCan you generate some questions that can be answered with the following information? The questions can related to either the date (if provided) or the facts. Please also give the answers based on the information.\n\n{date}{instruction}\n\n### Response:\n",
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCan you generate some questions that can be answered with the following information? The questions can related to either the date (if provided) or the facts. Make sure the questions don't coreference to any concept in the above information, such as 'this event' or 'that person'.\n\n{date}{instruction}\n\n### Response:\n",
    "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{date}{context}\n\n### Response:\n",
]

def parse_field_mappings(field_mappings:str):
    field_mappings = field_mappings.split(",")
    d = {}
    for field_mapping in field_mappings:
        src_fields, tgt_field = field_mapping.split('=')
        d[tgt_field] = src_fields.split('+')
    return d

def get_fields(mappings, input_item):
    fields = {k: ' '.join([input_item[vv] if input_item[vv] is not None else '' for vv in v]) for k, v in mappings.items()}
    if 'date' in fields:
        if fields['date'].strip() == '':
            fields['date'] = ''
        else:
            fields['date'] = f"Date: {fields['date']}. "
    
    return fields

def main(argv):
    if FLAGS.initialize_jax_distributed:
        jax.distributed.initialize()
    set_random_seed(FLAGS.seed)
    print("loading inputs")
    if FLAGS.prediction_input_files.count(',') > 0:
        FLAGS.prediction_input_files = FLAGS.prediction_input_files.split(',')
    else:
        FLAGS.prediction_input_files = [FLAGS.prediction_input_files]
    input_lines = []
    for input_file in FLAGS.prediction_input_files:
        if not os.path.exists(input_file):
            raise ValueError(f'Input file {input_file} does not exist')
        with open(input_file, 'r') as f:
            input_lines += [json.loads(line) for line in f]
    field_mappings = parse_field_mappings(FLAGS.prediction_input_field_mappings)

    input_text = [templates[FLAGS.template_index].format(**get_fields(field_mappings, line)) for line in input_lines]
    print(input_text[:3])
    input_chunks = [input_text[i:i + FLAGS.prediction_batch_size] for i in range(0, len(input_text), FLAGS.prediction_batch_size)]
    print("loading model")
    prefix_tokenizer = LLaMAConfig.get_tokenizer(
        FLAGS.tokenizer, truncation_side='left', padding_side='left'
    )
    tokenizer = LLaMAConfig.get_tokenizer(
        FLAGS.tokenizer, truncation_side='right', padding_side='right'
    )

    with jax.default_device(jax.devices("cpu")[0]):
        llama_config = LLaMAConfig.load_config(FLAGS.load_llama_config)
        _, params = StreamingCheckpointer.load_trainstate_checkpoint(
            FLAGS.load_checkpoint, disallow_trainstate=True
        )

        hf_model = FlaxLLaMAForCausalLM(
            llama_config,
            input_shape=(1, FLAGS.seq_length),
            seed=FLAGS.seed,
            _do_init=False
        )
        params = jax.device_put(params, device=jax.devices("cpu")[0])

    model_ps = match_partition_rules(
        LLaMAConfig.get_partition_rules(), params
    )
    shard_fns, _ = make_shard_and_gather_fns(
        model_ps, get_float_dtype_by_name(FLAGS.dtype)
    )

    @partial(
        pjit,
        in_axis_resources=(model_ps, PS(), PS(), PS()),
        out_axis_resources=(PS(), PS())
    )
    def forward_generate(params, rng, batch, temperature):
        batch = with_sharding_constraint(batch, PS('dp'))
        rng_generator = JaxRNG(rng)
        output = hf_model.generate(
            batch['input_tokens'],
            attention_mask=batch['attention_mask'],
            params=params['params'],
            prng_key=rng_generator(),
            logits_processor=FlaxLogitsProcessorList(
                [FlaxTemperatureLogitsWarper(temperature)]
            ),
            generation_config=GenerationConfig(
                max_new_tokens=FLAGS.seq_length - FLAGS.input_length,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=FLAGS.do_sample,
                num_beams=FLAGS.num_beams,
                top_k=FLAGS.top_k,
                top_p=FLAGS.top_p,
            )
        ).sequences[:, batch['input_tokens'].shape[1]:]
        return output, rng_generator()

    mesh = get_jax_mp_mesh(FLAGS.mp_mesh_dim)
    assert len(mesh.shape) == 3, 'MP mesh must be 2D'
    with mesh:
        params = tree_apply(shard_fns, params)
        sharded_rng = next_rng()
    

    def generate(text, temperature):
        nonlocal sharded_rng
        inputs = prefix_tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=FLAGS.input_length,
            return_tensors='np',
        )
        batch = dict(
            input_tokens=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
        with mesh:
            output, sharded_rng = forward_generate(
                params, sharded_rng, batch, temperature
            )
            output = jax.device_get(output)
        output_text = []
        for text in list(tokenizer.batch_decode(output)):
            if tokenizer.eos_token in text:
                text = text.split(tokenizer.eos_token, maxsplit=1)[0]
            output_text.append(text)

        return output_text
    
    outputs = [generate(text_chunk, FLAGS.temperature) for text_chunk in tqdm(input_chunks)]
    outputs = [item for sublist in outputs for item in sublist]

    with open(FLAGS.prediction_output_file, 'w') as f:
        for line, output in zip(input_lines, outputs):
            line[FLAGS.prediction_output_field] = output
            f.write(json.dumps(line) + '\n')

if __name__ == "__main__":
    mlxu.run(main)
