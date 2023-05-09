import dataclasses
import pprint
from functools import partial
import json

import mlxu
from ml_collections.config_dict import config_dict
from ml_collections import ConfigDict
from tqdm import tqdm, trange
import numpy as np

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


class TextProcessor(object):
    """ Example processor that converts a dictionary of texts into tokens. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.fields_from_example = ''
        config.fields = ''
        config.subfield_separator = ' '
        config.add_eos_token = True
        config.prepend_text = ''
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer):
        self.config = self.get_default_config(config)
        assert self.config.fields != '' or self.config.fields_from_example != '', (
            'Either fields or fields_from_example must be specified.'
        )
        self.tokenizer = tokenizer

    def __call__(self, example):
        token_buffer = []
        loss_mask_buffer = []
        if self.config.fields_from_example != '':
            fields = example[self.config.fields_from_example].split(',')
        else:
            fields = self.config.fields.split(',')

        for i, field in enumerate(fields):
            if field.startswith('[') and field.endswith(']'):
                # No loss for this field.
                field = field[1:-1]
                mask = 0.0
            else:
                mask = 1.0

            if field == '<|bos|>':
                token_buffer.append(self.tokenizer.bos_token_id)
                loss_mask_buffer.append(mask)
            elif field == '<|eos|>':
                token_buffer.append(self.tokenizer.eos_token_id)
                loss_mask_buffer.append(mask)
            else:
                subfields = field.split('+')
                text = self.config.subfield_separator.join(
                    [example[subfield] for subfield in subfields]
                )
                if i == 0:
                    text = self.config.prepend_text + text
                tokens = self.tokenizer.encode(text)
                token_buffer.extend(tokens)
                loss_mask_buffer.extend([mask for _ in range(len(tokens))])

        if self.config.add_eos_token:
            token_buffer.append(self.tokenizer.eos_token_id)
            loss_mask_buffer.append(1.0)

        return token_buffer, loss_mask_buffer


class JsonDataset(object):
    """ JSON dataset, where each line of the data file contains a JSON
        dictionary with text fields.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ''
        config.seq_length = 512
        config.batch_size = 8

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        assert self.config.path != ''
        self._tokenizer = tokenizer
        self._text_processor = text_processor

    def json_iterator(self):
        while True:
            with mlxu.open_file(self.config.path, 'r') as fin:
                for line in fin:
                    if not line or line == '\n':
                        continue
                    try:
                        data = json.loads(line)
                    except json.decoder.JSONDecodeError:
                        print(f'Error parsing json line:\n{line}')
                        continue
                    yield data
    
    def _compute_pad_length(self, l:int):
        return min(((l - 1) // 128 + 1) * 128, self.config.seq_length)

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        token_buffer = []
        loss_mask_buffer = []
        for example in self.json_iterator():
            tokens, loss_masks = self.text_processor(example)
            if len(tokens) > self.config.seq_length:
                tokens = tokens[:self.config.seq_length]
                loss_masks = tokens[:self.config.seq_length]
            token_buffer.append(tokens)
            loss_mask_buffer.append(loss_masks)
            while len(token_buffer) > self.config.batch_size:
                batch_tokens = token_buffer[:self.config.batch_size]
                max_length = self._compute_pad_length(max(len(t) for t in batch_tokens))
                batch_padded_tokens = [t + [self._tokenizer.pad_token_id] * (max_length - len(t)) for t in batch_tokens]
                batch_padded_loss_mask = [t + [0.] * (max_length - len(t)) for t in loss_mask_buffer[:self.config.batch_size]]
                yield {
                    'tokens': np.array(batch_padded_tokens, dtype=np.int32),
                    'loss_masks': np.array(batch_padded_loss_mask, dtype=np.float32),
                }
                token_buffer = token_buffer[self.config.batch_size:]
                loss_mask_buffer = loss_mask_buffer[self.config.batch_size:]

    def __getstate__(self):
        return self.config, self.tokenizer

    def __setstate__(self, state):
        config, tokenizer = state
        self.__init__(config, tokenizer)

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def vocab_size(self):
        return len(self.tokenizer)


class KBDataset(Dataset):
    def __init__(self, tokenizer, path):
        self.tokenizer = tokenizer
        self.examples = []

        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            for line in lines:
                data_item = json.loads(line)
                encodings = self.tokenizer(data_item["document"])
                input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]

                self.examples.append({"input_ids": input_ids, "attn_mask": attention_mask, "document": data_item["document"]})
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def collate_fn(self, batch_examples):
        batch_items = [{"input_ids": example["input_ids"], "attention_mask": example["attn_mask"]} for example in batch_examples]
        padded_items = self.tokenizer.pad(batch_items, return_tensors="jax")
        return {
            "input_ids": padded_items["input_ids"],
            "attention_mask": padded_items["attention_mask"]
        }



if __name__ == "__main__":
    from EasyLM.models.llama.llama_model import LLaMAConfig
    config = JsonDataset.get_default_config()
    config.path = "/home/zhangzx.sjtu/ralm/src/data/example_data.jsonl"

    print(config)
    llama_tokenizer_config = LLaMAConfig.load_config("7b").get_tokenizer_config()
    llama_tokenizer_config.vocab_file = "./ralm_ckpt/tokenizer.llama"
    llama_tokenizer = LLaMAConfig.get_tokenizer(llama_tokenizer_config)
    print(llama_tokenizer)

    sents = ["You can set environment variables in a service's containers with the environment attribute in your Compose file.", "However, it's best to keep these values inside "]

    sentences = [sents, sents]

    print(llama_tokenizer(sents[0]))
    print(llama_tokenizer.bos_token_id)
    print(llama_tokenizer.eos_token_id)
    print(llama_tokenizer.pad_token_id)
    print(llama_tokenizer.sep_token_id)

    sent = "Below are some knowledge facts that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."

    '''
    example = {
        "input": "You can set environment variables in a service's containers with the environment attribute in your Compose file.", 
        "output": "However, it's best to keep these values inside."
    }
    text_processor_config = TextProcessor.get_default_config()
    text_processor_config.fields = "[input],output"
    text_processor = TextProcessor(text_processor_config, llama_tokenizer)

    print(text_processor(example))

    dataset = JsonDataset(config, llama_tokenizer, text_processor)

    # for batch in dataset:
    #     print(batch["tokens"].shape)

    from transformers import BertTokenizerFast
    b = BertTokenizerFast.from_pretrained("facebook/contriever")

    kbd = KBDataset(b, "/home/zhangzx.sjtu/ralm/src/data/example_kb.jsonl")
    kbdl = DataLoader(kbd, batch_size=8, collate_fn=kbd.collate_fn)
    for batch in kbdl:
        print(batch)
    '''


