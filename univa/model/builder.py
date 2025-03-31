#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import sys
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from univa.model import *
from univa.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    elif getattr(kwargs, 'torch_dtype', False):
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'qwen2' in model_name.lower():
        print(f'=> loading UnivaQwen2ForCausalLM ...')
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = UnivaQwen2ForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True,
            **kwargs
        )
    else:
        print(f'=> loading UnivaLlamaForCausalLM ...')
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = UnivaLlamaForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True,
            **kwargs
        )
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        print('[WARNING] CLIP is not loaded!! Loading now ...')
        vision_tower.load_model(device_map=device_map)
    if device_map != 'auto':
        vision_tower.to(device=device_map, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    new_size = getattr(model.config, 'mm_vision_resolution', False)
    if new_size:
        size_dict = {'height': new_size, 'width': new_size}
        image_processor.crop_size = size_dict
        image_processor.size = {"shortest_edge": new_size}
        print(f'Crop size changed to {new_size}x{new_size}')

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 4096

    return tokenizer, model, image_processor, context_len
