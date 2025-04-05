#    Copyright 2024 Haochen Wang
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
#    --------------------------------------------------------
#    References:
#    LLaVA: https://github.com/haotian-liu/LLaVA
#    transformers: https://github.com/huggingface/transformers
#    --------------------------------------------------------
import sys
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import transformers

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
    Qwen2Config, Qwen2Model, Qwen2ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..univa_arch import UnivaMetaModel, UnivaMetaForCausalLM, CausalLMOutputWithPastWithVM

from transformers.models.qwen2 import modeling_qwen2
from univa.constants import IMAGE_TOKEN_INDEX

class CompiledQwen2MLP(modeling_qwen2.Qwen2MLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.compile
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
class CompiledQwen2RMSNorm(modeling_qwen2.Qwen2RMSNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.compile
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

class CompiledQwen2RotaryEmbedding(modeling_qwen2.Qwen2RotaryEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.compile
    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
# modeling_qwen2.Qwen2MLP = CompiledQwen2MLP
# modeling_qwen2.Qwen2RMSNorm = CompiledQwen2RMSNorm
# modeling_qwen2.Qwen2RotaryEmbedding = CompiledQwen2RotaryEmbedding


class UnivaConfig(Qwen2Config):
    model_type = "univa_qwen2"


class UnivaQwen2Model(UnivaMetaModel, Qwen2Model):
    config_class = UnivaConfig

    def __init__(self, config: Qwen2Config):
        super(UnivaQwen2Model, self).__init__(config)


class UnivaQwen2ForCausalLM(Qwen2ForCausalLM, UnivaMetaForCausalLM):
    config_class = UnivaConfig

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = UnivaQwen2Model(config)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        task_types: Optional[List[str]] = None,
        images_wo_scale_norm: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            # print(f'\n\nforward prepare_inputs_labels_for_multimodal exec 1 attention_mask: {attention_mask.shape}, task_types: {task_types}')
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                boi_ids,
                eoi_ids,
                cache_position,
                encoder_out, 
                image_sizes, 
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                cache_position,
                task_types, 
            )
        else:
            # print('\n\nforward exec 1\n\n')
            boi_ids, eoi_ids, encoder_out = None, None, None
            
        return self.inner_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            boi_ids=boi_ids,
            eoi_ids=eoi_ids,
            images=images,
            cache_position=cache_position,
            encoder_out=encoder_out, 
            task_types=task_types, 
            images_wo_scale_norm=images_wo_scale_norm, 
            **kwargs,
        )

    def inner_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        boi_ids: Optional[torch.LongTensor] = None,
        eoi_ids: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        encoder_out: Optional[torch.LongTensor] = None,
        task_types: Optional[List[str]] = None,
        images_wo_scale_norm: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # mostly obtained from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained("meta-Qwen2/Qwen2-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-Qwen2/Qwen2-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states or getattr(self.config, 'eye_enable', False) or getattr(self.config, 'mask_enable', False),
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        # import ipdb;ipdb.set_trace()
        # if task_types is not None:
        #     import ipdb;ipdb.set_trace()
        #     logits[:, :, 151645] = logits[:, :, 151645] * 100.0
        # print(task_types)/
        logits = logits.float()
        # print(f'eoi predict 151645 {torch.topk(logits[0][-3], k=10)[1].tolist()}, eos predict 192 {torch.topk(logits[0][-2], k=10)[1].tolist()}, \n predict {torch.topk(logits[0][-1], k=10)[1].tolist()}')
        
        loss, lm_loss = None, None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            lm_loss = loss.detach().clone()

        vm_loss = None
        if self.training and getattr(self.config, 'ross_enable', False):
            vm_loss = self.compute_vm_loss(images, hidden_states, boi_ids, eoi_ids)
            loss = loss + vm_loss

        eye_loss = None
        if self.training and getattr(self.config, 'eye_enable', False):
            eye_loss = self.compute_eye_loss(hidden_states, labels, **kwargs)
            loss = loss + eye_loss * self.config.mm_eye_weight

        mask_loss = None
        if self.training and getattr(self.config, 'mask_enable', False):
            mask, ids_restore = encoder_out.pop('mask'), encoder_out.pop('ids_restore')
            mask_loss = self.compute_mask_loss(images, hidden_states, mask, ids_restore, boi_ids, eoi_ids)
            loss = loss + mask_loss * self.config.mm_mask_weight

        denoise_loss = None
        if self.training and getattr(self.config, 'denoise_enable', False):
            batch_gen_idx = [i for i, task_type in enumerate(task_types) if task_type == 'gen']
            if len(batch_gen_idx) > 0:
                batch_gen_images = [images_wo_scale_norm[i] for i in batch_gen_idx]
                batch_gen_conditions = [hidden_states[i] for i in batch_gen_idx]
                batch_gen_boi_ids = [boi_ids[i] for i in batch_gen_idx]
                denoise_loss = self.compute_denoise_loss(batch_gen_images, batch_gen_conditions, batch_gen_boi_ids)
                loss = loss + denoise_loss * self.config.mm_denoise_weight
            else:
                # dummy forward but no comtribute to loss
                batch_gen_images = [images_wo_scale_norm[0]]
                batch_gen_conditions = [hidden_states[0]]
                # und task has image
                batch_gen_boi_ids = [boi_ids[0]]
                denoise_loss = self.compute_denoise_loss(batch_gen_images, batch_gen_conditions, batch_gen_boi_ids)
                loss = loss + denoise_loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastWithVM(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            lm_loss=lm_loss,
            vm_loss=vm_loss,
            eye_loss=eye_loss,
            mask_loss=mask_loss,
            denoise_loss=denoise_loss,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        if not isinstance(image_sizes[0][0], list):
            image_sizes = [[i] for i in image_sizes]
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            # print('\n\ng/enerate prepare_inputs_labels_for_multimodal exec 1\n\n')
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                boi_ids,
                eoi_ids,
                cache_positions,
                encoder_out, 
                image_sizes, 
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
            )
        else:
            raise NotImplementedError
            inputs_embeds = self.get_model().embed_tokens(inputs)
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            image_sizes=image_sizes, 
            **kwargs
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        attention_mask=None,
        **kwargs,
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)

        # print('before prepare_inputs_for_generation')
        # print(
        #     '\tinput_ids', input_ids, 
        #     '\n\tpast_key_values', len(past_key_values.key_cache), past_key_values.key_cache[0].shape if len(past_key_values.key_cache) > 0 else [], 
        #     '\n\tinputs_embeds', inputs_embeds.shape if inputs_embeds is not None else 'None', 
        #     '\n\tcache_position', (kwargs['cache_position'].shape, kwargs['cache_position'][-1]) if kwargs['cache_position'] is not None else 'None', 
        #     '\n\tposition_ids', (kwargs['position_ids'].shape, kwargs['position_ids'][-1][-1]) if kwargs['position_ids'] is not None else 'None', 
        #     '\n\tattention_mask', attention_mask.shape, attention_mask[-1][-1], 
        #     '\n\tuse_cache', kwargs['use_cache'], 
        #     '\n\timages', images.shape if images is not None else 'None', 
        #     '\n\timage_sizes', image_sizes if image_sizes is not None else 'None', 
        #     )
        # import ipdb;ipdb.set_trace()
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            **kwargs
        )
        # print('after prepare_inputs_for_generation')
        # print(
        #     '\tinput_ids', _inputs['input_ids'], 
        #     '\n\tpast_key_values', len(_inputs['past_key_values'].key_cache), _inputs['past_key_values'].key_cache[0].shape if len(_inputs['past_key_values'].key_cache) > 0 else 'None', 
        #     '\n\tinputs_embeds', _inputs['inputs_embeds'].shape if _inputs['inputs_embeds'] is not None else 'None', 
        #     '\n\tcache_position', (_inputs['cache_position'].shape, _inputs['cache_position'][-1]) if _inputs['cache_position'] is not None else 'None', 
        #     '\n\tposition_ids', (_inputs['position_ids'].shape, _inputs['position_ids'][-1][-1]) if _inputs['position_ids'] is not None else 'None', 
        #     '\n\tattention_mask', _inputs['attention_mask'].shape, attention_mask[-1][-1], 
        #     '\n\tuse_cache', _inputs['use_cache'], 
        #     )
        if input_ids.numel() > 0 and input_ids[-1][-1] == self.config.im_start_token:
            assert input_ids.shape[0] == 1
            assert image_sizes is not None
            # generate image
            images = self.sample_and_process_images(image_sizes[-1][-1], inputs_embeds).to(inputs_embeds.dtype)
            _inputs['images'] = images
            _inputs['image_sizes'] = image_sizes
            # input_ids = torch.cat(
            #     [input_ids, torch.LongTensor([[IMAGE_TOKEN_INDEX, self.config.im_end_token]]).to(self.device)], 
            #     dim=1, 
            #     )
            input_ids = torch.cat(
                [input_ids, torch.LongTensor([[IMAGE_TOKEN_INDEX]]).to(self.device)], 
                dim=1, 
                )
            _inputs['input_ids'] = input_ids
            _inputs['task_types'] = ['gen']
        else:
            if images is not None:
                _inputs['images'] = images
            if image_sizes is not None:
                _inputs['image_sizes'] = image_sizes
            _inputs['task_types'] = None
        return _inputs



AutoConfig.register("univa_qwen2", UnivaConfig)
AutoModelForCausalLM.register(UnivaConfig, UnivaQwen2ForCausalLM)
