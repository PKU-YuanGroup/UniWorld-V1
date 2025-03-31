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
#    --------------------------------------------------------
import sys
from typing import Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput
from einops import rearrange

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector, build_inv_projector, build_eye_projector, build_mask_projector
from .pixel_decoder.builder import build_pixel_decoder

from .multimodal_denoiser.builder import build_denoise_tower
from .multimodal_projector.builder import build_denoise_projector

from univa.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)


class UnivaMetaModel:

    def __init__(self, config):
        super(UnivaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=False)
            self.mm_projector = build_vision_projector(config)
            
        if getattr(config, "mm_denoise_tower", None) is not None:
            self.diffusion_tower = build_denoise_tower(config, delay_load=False)
            self.mm_denoise_projector = build_denoise_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def get_denoise_tower(self):
        denoise_tower = getattr(self, 'denoise_tower', None)
        if type(denoise_tower) is list:
            denoise_tower = denoise_tower[0]
        return denoise_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        # for pixel_decoder
        mm_pixel_decoder = model_args.mm_pixel_decoder
        pretrain_mm_inv_mlp_adapter = model_args.pretrain_mm_inv_mlp_adapter
        self.config.mm_pixel_decoder = mm_pixel_decoder

        # for pretrain_eye_adapter
        mm_eye_adapter = model_args.mm_eye_adapter
        pretrain_mm_eye_mlp_adapter = model_args.pretrain_mm_eye_mlp_adapter
        self.config.mm_eye_adapter = mm_eye_adapter

        # for pretrain_mask_adapter
        mm_mask_adapter = model_args.mm_mask_adapter
        pretrain_mm_mask_mlp_adapter = model_args.pretrain_mm_mask_mlp_adapter
        self.config.mm_mask_adapter = mm_mask_adapter

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()
        # self.image_embed_len = (self.vision_tower.config.image_size // self.vision_tower.config.patch_size) ** 2
        self.image_embed_len = (self.vision_tower.image_size() // self.vision_tower.patch_size()) ** 2
        self.config.image_embed_len = self.image_embed_len
        self.config.image_mean = self.vision_tower.image_processor.image_mean
        self.config.image_std = self.vision_tower.image_processor.image_std
        self.config.decode_image_size = self.vision_tower.image_size() // self.vision_tower.patch_size() * 16  # 336 -> 384; 384 -> 432
        self.config.mm_patch_size = self.vision_tower.patch_size()

        # for convnext
        if 'conv' in model_args.vision_tower.lower():
            self.config.mm_vision_resolution = model_args.mm_vision_resolution
        
        self.config.mm_anyres = model_args.mm_anyres
        self.config.mm_anyres_min_pixels = model_args.mm_anyres_min_pixels
        self.config.mm_anyres_max_pixels = model_args.mm_anyres_max_pixels

        ## build eye projector
        self.config.mm_eye_model_path = model_args.mm_eye_model_path
        self.config.mm_eye_depth = model_args.mm_eye_depth
        self.config.mm_eye_weight = model_args.mm_eye_weight

        ## build mask projector
        self.config.mm_mask_ratio = model_args.mm_mask_ratio
        self.config.mm_mask_depth = model_args.mm_mask_depth
        self.config.mm_mask_weight = model_args.mm_mask_weight
        if mm_mask_adapter:
            self.image_embed_len = int(self.image_embed_len * model_args.mm_mask_ratio)

        ### build CLIP-LLM projector
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            print(f"=> loading pretrain_mm_mlp_adapter from {pretrain_mm_mlp_adapter} ...")
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


        denoise_tower = getattr(model_args, "denoise_tower", None)
        if denoise_tower is not None:
            self.config.mm_denoise_tower = denoise_tower
            pretrain_mm_denoise_mlp_adapter = model_args.pretrain_mm_denoise_mlp_adapter

        if denoise_tower is not None:
            if self.get_denoise_tower() is None:
                denoise_tower = build_denoise_tower(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.denoise_tower = [denoise_tower]
                else:
                    self.denoise_tower = denoise_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    denoise_tower = self.denoise_tower[0]
                else:
                    denoise_tower = self.denoise_tower
                denoise_tower.load_model()

            if getattr(self, 'mm_denoise_projector', None) is None:
                self.mm_denoise_projector = build_denoise_projector(self.config)
            else:
                # In case it is frozen by LoRA
                for p in self.mm_denoise_projector.parameters():
                    p.requires_grad = True

            if pretrain_mm_denoise_mlp_adapter is not None:
                print(f"=> loading pretrain_mm_denoise_mlp_adapter from {pretrain_mm_denoise_mlp_adapter} ...")
                mm_denoise_projector_weights = torch.load(pretrain_mm_denoise_mlp_adapter, map_location='cpu')

                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

                self.mm_denoise_projector.load_state_dict(get_w(mm_denoise_projector_weights, 'mm_denoise_projector'))


        self.config.eye_enable = False
        if getattr(model_args, 'mm_eye_adapter', False):
            self.config.eye_enable = True
            self.config.use_mm_eye_proj = True
            self.config.mm_eye_projector_type = getattr(model_args, 'mm_eye_projector_type', 'linear')
            if getattr(self, 'mm_eye_projector', None) is None:
                self.mm_eye_projector = build_eye_projector(self.config)
            else:
                # In case it is frozen by LoRA
                for p in self.mm_eye_projector.parameters():
                    p.requires_grad = True

            if pretrain_mm_eye_mlp_adapter is not None:
                print(f"=> loading pretrain_mm_eye_mlp_adapter from {pretrain_mm_eye_mlp_adapter} ...")
                mm_eye_projector_weights = torch.load(pretrain_mm_eye_mlp_adapter, map_location='cpu')

                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

                self.mm_eye_projector.load_state_dict(get_w(mm_eye_projector_weights, 'mm_eye_projector'))

        self.config.mask_enable = False
        if getattr(model_args, 'mm_mask_adapter', False):
            self.config.mask_enable = True
            self.config.use_mm_mask_proj = True
            self.config.mm_mask_projector_type = getattr(model_args, 'mm_mask_projector_type', 'linear')
            if getattr(self, 'mm_mask_projector', None) is None:
                self.mm_mask_projector = build_mask_projector(self.config)
            else:
                # In case it is frozen by LoRA
                for p in self.mm_mask_projector.parameters():
                    p.requires_grad = True

            if pretrain_mm_mask_mlp_adapter is not None:
                print(f"=> loading pretrain_mm_mask_mlp_adapter from {pretrain_mm_mask_mlp_adapter} ...")
                mm_mask_projector_weights = torch.load(pretrain_mm_mask_mlp_adapter, map_location='cpu')

                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

                self.mm_mask_projector.load_state_dict(get_w(mm_mask_projector_weights, 'mm_mask_projector'))

        self.config.ross_enable = False
        if getattr(model_args, 'mm_pixel_decoder', False):
            self.config.ross_enable = True
            ### build pixel decoder
            self.pixel_decoder = build_pixel_decoder(self.config)
            self.config.mm_inv_hidden_size = self.pixel_decoder.latent_dim

            ### build LLM-CLIP projector
            self.config.use_mm_inv_proj = True
            self.config.mm_inv_projector_type = getattr(model_args, 'mm_inv_projector_type', 'linear')

            if getattr(self, 'mm_inv_projector', None) is None:
                self.mm_inv_projector = build_inv_projector(self.config)
            else:
                # In case it is frozen by LoRA
                for p in self.mm_inv_projector.parameters():
                    p.requires_grad = True
            if pretrain_mm_inv_mlp_adapter is not None:
                print(f"=> loading pretrain_mm_inv_mlp_adapter from {pretrain_mm_inv_mlp_adapter} ...")
                mm_inv_projector_weights = torch.load(pretrain_mm_inv_mlp_adapter, map_location='cpu')

                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

                self.mm_inv_projector.load_state_dict(get_w(mm_inv_projector_weights, 'mm_inv_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


def count_consecutive_blocks(lst, v):
    in_block = False
    block_count = 0

    for value in lst:
        if value == v:
            if not in_block:
                # 开始了一个新的连续块
                in_block = True
                block_count += 1
        else:
            in_block = False

    return block_count


class UnivaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def get_denoise_tower(self):
        return self.get_model().get_denoise_tower()

    def encode_images(self, images):
        encoder_out = self.get_model().get_vision_tower()(images, return_cls_token=False)
        image_features = encoder_out.pop('image_features')
        if isinstance(image_features, list):
            image_features = [self.get_model().mm_projector(i)[0] for i in image_features]
        else:
            image_features = self.get_model().mm_projector(image_features)
        return image_features, encoder_out

    def decode_images(self, conditions):
        conditions = self.get_model().mm_denoise_projector(conditions)
        images = self.get_model().get_denoise_tower()(conditions)
        return images

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, cache_position=None,
    ):  
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None, cache_position, None

        image_features, encoder_out = self.encode_images(images)
        
        
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in
                     zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        boi_ids, eoi_ids = [[] for _ in range(len(input_ids))], [[] for _ in range(len(input_ids))]
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # images have been normalized by CLIP
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_position = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            # assert len(image_position) == 1

            for i in range(num_images):
                boi_ids[batch_idx].append(image_position[i])
                eoi_ids[batch_idx].append(image_position[i] + image_features[cur_image_idx+i].shape[0] - 1)

            image_token_indices = [-1] + image_position + [cur_input_ids.shape[0]]

            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device,
                                   dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype,
                                       device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                              device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                             device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        cache_position = None if (_attention_mask is None or cache_position is None) else torch.arange(
            attention_mask.shape[1], device=attention_mask.device)

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, boi_ids, eoi_ids, cache_position, encoder_out
      
    def compute_vm_loss(
        self,
        images: torch.Tensor,
        hidden_states: torch.Tensor,
        boi_ids: torch.Tensor,
        eoi_ids: torch.Tensor,
        eps: float = 1e-6,
    ):
        batch_size = hidden_states.shape[0]
        vm_loss_mask = torch.zeros((batch_size,), device=hidden_states.device).bool()
        image_hidden_states = torch.zeros((batch_size, self.model.image_embed_len, hidden_states.shape[-1]),
                                          dtype=hidden_states.dtype,
                                          device=hidden_states.device)

        for batch_index, (cur_boi_id, cur_eoi_id, cur_hidden_state) in enumerate(zip(boi_ids, eoi_ids, hidden_states)):
            if (cur_boi_id is not None) and (cur_eoi_id is not None):
                assert cur_eoi_id - cur_boi_id + 1 == self.model.image_embed_len
                assert cur_hidden_state.shape[0] >= cur_eoi_id
                image_hidden_states[batch_index] = cur_hidden_state[cur_boi_id: cur_eoi_id + 1]
                vm_loss_mask[batch_index] = True

        images_std = torch.tensor(self.config.image_std, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
        images_mean = torch.tensor(self.config.image_mean, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
        images_vae = ((images * images_std + images_mean - 0.5) / 0.5).clamp(-1., 1.)
        images_vae = F.interpolate(images_vae, size=(self.config.decode_image_size, self.config.decode_image_size), mode='bilinear')

        with torch.no_grad():
            posterior = self.model.pixel_decoder.encode(images_vae).latent_dist
            z_q = (posterior.sample() - self.model.pixel_decoder.shift_factor) * self.model.pixel_decoder.scaling_factor
            # group each (2x2) window
            z_q = z_q.unfold(2, 2, 2).unfold(3, 2, 2)
            z_q = rearrange(z_q, 'b c h w p1 p2 -> b (c p1 p2) h w').contiguous()

        with torch.amp.autocast('cuda', dtype=torch.float32):
            # image_hidden_states = self.model.mm_inv_projector.ln_pre(
            #     image_hidden_states) + self.model.mm_inv_projector.pos_embed
            image_hidden_states = self.model.mm_inv_projector.ln_pre(image_hidden_states)
            h = w = int(image_hidden_states.shape[1] ** 0.5)
            image_hidden_states = rearrange(image_hidden_states, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
            vm_loss = self.model.mm_inv_projector(
                z=image_hidden_states.repeat(4, 1, 1, 1).contiguous().float(),
                target=z_q.repeat(4, 1, 1, 1).contiguous().float(),
            )
        vm_loss = vm_loss.float()
        vm_loss_mask = vm_loss_mask.repeat(4)
        vm_loss = (vm_loss.view(batch_size, -1).mean() * vm_loss_mask).sum() / (vm_loss_mask.sum() + eps)
        return vm_loss
    
    def compute_eye_loss(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if isinstance(self.config.mm_eye_depth, float):
            mm_eye_depth = int((len(hidden_states) - 1) * self.config.mm_eye_depth)
        else:
            mm_eye_depth = self.config.mm_eye_depth
        hidden_state = hidden_states[mm_eye_depth]
        logits = self.model.mm_eye_projector(hidden_state)

        eye_loss = self.model.loss_function(logits=logits, labels=labels, vocab_size=logits.shape[-1], **kwargs)
        return eye_loss
    
    def compute_mask_loss(
        self,
        images: torch.Tensor,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        ids_restore: torch.Tensor,
        boi_ids: torch.Tensor,
        eoi_ids: torch.Tensor,
        **kwargs,
    ):
        if isinstance(self.config.mm_mask_depth, float):
            mm_mask_depth = int((len(hidden_states) - 1) * self.config.mm_mask_depth)
        else:
            mm_mask_depth = self.config.mm_mask_depth
        hidden_states = hidden_states[mm_mask_depth]

        batch_size = hidden_states.shape[0]
        mask_loss_mask = torch.zeros((batch_size,), device=hidden_states.device).bool()
        image_hidden_states = torch.zeros((batch_size, self.model.image_embed_len, hidden_states.shape[-1]),
                                          dtype=hidden_states.dtype,
                                          device=hidden_states.device)

        for batch_index, (cur_boi_id, cur_eoi_id, cur_hidden_state) in enumerate(zip(boi_ids, eoi_ids, hidden_states)):
            if (cur_boi_id is not None) and (cur_eoi_id is not None):
                assert cur_eoi_id - cur_boi_id + 1 == self.model.image_embed_len
                assert cur_hidden_state.shape[0] >= cur_eoi_id
                image_hidden_states[batch_index] = cur_hidden_state[cur_boi_id: cur_eoi_id + 1]
                mask_loss_mask[batch_index] = True

        pred = self.model.mm_mask_projector(image_hidden_states, ids_restore)  # [N, L, p*p*3]
        target = self.model.mm_mask_projector.patchify(images)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5

        mask_loss = (pred - target) ** 2
        mask_loss = mask_loss.mean(dim=-1)  # [N, L], mean loss per patch
        mask = mask * mask_loss_mask[:, None]
        mask_loss = (mask_loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return mask_loss

@dataclass
class CausalLMOutputWithPastWithVM(ModelOutput):
    lm_loss: Optional[torch.FloatTensor] = None
    vm_loss: Optional[torch.FloatTensor] = None
    eye_loss: Optional[torch.FloatTensor] = None
    mask_loss: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
