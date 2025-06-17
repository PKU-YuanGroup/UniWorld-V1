from typing import List, Dict
from transformers import PreTrainedTokenizer
import torch
import torch.nn.functional as F
import sys
from univa.utils.constant import SPACIAL_TOKEN

def pad_list_of_tensors(tensor_list, padding_value=0):
    # tensor_list: list of tensors, each of shape (b, c, h, w)

    # if all empty list, which means all data are t2i within this batch
    if all(not isinstance(tensor, torch.Tensor) for tensor in tensor_list):
        return []
    else:
        for tmp_tensor in tensor_list:
            if isinstance(tmp_tensor, torch.Tensor):
                # find a tensor
                break
        # this line pad zero_tensor when batch mixed between t2i and others.
        # t2i can be considered a uncondition (no-reference image) editing
        tensor_list = [
            torch.zeros_like(tmp_tensor) if isinstance(tensor, list) else tensor for tensor in tensor_list
            ]
    assert all(tensor.shape[1] == tensor_list[0].shape[1] for tensor in tensor_list)
    # 找到最大的 b, h, w
    max_b = max(tensor.shape[0] for tensor in tensor_list)
    max_c = tensor_list[0].shape[1]  # 假设c都是一样的
    max_h = max(tensor.shape[2] for tensor in tensor_list)
    max_w = max(tensor.shape[3] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        b, c, h, w = tensor.shape
        pad_b = max_b - b
        pad_h = max_h - h
        pad_w = max_w - w

        # 先 pad h, w (最后两维)
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=padding_value)
        # 再 pad b 维（最前面），要扩成 (max_b, c, h, w)
        if pad_b > 0:
            padding_shape = (pad_b, c, max_h, max_w)
            pad_tensor = torch.full(padding_shape, fill_value=padding_value, dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, pad_tensor], dim=0)

        padded_tensors.append(tensor)

    # 最后 stack 成 (B, b_max, c, h_max, w_max)
    return torch.stack(padded_tensors)

def resize_list_of_tensors(weights):
    # suppose weights is your list of [1, H, W] tensors
    # 1) find the max height and width
    heights = [w.shape[-2] for w in weights]
    widths  = [w.shape[-1] for w in weights]
    max_h, max_w = max(heights), max(widths)

    # 2) interpolate each mask to (max_h, max_w)
    resized = []
    for w in weights:
        # F.interpolate expects a 4D tensor: (N, C, H, W)
        w_4d = w.unsqueeze(0)             # -> [1, 1, H, W]
        w_4d = w_4d.unsqueeze(0) if w_4d.ndim == 3 else w_4d
        # but since w is already [1,H,W], unsqueeze once is enough:
        # w_4d = w.unsqueeze(0) # [1, 1, H, W]

        w_resized = F.interpolate(
            w_4d, size=(max_h, max_w), mode='nearest'
        )
        # back to [1, H', W']
        w_resized = w_resized.squeeze(0)
        resized.append(w_resized)

    # 3) stack into a single tensor [N, 1, max_h, max_w]
    weights = torch.stack(resized)  # -> [N, 1, max_h, max_w]
    return weights

class DataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, padding_side='right'):
        self.tokenizer = tokenizer
        self.padding_side = padding_side

    def __call__(self, instances: List[Dict]) -> Dict:
        input_ids = [instance["input_ids"][0] for instance in instances]
        labels = [instance["labels"][0] for instance in instances]
        image_position = [instance["image_position"] for instance in instances]

        pixel_values = [
            instance["pixel_values"] for instance in instances if len(instance["pixel_values"]) > 0
        ]
        pixel_values = torch.cat(pixel_values) if len(pixel_values) > 0 else None

        image_grid_thw = [
            instance["image_grid_thw"] for instance in instances if len(instance["image_grid_thw"]) > 0
        ]
        image_grid_thw = torch.cat(image_grid_thw) if len(image_grid_thw) > 0 else None

        pil_pixel_values = [
            instance["pil_pixel_values"] for instance in instances
        ]

        prompts = [instance["prompt"] for instance in instances]

        ref_pixel_values = [
            instance["ref_pixel_values"] for instance in instances
        ]
        ref_pixel_values = pad_list_of_tensors(ref_pixel_values, padding_value=0)

        siglip_pixel_values = [
            instance["siglip_pixel_values"] for instance in instances if len(instance["siglip_pixel_values"]) > 0
        ]
        siglip_pixel_values = torch.cat(siglip_pixel_values, dim=0) if len(siglip_pixel_values) > 0 else []

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id, 
            padding_side=self.padding_side, 
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100, 
            padding_side=self.padding_side, 
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        weights = [
            instance["weights"] for instance in instances if len(instance["weights"]) > 0
        ]
        if len(weights) > 0:
            if all([i.shape == weights[0].shape for i in weights]):
                weights = torch.stack(weights)  
            else:
                weights = [i.unsqueeze(0) for i in weights]
        else:
            weights = None

        generated_image = [
            instance["generated_image"] for instance in instances if len(instance["generated_image"]) > 0
            ]
        if len(generated_image) > 0:
            if all([i.shape == generated_image[0].shape for i in generated_image]):
                generated_image = torch.stack(generated_image)  
            else:
                generated_image = [i.unsqueeze(0) for i in generated_image]
        else:
            generated_image = []
        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "labels": labels,
            "attention_mask": attention_mask,
            "image_position": image_position,
            "image_grid_thw": image_grid_thw, 
            "prompts": prompts, 
            "ref_pixel_values": ref_pixel_values, 
            "pil_pixel_values": pil_pixel_values, 
            "siglip_pixel_values": siglip_pixel_values, 
            "weights": weights, 
            "generated_image": generated_image, 
        }



class DataCollator_V1_1(DataCollator):
    def __init__(self, dataset_type: str, tokenizer: PreTrainedTokenizer, padding_side='right'):
        super().__init__(tokenizer=tokenizer, padding_side=padding_side)
        image_begin_token = SPACIAL_TOKEN[dataset_type]['image_begin_token']
        self.image_begin_token_id = tokenizer.convert_tokens_to_ids(image_begin_token)

    def __call__(self, *args, **kwargs) -> Dict:
        batch = super().__call__(*args, **kwargs)
        labels = batch.pop('labels')

        mask_image_begin = (labels == self.image_begin_token_id)
        # print("mask_image_begin:", mask_image_begin)  
        # idxs_image = mask_image_begin.nonzero(as_tuple=True)
        # print("positions of image_begin_token:", list(zip(*idxs_image)))
        labels[mask_image_begin] = -100

        batch.update({'labels': labels})
        return batch
