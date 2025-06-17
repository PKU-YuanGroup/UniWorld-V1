from typing import Any, Callable, Optional, List

import torch
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import os
from PIL import Image
from univa.utils.prompter import Prompter
import numpy as np
from einops import rearrange
import random
# from qwen_vl_utils.vision_process import fetch_image, fetch_video
from qwen_vl_utils.vision_process import to_rgb, smart_resize, fetch_video
from univa.utils.constant import SPACIAL_TOKEN, GENERATE_TOKEN
from univa.utils.get_mask import get_weight_mask
from univa.utils.get_ocr import get_ocr_result
from fractions import Fraction
from torchvision.transforms import functional
from torchvision import transforms
from io import BytesIO
import base64
import requests
import torch
from PIL import Image
from torchvision import io, transforms
from typing import Optional


def get_aspect_ratio(img):
    width, height = img.size
    return Fraction(width, height).limit_denominator()

def has_same_aspect_ratio(img1, img2):
    if not isinstance(img1, Image.Image):
        img1 = Image.open(img1).convert('RGB')
    if not isinstance(img2, Image.Image):
        img2 = Image.open(img2).convert('RGB')
    ratio1 = get_aspect_ratio(img1)
    ratio2 = get_aspect_ratio(img2)
    return ratio1 == ratio2

def has_same_resolution(img1, img2):
    if not isinstance(img1, Image.Image):
        img1 = Image.open(img1).convert('RGB')
    if not isinstance(img2, Image.Image):
        img2 = Image.open(img2).convert('RGB')
    return img1.size == img2.size

class Qwen2p5VLDataset(Dataset):
    def __init__(
        self,
        dataset_type: str,
        data_txt: str,
        transform: Callable, 
        tokenizer: PreTrainedTokenizer,
        prompter: Prompter,
        image_processor: Callable,
        processor: Callable = None,
        min_pixels: int = 384*384, 
        max_pixels: int = 384*384, 
        image_token_length: int = 729,
        only_generated_task: bool = False,
        drop_prompt_rate: float = 0.0,
        joint_ref_feature: bool = False,
        anyres: bool = False, 
        mask_weight_type: str = 'log', 
        siglip_processor: Callable = None,
        ocr_enhancer: bool = False, 
        random_data: bool = False, 
        maxnum_per_data: int = -1, 
        notry: bool = False, 
    ):
        assert dataset_type == 'qwen2vl' or dataset_type == 'qwen2p5vl', "dataset_type == 'qwen2vl' or dataset_type == 'qwen2p5vl'"
        with open(data_txt, "r") as f:
            self.datasets = [line.strip() for line in f.readlines()]

        self.data = []
        self._load_data(maxnum_per_data)
        
        self.transform = transform
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.prompter = prompter
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_token = SPACIAL_TOKEN[dataset_type]['image_token']
        self.image_begin_token = SPACIAL_TOKEN[dataset_type]['image_begin_token']
        self.image_end_token = SPACIAL_TOKEN[dataset_type]['image_end_token']
        self.generated_image_token = GENERATE_TOKEN
        self.image_processor = processor.image_processor
        # self.factor = 4 if joint_ref_feature else 1
        self.factor = 2

        self.only_generated_task = only_generated_task  # For denoiser training
        self.drop_prompt_rate = drop_prompt_rate
        if self.drop_prompt_rate > 0:
            assert self.only_generated_task, (
                "Only generated task is supported when drop_prompt_rate > 0"
            )
        self.mask_weight_type = mask_weight_type
        self.siglip_processor = siglip_processor
        self.ocr_enhancer = ocr_enhancer
        self.random_data = random_data
        self.notry = notry

        # Add image token if not exists.
        assert self.image_token in self.tokenizer.get_vocab()
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

        self.image_begin_token_id = self.tokenizer.convert_tokens_to_ids(
            self.image_begin_token
        )
        assert isinstance(self.image_begin_token_id, int), (
            f"tokenizer miss image begin token `{self.image_begin_token}`"
        )
        self.image_end_token_id = self.tokenizer.convert_tokens_to_ids(
            self.image_end_token
        )
        assert isinstance(self.image_end_token_id, int), (
            f"tokenizer miss image end token `{self.image_end_token}`"
        )

    def _load_data(self, maxnum_per_data=-1):
        for dataset in self.datasets:
            image_root, json_file, need_weight = dataset.split(",")

            # Load json file
            with open(json_file, "r") as f:
                data = json.load(f)
            if maxnum_per_data > 0 and maxnum_per_data < len(data):
                print(f'original data: {len(data)}, sample: {maxnum_per_data}')
                data = random.sample(data, maxnum_per_data)
            dataset_data = []
            for line in tqdm(data):
                if "image" not in line:
                    line["image"] = []
                # Ensure `image` is a list
                if isinstance(line["image"], str):
                    line["image"] = [line["image"]]
                assert isinstance(line["image"], list), (
                    "`image` must be a str or a list."
                )

                # Convert image path to absolute path
                line["need_weight"] = need_weight
                line["image"] = [
                    os.path.join(image_root, image_path) for image_path in line["image"]
                ]
                dataset_data.append(line)

            print(f"Load {len(dataset_data)} data from {json_file}.")
            self.data.extend(dataset_data)

    def __len__(self):
        return len(self.data)

    def _get_random_data(self, ):
        
        prompt = self.prompter(
            [
                {"from": "system", "value": "You are a helpful assistant."},
                {
                    "from": "user",
                    "value": f"test an image {self.image_token}",
                },
            ]
        )
        input_ids = self.tokenizer.batch_encode_plus(
            [prompt], return_tensors="pt", truncation=False,
        ).input_ids
        labels = input_ids

        width, height = 448, 448
        random_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        image = Image.fromarray(random_data, 'RGB')

        image_slice = [image]
        image_dict = self._load_image(
            image_slice, self.max_pixels, self.min_pixels, 
            processor=self.processor, image_token=self.image_token, 
            factor=self.factor, 
            last_image=image,
            vae_image_transform=self.transform, 
            drop_prompt=False, 
            prompt=prompt, 
            mask_weight_type=self.mask_weight_type, 
            siglip_processor=self.siglip_processor, 
            )
        
        image_token_lengths = image_dict['image_token_lengths']
        pixel_values = image_dict['pixel_values']
        image_grid_thw = image_dict['image_grid_thw']
        ref_pixel_values = image_dict['ref_pixel_values']
        pil_pixel_values = image_dict['pil_pixel_values']
        siglip_pixel_values = image_dict['siglip_pixel_values']
        weights = image_dict['weights']

        input_ids, labels, image_position = self._process_image_token(
                input_ids,
                labels=labels,
                image_token_id=self.image_token_id,
                image_begin_token_id=self.image_begin_token_id,
                image_end_token_id=self.image_end_token_id,
                image_token_lengths=image_token_lengths, 
            )
        
        generated_image = torch.randn(3, 512, 512)
        
        return_data = {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_position": image_position,
            "image_grid_thw": image_grid_thw, 
            "prompt": prompt,
            "ref_pixel_values": ref_pixel_values, 
            "pil_pixel_values": pil_pixel_values, 
            "siglip_pixel_values": siglip_pixel_values, 
            "weights": weights, 
            "generated_image": generated_image, 
        }
        return return_data

    def getitem(self, data):
        # Reformat the conversation to the format of prompter
        conversations = []
        prompt = ""
        for item in data["conversations"]:
            if item["from"] == "human":
                role = self.prompter.user_role
                prompt = item["value"]
            elif item["from"] == "gpt":
                role = self.prompter.assistant_role
            else:
                raise ValueError(f"Unknown role: {item['from']}")
            conversations.append({"from": role, "value": item["value"]})
        assert prompt != "", "prompt != ''"
        # The last turn instruction will be used for t5_embed
        prompt = prompt.replace('<image>', '').replace('\n', '')

        # Make prompt
        drop_prompt = False
        if self.only_generated_task:
            if self.drop_prompt_rate < random.random():  # Randomly drop the prompt
                prompt_list = self.prompter.get_train_prompt(conversations)
            else:
                drop_prompt = True
                num_images = (''.join([i['value'] for i in conversations])).count('<image>')
                # Drop the prompt
                prompt_list = [
                    {
                        "from": self.prompter.system_role,
                        "value": "You are a helpful assistant.",
                    },
                    {
                        "from": self.prompter.user_role,
                        # "value": f"{num_images * '<image>'} Generate an image.",
                        "value": "Generate an image.",
                    },
                    {
                        "from": self.prompter.assistant_role,
                        "value": self.generated_image_token,
                    },
                ]
                prompt_list = self.prompter.get_train_prompt(prompt_list)
        else:
            prompt_list = self.prompter.get_train_prompt(conversations)
            
        input_ids = []
        labels = []
        has_generated_image = False
        cur_i = 0
        for item in prompt_list:
            item["prompt"] = item["prompt"].replace('<image>', self.image_token)
            
            if self.generated_image_token in item["prompt"]:  # Check if self.generated_image_token in prompt
                assert item["from"] == self.prompter.assistant_role, (
                    "Generated image token must be in assistant role"
                )
                assert (
                    f"{self.generated_image_token}{self.prompter.eos_token}"
                    in item["prompt"]
                ), "Generated image token must in end of prompt"

                # Replace the generated image token with image begin token and without eos token
                item["prompt"] = item["prompt"].replace(
                    f"{self.generated_image_token}{self.prompter.eos_token}",
                    self.image_begin_token,
                )
                has_generated_image = True

            if self.ocr_enhancer and (self.image_token in item["prompt"]):
                # print('item["prompt"]', item["prompt"])
                if not has_generated_image:
                    num_img = item["prompt"].count(self.image_token)
                    ocr_sentences = []
                    for i in range(num_img):
                        ocr_sentences.append(get_ocr_result(data["image"][cur_i], cur_i))
                        cur_i += 1
                    ocr_sentences = '\n'.join(ocr_sentences)
                    if len(ocr_sentences.split()) > 256:
                        print(f'ocr_sentences too long, total len {len(ocr_sentences.split())} trunk first 256')
                        ocr_sentences = ' '.join(ocr_sentences.split()[:256])
                    # ocr_sentences = ''
                    assert item['prompt'][-len(self.prompter.eos_token):] == self.prompter.eos_token, \
                        "item['prompt'][-len(self.prompter.eos_token):] == self.prompter.eos_token"
                    assert item['prompt'].count(self.prompter.eos_token) == 1, \
                        "item['prompt'].count(self.prompter.eos_token) == 1"
                    item["prompt"] = item["prompt"].replace(self.prompter.eos_token, f'{ocr_sentences} {self.prompter.eos_token}')

            tokenized_item = self.tokenizer(
                item["prompt"],
                return_tensors="pt",
                truncation=True,
                max_length=1024, 
            )
            if item["is_labels"]:  # If this prompt is labels
                labels.append(tokenized_item.input_ids)
            else:
                labels.append(torch.full_like(tokenized_item.input_ids, -100))
            input_ids.append(tokenized_item.input_ids)

        if (
            self.only_generated_task and not has_generated_image
        ):  # For denoiser training
            raise ValueError(
                f"Only generated task is not supported. But this prompt not contains generated image token: {prompt_list[0]['prompt']}"
            )

        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)

        # Load images
        if has_generated_image:
            # generate task
            # process images but exclude the last image, which need to generate
            image_slice = data["image"][:-1]
        else:
            # understanding task
            image_slice = data["image"]


        image_dict = self._load_image(
            image_slice, self.max_pixels, self.min_pixels, 
            processor=self.processor, image_token=self.image_token, 
            factor=self.factor, 
            last_image=data["image"][-1] if has_generated_image else None,
            vae_image_transform=self.transform, 
            drop_prompt=drop_prompt, 
            prompt=prompt, 
            mask_weight_type=self.mask_weight_type, 
            siglip_processor=self.siglip_processor, 
            need_weight=data['need_weight'], 
            )
        
        image_token_lengths = image_dict['image_token_lengths']
        pixel_values = image_dict['pixel_values']
        image_grid_thw = image_dict['image_grid_thw']
        ref_pixel_values = image_dict['ref_pixel_values']
        pil_pixel_values = image_dict['pil_pixel_values']
        siglip_pixel_values = image_dict['siglip_pixel_values']
        weights = image_dict['weights']

        input_ids, labels, image_position = self._process_image_token(
            input_ids,
            labels=labels,
            image_token_id=self.image_token_id,
            image_begin_token_id=self.image_begin_token_id,
            image_end_token_id=self.image_end_token_id,
            image_token_lengths=image_token_lengths, 
        )


        return_data = {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_position": image_position,
            "image_grid_thw": image_grid_thw, 
            "prompt": prompt,
            "ref_pixel_values": ref_pixel_values, 
            "pil_pixel_values": pil_pixel_values, 
            "siglip_pixel_values": siglip_pixel_values, 
            "weights": weights, 
        }

        if has_generated_image: # If this item is a generation task
            image = Image.open(data["image"][-1]).convert("RGB")
            # if self.anyres:
            #     image = image.resize(pil_pixel_values[-1].size)
            image_tensor = torch.tensor(np.array(image)) / 255.0  # scale to 0-1
            image_tensor = rearrange(image_tensor, "h w c -> c h w")
            return_data["generated_image"] = self.transform(image_tensor)
        else:
            return_data["generated_image"] = []
        return return_data
    
    def __getitem__(self, idx):
        if self.random_data:
            return self._get_random_data()
        
        data: Any = self.data[idx]
        if self.notry:
            return self.getitem(data)
        try:
            return self.getitem(data)
        except Exception as e:
            print(f'Error with {e}')
            return self.__getitem__(random.randint(0, self.__len__()-1))

    @staticmethod
    def _load_image(
        image_slice: List[str],
        max_pixels: int = 448*448,  
        min_pixels: int = 448*448, 
        processor: Callable = None, 
        image_processor: Callable = None, 
        image_token_lengths: int = 729, 
        image_token: str = '<|image_pad|>', 
        factor: int = 1, 
        last_image: Optional[str] = None, 
        vae_image_transform: Callable = None,
        drop_prompt: bool = False, 
        prompt: str = '', 
        mask_weight_type: str = None, 
        siglip_processor: Callable = None, 
        need_weight: str = 'true', 
    ):
        resize_ref_image = False
        pil_pixel_values_last = []
        if last_image is not None:
            last_vision_infos = dict(
                image=last_image, min_pixels=min_pixels, max_pixels=max_pixels
                )
            # last_image will be resize by qwenvl-processor automatically
            # generated variable resolution
            last_image_inputs, last_video_inputs = process_vision_info([last_vision_infos], factor=factor)

            # logging what size will be process when use qwenvl-processor
            pil_pixel_values_last.append(last_image_inputs[0])
            
            # not all reference images are same resolution
            # if multiple reference images and they have different resolution, resize it depend on last_image (generated_image)
            if not all([has_same_resolution(image_path, last_image) for image_path in image_slice]):
                resize_ref_image = True
                resize_w, resize_h = last_image_inputs[0].size

        image_token_lengths = []
        pixel_values = []
        image_grid_thw = []
        ref_pixel_values = []
        pil_pixel_values = []
        siglip_pixel_values = []
        # Ignore the last image (generated image)
        for image_path in image_slice: 
            vision_infos = dict(image=image_path, min_pixels=min_pixels, max_pixels=max_pixels)
            
            # if multiple reference images and they have different aspect ratio, resize it depend on generated_image (last_image)
            if resize_ref_image:
                vision_infos.update(
                    dict(resized_height=resize_h, resized_width=resize_w)
                    )
            image_inputs, video_inputs = process_vision_info([vision_infos], factor=factor)
            inputs = processor(text=[f'dummy {image_token}'], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            
            if not drop_prompt:
                pixel_values.append(inputs.pixel_values)  # inputs.pixel_values shape is (token, dim)
                image_grid_thw.append(inputs.image_grid_thw)  # image_grid_thw List[int, int, int]
                image_token_length = (inputs.input_ids[0] == processor.tokenizer.convert_tokens_to_ids(image_token)).sum()
                image_token_lengths.append(image_token_length)

            image_tensor = torch.tensor(np.array(image_inputs[0])) / 255.0  # scale to 0-1
            image_tensor = rearrange(image_tensor, "h w c -> 1 c h w")
            if vae_image_transform is not None:
                # image_tensor has been resized by qwenvl-processor
                image_tensor = (image_tensor - 0.5) / 0.5  # shift [0, 1] to [-1, 1]
            pil_pixel_values.append(image_inputs[0])

            if siglip_processor is not None:
                siglip_pixel_value = siglip_processor.preprocess(
                            images=Image.open(image_path).convert('RGB') if isinstance(image_path, str) else image_path, 
                            do_resize=True, return_tensors="pt", do_convert_rgb=True
                        ).pixel_values  # 1 c h w
                if drop_prompt:
                    siglip_pixel_values.append(torch.zeros_like(siglip_pixel_value))
                else:
                    siglip_pixel_values.append(siglip_pixel_value)
            # use zero_image as uncondition reference image
            if drop_prompt:
                ref_pixel_values.append(torch.zeros_like(image_tensor))
            else:
                ref_pixel_values.append(image_tensor)
            

            
        # if multi-image in a sample, concat them
        # assume pixel_values[0] (n1, 1176), pixel_values[1] (n2, 1176), pixel_values will be (n1+n2, 1176)
        if len(pixel_values) > 0:
            pixel_values = torch.concat(pixel_values)
            image_grid_thw = torch.concat(image_grid_thw)  # (b, 3), 3 mean the grid of t, h, w
        # if len(ref_pixel_values) > 0: 
        #     ref_pixel_values = torch.concat(ref_pixel_values)  # b c h w
        ref_pixel_values = []
        if len(siglip_pixel_values) > 0: 
            siglip_pixel_values = torch.concat(siglip_pixel_values)  # b c h w

        pil_pixel_values = pil_pixel_values + pil_pixel_values_last
        
        if mask_weight_type is not None:
            _, weights = get_weight_mask(pil_pixel_values, prompt, mask_weight_type, need_weight)
            if need_weight.lower() == 'false':
                assert torch.all(weights == 1)
        else:
            weights = []
        return {
            'pixel_values': pixel_values, 
            'image_grid_thw': image_grid_thw, 
            'image_token_lengths': image_token_lengths, 
            'ref_pixel_values': ref_pixel_values, 
            'pil_pixel_values': pil_pixel_values, 
            'siglip_pixel_values': siglip_pixel_values, 
            'weights': weights, 
            }

    @staticmethod
    def _process_image_token(
        input_ids: torch.Tensor,
        image_token_id: int,
        image_begin_token_id: int,
        image_end_token_id: int,
        image_token_lengths: List[int],
        labels: Optional[torch.Tensor] = None,
    ):
        # Find the indices of the image token
        image_token_indices = (input_ids == image_token_id).nonzero(as_tuple=True)
        # assert len(image_token_lengths) == image_token_indices[1].numel()
        image_position = []
        offset = 0
        cur_i = 0
        if isinstance(image_token_lengths, int):
            image_token_lengths = [image_token_lengths] * len(image_token_indices[1])
        for idx in image_token_indices[1]:
            image_token_length = image_token_lengths[cur_i]
            adjusted_idx = idx + offset
            assert input_ids[0, adjusted_idx] == image_token_id, "assert input_ids[0, adjusted_idx] == image_token_id"

            # Add image begin and end token
            input_ids = torch.cat(
                [
                    input_ids[:, :adjusted_idx],
                    input_ids.new_full(
                        (1, 1), image_begin_token_id
                    ),  # image begin token
                    input_ids.new_full(
                        (1, image_token_length), image_token_id
                    ),  # Repeat the image token to the length of image_token_length
                    input_ids.new_full((1, 1), image_end_token_id),  # image end token
                    input_ids[:, adjusted_idx + 1 :],
                ],
                dim=1,
            )
            if labels is not None:
                labels = torch.cat(
                    [
                        labels[:, :adjusted_idx],
                        labels.new_full(
                            (1, 1), image_begin_token_id
                        ),  # Make begin token as label
                        labels.new_full((1, image_token_length), -100),
                        labels.new_full((1, 1), -100),
                        labels[:, adjusted_idx + 1 :],
                    ],
                    dim=1,
                )

            adjusted_idx += 1  # skip the image begin token
            image_position.append(adjusted_idx.item())
            offset += image_token_length - 1
            offset += 2  # begin and end token

            cur_i += 1

        return input_ids, labels, image_position
    

def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = 28) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        response = requests.get(image, stream=True)
        image_obj = Image.open(BytesIO(response.content))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj)
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels")
        max_pixels = ele.get("max_pixels")
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height), resample=Image.Resampling.BICUBIC)

    return image

def process_vision_info(
    vision_infos: list,
    return_video_kwargs: bool = False,
    factor: int = 1, 
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info, size_factor=28*factor))
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs