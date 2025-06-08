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


class ReduxDataset(Dataset):
    def __init__(
        self,
        data_txt: str,
        image_processors: List[Callable],
        image_transform: Callable,
    ):
        self.image_processors = image_processors
        self.image_transform = image_transform
        with open(data_txt, "r") as f:
            self.datasets = [line.strip() for line in f.readlines()]

        self.data = []
        self._load_data()

    def _load_data(self):
        for dataset in self.datasets:
            image_root, json_file = dataset.split(",")

            # Load json file
            with open(json_file, "r") as f:
                data = json.load(f)

            dataset_data = []
            for line in tqdm(data):
                # Ensure `image` is a list
                if isinstance(line["image"], str):
                    line["image"] = [line["image"]]
                assert isinstance(
                    line["image"], list
                ), "`image` must be a str or a list."

                # Convert image path to absolute path
                line["image"] = [
                    os.path.join(image_root, image_path) for image_path in line["image"]
                ]

                dataset_data.append(line)

            print(f"Load {len(dataset_data)} data from {json_file}.")
            self.data.extend(dataset_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data: Any = self.data[idx]

        # Reformat the conversation to the format of prompter
        has_generated_image = False
        for item in data["conversations"]:
            if "<gen_image>" in item["value"]:
                has_generated_image = True
        assert has_generated_image, "No generated image found in the conversation."

        result = {
            f"image_{image_processor_idx}": []
            for image_processor_idx in range(len(self.image_processors))
        }

        for image_path in data["image"][:-1]:
            image = Image.open(image_path).convert("RGB")
            for image_processor_idx, image_processor in enumerate(self.image_processors):
                image_encoder_image = image_processor(
                    image, return_tensors="pt", do_resize=True, do_center_crop=True
                ).pixel_values.squeeze(0)
                result[f"image_{image_processor_idx}"].append(image_encoder_image)

        generated_image = Image.open(data["image"][-1]).convert("RGB")
        generated_image_tensor = torch.tensor(np.array(generated_image)) / 255.0  # scale to 0-1
        generated_image_tensor = rearrange(generated_image_tensor, "h w c -> c h w")
        generated_image_tensor = self.image_transform(generated_image_tensor)
        result["generated_image"] = generated_image_tensor
        
        return result

