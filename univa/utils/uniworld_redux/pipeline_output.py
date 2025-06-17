from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image
import torch

from diffusers.utils import BaseOutput


@dataclass
class UniWorldPipelineOutput(BaseOutput):
    """
    Output class for Uniworld pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


@dataclass
class UniWorldPriorReduxPipelineOutput(BaseOutput):
    """
    Output class for UniWorld Prior Redux pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    height: int
    width: int
