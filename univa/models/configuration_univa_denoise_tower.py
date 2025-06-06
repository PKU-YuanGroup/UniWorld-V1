from transformers.configuration_utils import PretrainedConfig
from typing import Literal, Optional, Union
import json


class UnivaDenoiseTowerConfig(PretrainedConfig):
    model_type = "univa_denoise_tower"

    def __init__(
        self,
        denoiser_type: Literal["flux", "sd3"] = "flux",
        denoise_projector_type: str = "mlp2x_gelu",
        vae_projector_type: str = "mlp2x_gelu",
        input_hidden_size: int = 1152,
        vae_input_hidden_size: int = 64,
        output_hidden_size: int = 4096,
        denoiser_config: Optional[Union[str, dict]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._attn_implementation_autoset = True

        self.denoiser_type = denoiser_type
        self.denoise_projector_type = denoise_projector_type
        self.vae_projector_type = vae_projector_type
        self.input_hidden_size = input_hidden_size
        self.vae_input_hidden_size = vae_input_hidden_size
        self.output_hidden_size = output_hidden_size
        self.denoiser_config = denoiser_config

        if isinstance(denoiser_config, str):
            with open(denoiser_config, "r") as f:
                self.denoiser_config = json.load(f)
        else:
            self.denoiser_config = denoiser_config
