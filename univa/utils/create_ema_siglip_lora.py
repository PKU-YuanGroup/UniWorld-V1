import os
import json
from copy import deepcopy
import numpy as np
import torch
from diffusers.training_utils import EMAModel, compute_snr
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import SiglipVisionModel

# Adapted from diffusers-style ema https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py#L263
class EMAModel_SigLIP_LoRA(EMAModel):
    """
    Exponential Moving Average of models weights
    """
    def __init__(
        self,
        *args, 
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, path, model_cls, pretrained_siglip_name_or_path, foreach=False) -> "EMAModel_SigLIP_LoRA":

        siglip_model = SiglipVisionModel.from_pretrained(pretrained_siglip_name_or_path)
        siglip_model_lora = PeftModel.from_pretrained(siglip_model, path)
        ema_model = cls(siglip_model_lora.parameters(), model_cls=model_cls, model_config=siglip_model_lora.config, foreach=foreach)

        with open(os.path.join(path, 'ema_kwargs.json'), 'r') as f:
            ema_kwargs = json.load(f)
        ema_model.load_state_dict(ema_kwargs)

        return ema_model

    def save_pretrained(self, path, pretrained_siglip_name_or_path,  lora_config):
        if self.model_cls is None:
            raise ValueError("`save_pretrained` can only be used if `model_cls` was defined at __init__.")

        if self.model_config is None:
            raise ValueError("`save_pretrained` can only be used if `model_config` was defined at __init__.")

        siglip_model = SiglipVisionModel.from_pretrained(pretrained_siglip_name_or_path)
        siglip_model = get_peft_model(siglip_model, lora_config)
        state_dict = self.state_dict()
        state_dict.pop("shadow_params", None)

        self.copy_to(siglip_model.parameters())
        siglip_model.save_pretrained(path)
        
        with open(os.path.join(path, 'ema_kwargs.json'), 'w') as f:
            json.dump(state_dict, f, indent=2)

if __name__ == "__main__":

    GB = 1024 * 1024 * 1024
    device = torch.device('cuda')

    pretrained_siglip_name_or_path = '/mnt/data/checkpoints/google/siglip2-so400m-patch16-512'
    siglip_model = SiglipVisionModel.from_pretrained(pretrained_siglip_name_or_path).to(device)

    lora_config = LoraConfig(
        r=256,
        lora_alpha=256,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2'],
        lora_dropout=0.0,
    )
    siglip_model = get_peft_model(siglip_model, lora_config)
    print(siglip_model)
    print(f"Load siglip_model finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")
    ema_siglip_lora = deepcopy(siglip_model)
    ema_siglip_lora = EMAModel_SigLIP_LoRA(
        ema_siglip_lora.parameters(), model_cls=PeftModel, model_config=ema_siglip_lora.config, 
        )
    ema_siglip_lora.to(device)
    print(f"Load ema model finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")

    ema_siglip_lora.save_pretrained('ema_siglip_lora', pretrained_siglip_name_or_path, lora_config)
    print(ema_siglip_lora)
    
    load_ema_siglip_lora = EMAModel_SigLIP_LoRA.from_pretrained('ema_siglip_lora', PeftModel, pretrained_siglip_name_or_path)
    print(load_ema_siglip_lora)
    print(f"Load ema model finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")
    '''

    CUDA_VISIBLE_DEVICES=4 python univa/utils/create_ema_siglip_lora.py
    '''