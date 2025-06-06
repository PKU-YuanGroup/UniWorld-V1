import os
from transformers.configuration_utils import PretrainedConfig
from transformers import SiglipVisionConfig
from typing import Literal, Optional, Union


class UnivaVisionTowerConfig(PretrainedConfig):
    model_type = "univa_vision_tower"
    sub_configs = {
        "vision_tower_config": SiglipVisionConfig,
    }
    
    def __init__(
        self,
        vision_tower_type: Literal["siglip"] = "siglip",
        vision_tower_config: Optional[Union[str, dict]] = None,
        mm_projector_type: str = "mlp2x_gelu",
        feature_select_layer: int = -1,
        output_hidden_size: int = 1152,
        shortcut_projector_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vision_tower_type = vision_tower_type
        self.mm_projector_type = mm_projector_type
        self.feature_select_layer = feature_select_layer
        self.output_hidden_size = output_hidden_size
        self.shortcut_projector_type = shortcut_projector_type

        if vision_tower_type == "siglip":
            config_cls = SiglipVisionConfig
        else:
            raise ValueError(f"Unknown vision tower type: {vision_tower_type}")

        if vision_tower_config is not None:
            if isinstance(vision_tower_config, str):
                if not os.path.exists(vision_tower_config):
                    raise FileNotFoundError(
                        f"Vision tower config file not found: {vision_tower_config}"
                    )
                self.vision_tower_config = config_cls.from_pretrained(
                    vision_tower_config
                )
            elif isinstance(vision_tower_config, dict):
                self.vision_tower_config = config_cls(**vision_tower_config)
        else:
            self.vision_tower_config = config_cls()
