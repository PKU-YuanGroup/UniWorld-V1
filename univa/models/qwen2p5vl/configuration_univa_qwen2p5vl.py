from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig
from univa.models.configuration_univa_denoise_tower import UnivaDenoiseTowerConfig
from typing import Optional


class UnivaQwen2p5VLConfig(Qwen2_5_VLConfig):
    model_type = "univa_qwen2p5vl"
    sub_configs = {
        "vision_config": Qwen2_5_VLVisionConfig,
        # "vision_tower": UnivaQwen2p5VLVisionTowerConfig,
        "denoise_tower": UnivaDenoiseTowerConfig,
    }

    def __init__(
        self,
        # vision_tower: UnivaQwen2p5VLVisionTowerConfig = None,
        denoise_tower: UnivaDenoiseTowerConfig = None,
        image_token_length: Optional[int] = None,
        shortcut_image_embeds: bool = False,
        shortcut_image_embeds_scale: float = 0.5,
        shortcut_projector_type: Optional[str] = "mlp2x_gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_token_length = image_token_length
        self.shortcut_image_embeds = shortcut_image_embeds
        self.shortcut_image_embeds_scale = shortcut_image_embeds_scale

        if not shortcut_image_embeds:
            shortcut_projector_type = None
        self.shortcut_projector_type = shortcut_projector_type

        self.image_begin_token_id = None
        self.image_end_token_id = None
        self.think_begin_token = None
        self.think_end_token = None
        self.think_token = None
        self.no_think_token = None
        
        # if isinstance(vision_tower, dict):
        #     vision_tower["shortcut_projector_type"] = shortcut_projector_type
        #     self.vision_tower = UnivaQwen2p5VLVisionTowerConfig(**vision_tower)
        # elif vision_tower is None:
        #     self.vision_tower = UnivaQwen2p5VLVisionTowerConfig(
        #         shortcut_projector_type=shortcut_projector_type
        #     )
        # else:
        #     self.vision_tower = vision_tower

        print(denoise_tower)

        if isinstance(denoise_tower, dict):
            denoise_tower["input_hidden_size"] = self.hidden_size
            self.denoise_tower = UnivaDenoiseTowerConfig(**denoise_tower)
        elif denoise_tower is None:
            self.denoise_tower = UnivaDenoiseTowerConfig(
                input_hidden_size=self.hidden_size
            )
        else:
            self.denoise_tower = denoise_tower
