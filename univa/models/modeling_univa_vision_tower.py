from univa.models.configuration_univa_vision_tower import UnivaVisionTowerConfig
from transformers.models.siglip.modeling_siglip import SiglipVisionModel
from transformers.modeling_utils import PreTrainedModel
import torch
import torch.nn as nn


class UnivaVisionTower(PreTrainedModel):
    config_class = UnivaVisionTowerConfig
    base_model_prefix = "model"
    _supports_flash_attn_2 = True
    
    def __init__(self, config: UnivaVisionTowerConfig):
        super().__init__(config)
        self.config = config

        # Initialize vision tower
        if config.vision_tower_type == "siglip":
            self.model = SiglipVisionModel._from_config(self.config.vision_tower_config)
            self.mm_hidden_size = self.config.vision_tower_config.hidden_size
        else:
            raise ValueError(f"Unknown vision tower type: {config.vision_tower_type}")

        self._init_mm_projector()

        if self.config.shortcut_projector_type is not None:
            self._init_shortcut_projector()

    def _init_mm_projector(self):
        """Initialize the mm_projector for multi model input."""
        if self.config.mm_projector_type == "mlp2x_gelu":
            self.mm_projector = nn.Sequential(
                nn.Linear(
                    self.mm_hidden_size,
                    self.config.output_hidden_size,
                ),
                nn.GELU(),
                nn.Linear(
                    self.config.output_hidden_size, self.config.output_hidden_size
                ),
            )
        else:
            raise ValueError(
                f"Unknown mm_projector_type: {self.config.mm_projector_type}"
            )

    def _init_shortcut_projector(self):
        """Initialize the shortcut_projector for multi model input."""
        if self.config.shortcut_projector_type == "mlp2x_gelu":
            self.shortcut_projector = nn.Sequential(
                nn.Linear(
                    self.mm_hidden_size,
                    self.config.output_hidden_size,
                ),
                nn.GELU(),
                nn.Linear(
                    self.config.output_hidden_size, self.config.output_hidden_size
                ),
            )
        elif self.config.shortcut_projector_type == "share_mm_projector":
            ...
        else:
            raise ValueError(
                f"Unknown shortcut_projector_type: {self.config.shortcut_projector_type}"
            )

    def forward(self, pixel_values: torch.Tensor):
        x = self.model(pixel_values, output_hidden_states=True)
        x = x.hidden_states[self.config.feature_select_layer]

        if (
            self.config.shortcut_projector_type is not None
            and self.config.shortcut_projector_type != "share_mm_projector"
        ):
            shortcut_x = self.shortcut_projector(x)

        x = self.mm_projector(x)

        if self.config.shortcut_projector_type is not None:
            if (
                self.config.shortcut_projector_type == "share_mm_projector"
            ):  # Share the mm_projector as the shortcut_projector
                return x, x
            else:
                return x, shortcut_x
        else:
            return x, None
