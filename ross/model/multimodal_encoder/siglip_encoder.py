import torch
import torch.nn as nn

from transformers import SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig

from transformers.models.clip import modeling_clip

class CompiledCLIPMLP(modeling_clip.CLIPMLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.compile
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
class CompiledLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.compile
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
modeling_clip.CLIPMLP = CompiledCLIPMLP
modeling_clip.nn.LayerNorm = CompiledLayerNorm

class SiglipVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = -2
        self.mm_train_from_scratch = getattr(args, 'mm_train_from_scratch', False)
        self.unfreeze = getattr(args, 'unfreeze_mm_vision_tower', False)

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        
        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        self.image_processor.crop_size = self.image_processor.size
        if self.mm_train_from_scratch:
            config = SiglipVisionConfig.from_pretrained(self.vision_tower_name)
            self.vision_tower = SiglipVisionModel._from_config(config)
        else:
            self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        assert not ((not self.unfreeze) and self.mm_train_from_scratch)
        print(f'[debug]\tis train from scratch vision encoder? {self.mm_train_from_scratch}')
        print(f'[debug]\tis training? {self.unfreeze}')
        self.vision_tower.requires_grad_(self.unfreeze)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]

        return image_features

    def inner_forward(self, images, return_cls_token=False):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                                                      output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                                   output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    def forward(self, images, return_cls_token=False):
        if self.unfreeze:
            return self.inner_forward(images, return_cls_token)
        else:
            with torch.no_grad():
                return self.inner_forward(images, return_cls_token)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    # @property
    def image_size(self):
        return self.config.image_size
    
    # @property
    def patch_size(self):
        return self.config.patch_size
    
    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


