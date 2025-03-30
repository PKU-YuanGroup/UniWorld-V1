import torch
import torch.nn as nn
import math
from transformers import ConvNextModel, CLIPImageProcessor, ConvNextConfig

# class CompiledCLIPMLP(modeling_clip.CLIPMLP):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     @torch.compile
#     def forward(self, *args, **kwargs):
#         return super().forward(*args, **kwargs)
    
# class CompiledLayerNorm(nn.LayerNorm):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     @torch.compile
#     def forward(self, *args, **kwargs):
#         return super().forward(*args, **kwargs)
    
# modeling_clip.CLIPMLP = CompiledCLIPMLP
# modeling_clip.nn.LayerNorm = CompiledLayerNorm

class ConvNeXtCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.mm_vision_resolution = getattr(args, 'mm_vision_resolution', False)
        self.mm_train_from_scratch = getattr(args, 'mm_train_from_scratch', False)
        self.mm_vision_patch_size = getattr(args, 'mm_vision_patch_size', False)
        self.mm_anyres = getattr(args, 'mm_anyres', False)
        self.mm_anyres_min_pixels = getattr(args, 'mm_anyres_min_pixels', 320*320)
        self.mm_anyres_max_pixels = getattr(args, 'mm_anyres_max_pixels', 864*864)
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.unfreeze = getattr(args, 'unfreeze_mm_vision_tower', False)


        self.mm_mask_ratio = getattr(args, 'mm_mask_ratio', 0.0)
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = ConvNextConfig.from_pretrained(self.vision_tower_name)
        
    def load_model(self, device_map=None, pretrained=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        
        self.image_processor.anyres = self.mm_anyres
        self.image_processor.min_pixels = self.mm_anyres_min_pixels
        self.image_processor.max_pixels = self.mm_anyres_max_pixels

        if self.mm_train_from_scratch:
            config = ConvNextConfig.from_pretrained(self.vision_tower_name)
            self.vision_tower = ConvNextModel._from_config(config)
            print(f'[debug]\ttrain from scratch vision encoder')
        else:
            self.vision_tower = ConvNextModel.from_pretrained(self.vision_tower_name, device_map=device_map)
            print(f'[debug]\tload pretrained weight from {self.vision_tower_name}')
        assert not ((not self.unfreeze) and self.mm_train_from_scratch)
        print(f'[debug]\tis training? {self.unfreeze}')
        self.vision_tower.requires_grad_(self.unfreeze)
        print(f"[debug]\tself.vision_tower.requires_grad="
              f"{self.vision_tower.embeddings.patch_embeddings.weight.requires_grad}")

        if pretrained is not None:
            print(f"=> loading pretrained mm_vision_tower from {self.pretrained} ...")
            self.vision_tower.load_state_dict(torch.load(self.pretrained, map_location='cpu'))

        self.is_loaded = True

        if self.mm_vision_patch_size:
            self.cutoff_stage = int(math.log2(self.mm_vision_patch_size)) - 1
            for _ in range(4-self.cutoff_stage):
                self.vision_tower.encoder.stages.pop(-1)
        else:
            self.mm_vision_patch_size = 2 ** (len(self.vision_tower.encoder.stages) + 1)
            self.cutoff_stage = len(self.vision_tower.encoder.stages)
        del self.vision_tower.layernorm
        
        if self.mm_vision_resolution:
            self.set_crop_size(self.mm_vision_resolution)
            print(f'Crop size changed to {self.mm_vision_resolution}x{self.mm_vision_resolution}')
        else:
            self.mm_vision_resolution = self.config.image_size

        if self.select_layer == -2:
            self.select_layer = -1
            self.vision_tower.encoder.stages[-1].layers.pop(-1)
            print(f'Last block removed, select layer changed to {self.select_layer}')
            
        self.set_vision_tower_config()
    
    def inner_forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                # Get the embeddings of the image
                embedding_output = self.vision_tower.embeddings(image.unsqueeze(0))

                # Get the image features
                image_feature = self.vision_tower.encoder(embedding_output,
                                                        output_hidden_states=True,
                                                        return_dict=True)  # B C H W
                # image_feature = image_feature.hidden_states[-1].permute(0, 2, 3, 1)
                # image_feature = image_feature.reshape(image_features.shape[0], -1, image_features.shape[3]).to(images.dtype)
                image_feature = image_feature.hidden_states[-1].to(image.dtype)

                image_features.append(image_feature)
        else:
            embedding_output = self.vision_tower.embeddings(images)
            image_features = self.vision_tower.encoder(embedding_output,
                                                       output_hidden_states=True,
                                                       return_dict=True)  # B C H W
            # image_features = image_features.hidden_states[-1].permute(0, 2, 3, 1)
            # image_features = image_features.reshape(image_features.shape[0], -1, image_features.shape[3]).to(images.dtype)
            image_features = image_features.hidden_states[-1].to(images.dtype)
        return {'image_features': image_features}

    # @torch.compile
    def forward(self, images, return_cls_token=False):
        if self.unfreeze:
            encoder_output = self.inner_forward(images) if self.mm_mask_ratio == 0.0 else self.mask_inner_forward(images)
            return encoder_output
        else:
            with torch.no_grad():
                encoder_output = self.inner_forward(images) if self.mm_mask_ratio == 0.0 else self.mask_inner_forward(images)
                return encoder_output
            
    def set_crop_size(self, new_size):
        size_dict = {'height': new_size, 'width': new_size}
        self.image_processor.crop_size = size_dict
        self.image_processor.size = {"shortest_edge": new_size}

    def set_vision_tower_config(self):
        self.vision_tower.config.depths = self.vision_tower.config.depths[:self.cutoff_stage]
        self.vision_tower.config.hidden_sizes = self.vision_tower.config.hidden_sizes[:self.cutoff_stage]
        self.vision_tower.config.num_stages = len(self.vision_tower.encoder.stages)
        self.vision_tower.config.out_features = [f'stage{self.vision_tower.config.num_stages-1}']
        self.vision_tower.config.out_indices = [self.vision_tower.config.num_stages]
        self.vision_tower.config.stage_names = ['stem'] + [f'stage{i+1}' for i in range(self.vision_tower.config.num_stages)]
        self.vision_tower.config.image_size = self.mm_vision_resolution

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
        return self.config.hidden_sizes[-1]

    # @property
    def image_size(self):
        return self.config.image_size
    
    # @property
    def patch_size(self):
        return self.mm_vision_patch_size
    
    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

