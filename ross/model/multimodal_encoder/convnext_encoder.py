import torch
import torch.nn as nn

from transformers import ConvNextModel, CLIPImageProcessor, ConvNextConfig


class ConvNeXtCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.update_resolution = getattr(args, 'mm_vision_resolution', 256)
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.unfreeze = getattr(args, 'unfreeze_mm_vision_tower', False)

        if not delay_load:
            self.load_model()
        # elif self.unfreeze:
        #     self.load_model()
        else:
            self.cfg_only = ConvNextConfig.from_pretrained(self.vision_tower_name)
        
    def load_model(self, device_map=None, pretrained=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = ConvNextModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(self.unfreeze)
        print(f"[debug]\tself.vision_tower.requires_grad="
              f"{self.vision_tower.vision_model.embeddings.patch_embedding.weight.requires_grad}")

        if pretrained is not None:
            print(f"=> loading pretrained mm_vision_tower from {self.pretrained} ...")
            self.vision_tower.load_state_dict(torch.load(self.pretrained, map_location='cpu'))

        self.is_loaded = True

        if self.select_layer == -2:
            self.select_layer = -1
            self.vision_tower.encoder.stages[-1].layers.pop(-1)
            print(
                f'Last block removed, select layer changed to {self.select_layer}')
            
        if self.update_resolution > 256:
            self.set_crop_size(self.update_resolution)
            print(
                f'Crop size changed to {self.update_resolution}x{self.update_resolution}')
            
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                # Get the embeddings of the image
                embedding_output = self.vision_tower.embeddings(image.unsqueeze(0))

                # Get the image features
                image_feature = self.vision_tower.encoder(embedding_output,
                                                        output_hidden_states=True,
                                                        return_dict=True)
                image_feature = image_feature.hidden_states[-1].permute(0, 2, 3, 1)
                image_feature = image_feature.reshape(image_features.shape[0], -1, image_features.shape[3]).to(images.dtype)

                image_features.append(image_feature)
        else:
            embedding_output = self.vision_tower.embeddings(images)
            image_features = self.vision_tower.encoder(embedding_output,
                                                       output_hidden_states=True,
                                                       return_dict=True)
            image_features = image_features.hidden_states[-1].permute(0, 2, 3, 1)
            image_features = image_features.reshape(image_features.shape[0], -1, image_features.shape[3]).to(images.dtype)

        return image_features
    
    def set_crop_size(self, new_size):
        size_dict = {'height': new_size, 'width': new_size}
        self.image_processor.crop_size = size_dict
        self.image_processor.size = {"shortest_edge": new_size}
        self.vision_tower.config.image_size = new_size

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

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

