import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers.models.clip import modeling_clip

class FluxDenoiseTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.mm_train_from_scratch = getattr(args, 'mm_train_from_scratch', False)
        self.unfreeze = getattr(args, 'unfreeze_mm_vision_tower', False)

        self.mm_mask_ratio = getattr(args, 'mm_mask_ratio', 0.0)
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None, pretrained=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)       
        if self.mm_train_from_scratch:
            config = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
            self.vision_tower = CLIPVisionModel._from_config(config)
            print(f'[debug]\ttrain from scratch vision encoder')
        else:
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
            print(f'[debug]\tload pretrained weight from {self.vision_tower_name}')
        print(f'[debug]\tis training? {self.unfreeze}')
        assert not ((not self.unfreeze) and self.mm_train_from_scratch)
        self.vision_tower.requires_grad_(self.unfreeze)
        print(f"[debug]\tself.vision_tower.requires_grad="
              f"{self.vision_tower.vision_model.embeddings.patch_embedding.weight.requires_grad}")

        if pretrained is not None:
            print(f"=> loading pretrained mm_vision_tower from {self.pretrained} ...")
            self.vision_tower.load_state_dict(torch.load(self.pretrained, map_location='cpu'))

        self.is_loaded = True

    def feature_select(self, image_forward_outs, return_cls_token=False):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        # if self.select_feature == 'patch':
        #     image_features = image_features[:, 1:]
        # elif self.select_feature == 'cls_patch':
        #     image_features = image_features
        # else:
        #     raise ValueError(f'Unexpected select feature: {self.select_feature}')
        if not return_cls_token:
            assert self.select_feature == 'patch'
            image_features = image_features[:, 1:]
            
        return image_features

    def inner_forward(self, images, return_cls_token=False):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out, return_cls_token).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs, return_cls_token).to(images.dtype)

        return {'image_features': image_features}


    def random_masking(self, x, mm_mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mm_mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, hidden_states, output_hidden_states):
        # embed patches
        hidden_states = self.vision_tower.vision_model.embeddings(hidden_states)

        cls_tokens, hidden_states = hidden_states[:, :1, :], hidden_states[:, 1:, :]
        # masking: length -> length * mm_mask_ratio
        hidden_states, mask, ids_restore = self.random_masking(hidden_states, self.mm_mask_ratio)

        # append cls token
        hidden_states = torch.cat((cls_tokens, hidden_states), dim=1)

        # apply Transformer blocks
        hidden_states = self.vision_tower.vision_model.pre_layrnorm(hidden_states)
        image_forward_outs = self.vision_tower.vision_model.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=output_hidden_states,
        )

        return image_forward_outs, mask, ids_restore
    

    def mask_inner_forward(self, images, return_cls_token=False):
        if type(images) is list:
            image_features = []
            masks = []
            ids_restores = []
            for image in images:
                image_forward_out, mask, ids_restore = self.forward_encoder(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out, return_cls_token).to(image.dtype)
                image_features.append(image_feature)
                masks.append(mask)
                ids_restores.append(ids_restore)
        else:
            image_forward_outs, masks, ids_restores = self.forward_encoder(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs, return_cls_token).to(images.dtype)

        return {'image_features': image_features, 'masks': masks, 'ids_restores': ids_restores}

    def forward(self, images, return_cls_token=False):
        if self.unfreeze:
            encoder_output = self.inner_forward(images, return_cls_token) if self.mm_mask_ratio == 0.0 else self.mask_inner_forward(images, return_cls_token)
            return encoder_output
        else:
            with torch.no_grad():
                encoder_output = self.inner_forward(images, return_cls_token) if self.mm_mask_ratio == 0.0 else self.mask_inner_forward(images, return_cls_token)
                return encoder_output

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


