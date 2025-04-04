import torch
import torch.nn as nn
from copy import deepcopy
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from .scheduler import OpenSoraPlanFlowMatchEulerScheduler

class SD3DenoiseTower(nn.Module):
    def __init__(self, denoise_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.noise_scheduler = OpenSoraPlanFlowMatchEulerScheduler()
        self.denoise_tower_name = denoise_tower
        self.unfreeze = getattr(args, 'unfreeze_mm_denoise_tower', False)

        assert not delay_load
        if not delay_load:
            self.load_model()

        self.is_build_pipeline = False

    def load_model(self, device_map=None, pretrained=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        pipeline = StableDiffusion3Pipeline.from_pretrained(self.denoise_tower_name, device_map=device_map)

        # 1 1 dim
        self.register_buffer(
            "pooled_prompt_embeds", 
            self._get_pooled_prompt_embeds(
                pipeline.text_encoder, pipeline.tokenizer, pipeline.text_encoder_2, pipeline.tokenizer_2, prompt=''
                )
            )  
        
        
        print(f'[debug]\tload pretrained weight from {self.denoise_tower_name}')
        print(f'[debug]\tis training? {self.unfreeze}')
        self.transformer = deepcopy(pipeline.transformer)
        self.vae = deepcopy(pipeline.vae)

        self.transformer.requires_grad_(self.unfreeze)
        self.vae.requires_grad_(False)

        del pipeline

        print(f"[debug]\tself.denoise_tower.requires_grad="
              f"{self.transformer.pos_embed.proj.weight.requires_grad}")

        if pretrained is not None:
            print(f"=> loading pretrained mm_denoise_tower from {self.pretrained} ...")
            self.load_state_dict(torch.load(self.pretrained, map_location='cpu'))

        self.is_loaded = True

    def build_pipeline(self, dtype=torch.float16, device_map=None):
        pipeline = StableDiffusion3Pipeline.from_pretrained(self.denoise_tower_name, device_map=device_map)
        self.pipeline = pipeline
        self.pipeline.vae = self.vae
        self.pipeline.transformer = self.transformer
        self.is_build_pipeline = True
        self.pipeline.text_encoder.to(dtype)
        self.pipeline.text_encoder_2.to(dtype)
        self.pipeline.text_encoder_3.to(dtype)

    def _get_t5_prompt_embeds(self, text_encoder, tokenizer, max_sequence_length, prompt=''):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_embeds = text_encoder(text_inputs.input_ids.to(text_encoder.device))[0]

        return prompt_embeds


    def _encode_prompt_with_clip(self, text_encoder, tokenizer, prompt=''):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds = text_encoder(text_inputs.input_ids, output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        return pooled_prompt_embeds

    def _get_pooled_prompt_embeds(self, text_encoder, tokenizer, text_encoder_2, tokenizer_2, prompt=''):
        pooled_prompt_embeds = self._encode_prompt_with_clip(
            text_encoder, tokenizer, prompt=prompt
            )
        pooled_prompt_embeds_2 = self._encode_prompt_with_clip(
            text_encoder_2, tokenizer_2, prompt=prompt
            )
        pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, pooled_prompt_embeds_2], dim=-1)
        return pooled_prompt_embeds
    
    def inner_forward_batch(self, image, encoder_hidden_state):
        bsz, _, height, width = image.shape
        image = ((image / 255.0) - 0.5) / 0.5
        image = image.to(device=self.device, dtype=self.dtype)
        latent = self.vae.encode(image).latent_dist.sample()
        model_input = (latent / self.vae.config.scaling_factor) + self.vae.config.shift_factor

        noise = torch.randn_like(model_input)

        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        sigmas = self.noise_scheduler.compute_density_for_sigma_sampling(
            batch_size=bsz,
            logit_mean=0.0,
            logit_std=1.0,
        ).to(device=self.device)
        timestep = sigmas.clone() * self.noise_scheduler.rescale  # rescale to [0, 1000.0)
        while sigmas.ndim < latent.ndim:
            sigmas = sigmas.unsqueeze(-1)

        noisy_model_input = self.noise_scheduler.add_noise(latent, sigmas, noise)
        pooled_prompt_embeds = self.pooled_prompt_embeds.repeat(bsz, 1)
        model_pred = self.transformer(
            hidden_states=noisy_model_input, 
            encoder_hidden_states=encoder_hidden_state, 
            pooled_projections=pooled_prompt_embeds, 
            timestep=timestep, 
        ).sample

        target = noise - model_input
        weighting = self.noise_scheduler.compute_loss_weighting_for_sd3(sigmas=sigmas)

        return model_pred, target, weighting

    def inner_forward(
        self,
        images: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ):
        bsz = len(images)
        
        if isinstance(images, list):
            model_pred, target, weighting = tuple(zip(*[self.inner_forward_batch(images[i: i+1], encoder_hidden_states[i: i+1]) for i in range(bsz)]))
        else:
            model_pred, target, weighting = self.inner_forward_batch(images, encoder_hidden_states)
        return {'model_pred': model_pred, 'target': target, 'weighting': weighting}

    def forward(self, images, encoder_hidden_states):
        if self.unfreeze:
            denoiser_output = self.inner_forward(images, encoder_hidden_states)
            return denoiser_output
        else:
            with torch.no_grad():
                denoiser_output = self.inner_forward(images, encoder_hidden_states)
                return denoiser_output

    @torch.no_grad()
    def sample(
        self, 
        image_size, 
        prompt_embeds, 
        num_inference_steps=28, 
        guidance_scale=1.0, 
        output_type='pil', 
        **kwargs, 
        ):
        pooled_prompt_embeds = self.pooled_prompt_embeds
        negative_pooled_prompt_embeds = self.pooled_prompt_embeds
        negative_prompt_embeds = self._get_t5_prompt_embeds(
            self.pipeline.text_encoder_3, self.pipeline.tokenizer_3, prompt_embeds.shape[1]
            )
        assert negative_prompt_embeds.shape[1] >= prompt_embeds.shape[1], \
            f"negative_prompt_embeds.shape ({negative_prompt_embeds.shape}) vs prompt_embeds.shape ({prompt_embeds.shape})"
        negative_prompt_embeds = negative_prompt_embeds[:, :prompt_embeds.shape[1]]
        images = self.pipeline(        
            height=image_size[0],
            width=image_size[1],
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds, 
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, 
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type=output_type, 
            **kwargs, 
        ).images[0]
        images.save('tmp.jpg')
        return images

    @property
    def dtype(self):
        return self.transformer.dtype

    @property
    def device(self):
        return self.transformer.device

    @property
    def config(self):
        if self.is_loaded:
            return self.transformer.config
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


