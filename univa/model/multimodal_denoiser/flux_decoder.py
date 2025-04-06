import torch
import torch.nn as nn
from copy import deepcopy
from PIL import Image
from diffusers import FluxPipeline, FluxTransformer2DModel, AutoencoderKL
from .scheduler import OpenSoraPlanFlowMatchEulerScheduler
from univa.logger import setup_logger
from transformers import CLIPTextModel, CLIPTokenizer

logger = setup_logger(__name__, level="DEBUG")

class FluxDenoiseTower(nn.Module):
    def __init__(self, denoise_tower: str, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.noise_scheduler = OpenSoraPlanFlowMatchEulerScheduler()
        self.denoise_tower_name = denoise_tower

        self.unfreeze = getattr(
            args, "unfreeze_mm_denoise_tower", False
        )  # If train denoise tower
        self.guidance = getattr(
            args, "flux_guidance", 3.5
        ) # Guidance embed

        self.transformer = None
        self.vae = None

        assert not delay_load
        if not delay_load:
            self.load_model()

        self.is_build_pipeline = False

    def load_model(self, device_map=None, pretrained: str = None) -> None:
        if self.is_loaded:
            print(
                "{} is already loaded, `load_model` called again, skipping.".format(
                    self.vision_tower_name
                )
            )
            return

        self.transformer = FluxTransformer2DModel.from_config(
            self.denoise_tower_name, subfolder="transformer"
        )
        self.vae = AutoencoderKL.from_config(
            self.denoise_tower_name, subfolder="vae"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            self.denoise_tower_name, subfolder="text_encoder"
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            self.denoise_tower_name, subfolder="tokenizer"
        )

        self.transformer.requires_grad_(self.unfreeze)
        self.vae.requires_grad_(False)
        
        # 1 1 dim
        self.register_buffer(
            "pooled_prompt_embeds",
            self._get_pooled_prompt_embeds(
                text_encoder,
                tokenizer,
                prompt="",
            ),
        )

        if pretrained is not None:
            print(f"=> loading pretrained mm_denoise_tower from {self.pretrained} ...")
            self.load_state_dict(torch.load(self.pretrained, map_location="cpu"))

        self.is_loaded = True

    def build_pipeline(self, dtype: torch.dtype = torch.float16):
        pipeline = FluxPipeline.from_pretrained(
            self.denoise_tower_name, transformer=self.transformer, vae=self.vae, torch_dtype=dtype
        )
        self.pipeline = pipeline
        self.is_build_pipeline = True

    def _get_t5_prompt_embeds(
        self, text_encoder, tokenizer, max_sequence_length, prompt=""
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_embeds = text_encoder(
            text_inputs.input_ids.to(text_encoder.device), output_hidden_states=False
        )[0]

        return prompt_embeds

    def _encode_prompt_with_clip(self, text_encoder, tokenizer, prompt=""):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        pooled_prompt_embeds = text_encoder(
            text_inputs.input_ids, output_hidden_states=False
        )
        pooled_prompt_embeds = pooled_prompt_embeds.pooler_output
        return pooled_prompt_embeds

    def _get_pooled_prompt_embeds(self, text_encoder, tokenizer, prompt=""):
        pooled_prompt_embeds = self._encode_prompt_with_clip(
            text_encoder, tokenizer, prompt=prompt
        )
        return pooled_prompt_embeds

    def _prepare_latent_image_ids(
        self, batch_size: int, height: int, width: int, device, dtype
    ):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width)[None, :]
        )

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(
            batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        
        return latent_image_ids.to(device=device, dtype=dtype)
    def _pack_latents(self, latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents
    def inner_forward_batch(self, image, encoder_hidden_state):
        bsz, _, height, width = image.shape

        # Encode image to latent
        image = ((image / 255.0) - 0.5) / 0.5
        image = image.to(device=self.device, dtype=self.dtype)
        latent = self.vae.encode(image).latent_dist.sample()
        _, _, latent_height, latent_width = latent.shape
        latent = self._pack_latents(latent)
        

        # Get latent image ids
        latent_image_ids = self._prepare_latent_image_ids(
            bsz, latent_height // 2, latent_width // 2, device = self.device, dtype=self.dtype
        )

        # Get text ids
        text_ids = torch.zeros(bsz, encoder_hidden_state.shape[1], 3).to(
            device=self.device, dtype=self.dtype
        )

        # Scale latent
        model_input = (
            latent / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor

        # Sample a noise
        noise = torch.randn_like(model_input)

        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        sigmas = self.noise_scheduler.compute_density_for_sigma_sampling(
            batch_size=bsz,
            logit_mean=0.0,
            logit_std=1.0,
        ).to(device=self.device)
        timestep = (
            sigmas.clone() * self.noise_scheduler.rescale
        )  # rescale to [0, 1000.0)
        while sigmas.ndim < latent.ndim:
            sigmas = sigmas.unsqueeze(-1)

        noisy_model_input = self.noise_scheduler.add_noise(latent, sigmas, noise)
        pooled_prompt_embeds = self.pooled_prompt_embeds.repeat(bsz, 1)

        # Guidance input
        if self.transformer.config.guidance_embeds:
            guidance = torch.tensor([self.guidance], device=self.device)
            guidance = guidance.expand(bsz)
        else:
            guidance = None

        # Predict
        logger.debug(f"[debug] noisy_model_input.shape: {noisy_model_input.shape}")
        logger.debug(f"[debug] timestep: {timestep / 1000}")
        logger.debug(f"[debug] guidance.shape: {guidance.shape}")
        logger.debug(f"[debug] encoder_hidden_states.shape: {encoder_hidden_state.shape}")
        logger.debug(f"[debug] latent_image_ids.shape: {latent_image_ids.shape}")
        logger.debug(f"[debug] text_ids.shape: {text_ids.shape}")
        model_pred = self.transformer(
            hidden_states=noisy_model_input,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=encoder_hidden_state,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

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
            model_pred, target, weighting = tuple(
                zip(
                    *[
                        self.inner_forward_batch(
                            images[i : i + 1], encoder_hidden_states[i : i + 1]
                        )
                        for i in range(bsz)
                    ]
                )
            )
        else:
            model_pred, target, weighting = self.inner_forward_batch(
                images, encoder_hidden_states
            )
        return {"model_pred": model_pred, "target": target, "weighting": weighting}

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
        output_type="pil",
        **kwargs,
    ):
        pooled_prompt_embeds = self.pooled_prompt_embeds
        images = self.pipeline(
            height=image_size[0],
            width=image_size[1],
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type=output_type,
            **kwargs,
        ).images[0]
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
