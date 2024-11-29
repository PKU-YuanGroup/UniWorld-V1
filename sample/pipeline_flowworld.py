
import inspect
from typing import Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import torch
from einops import rearrange

from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import logging, BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from .scheduler import FlowMatchEulerScheduler


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class FlowWorldPipelineOutput(BaseOutput):
    videos: Union[List[torch.FloatTensor], np.ndarray]

def flowworld_linear_quadratic_schedule(num_inference_steps, approximate_steps=1000):
    assert approximate_steps % 2 == 0, "approximate_steps must be even"
    assert num_inference_steps % 2 == 0, "num_inference_steps must be even"
    assert num_inference_steps <= approximate_steps, "num_inference_steps must be less than or equal to approximate_steps"

    _num_inference_steps = num_inference_steps // 2
    _approximate_steps = approximate_steps // 2

    linear_sigmas = [i / (2 * _approximate_steps) for i in range(_num_inference_steps)]
    # NOTE we define a quadratic schedule that is f(x) = ax^2 + bx + c
    quadratic_a = (_approximate_steps - _num_inference_steps) / (_approximate_steps * _num_inference_steps ** 2)
    quadratic_b = (5 * _num_inference_steps - 4 * _approximate_steps) / (2 * _approximate_steps * _num_inference_steps)
    quadratic_c = (_approximate_steps - _num_inference_steps) / _approximate_steps
    quadratic_sigmas = [
        quadratic_a * i ** 2 + quadratic_b * i + quadratic_c for i in range(_num_inference_steps, 2 * _num_inference_steps)
    ]
    sigmas = linear_sigmas + quadratic_sigmas + [1.0]
    sigmas = [1.0 - x for x in sigmas]
    # sigmas.pop(-1) # remove 0
    return sigmas

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class FlowWorldPipeline(DiffusionPipeline):

    model_cpu_offload_seq = "transformer->vae"
    _callback_tensor_inputs = [
        "latents",
    ]

    def __init__(
        self,
        vae,
        transformer,
        scheduler,
    ):
        super().__init__()
        self.vae_scale_factor = [1, 8, 8]
        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )

    def encode_prompt(
        self,
        prompt: str,
        device: torch.device = None,
        dtype: torch.dtype = None,
        num_samples_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        max_sequence_length: Optional[int] = None,
        text_encoder_index: int = 0,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            dtype (`torch.dtype`):
                torch dtype
            num_samples_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the prompt. Required when `prompt_embeds` is passed directly.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the negative prompt. Required when `negative_prompt_embeds` is passed directly.
            max_sequence_length (`int`, *optional*): maximum sequence length to use for the prompt.
            text_encoder_index (`int`, *optional*):
                Index of the text encoder to use. `0` for T5 and `1` for clip.
        """
        if dtype is None:
            if self.text_encoder_2 is not None:
                dtype = self.text_encoder_2.dtype
            elif self.transformer is not None:
                dtype = self.transformer.dtype
            else:
                dtype = None

        if device is None:
            device = getattr(self, '_execution_device', None) or getattr(self, 'device', None) or torch.device('cuda')


        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = tokenizers[text_encoder_index]
        text_encoder = text_encoders[text_encoder_index]

        if max_sequence_length is None:
            if text_encoder_index == 0:
                max_length = 512
            if text_encoder_index == 1:
                max_length = 77
        else:
            max_length = max_sequence_length

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_attention_mask = text_inputs.attention_mask.to(device)
            prompt_embeds = text_encoder(
                text_input_ids.to(device),
                attention_mask=prompt_attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

            if text_encoder_index == 1:
                prompt_embeds = prompt_embeds.unsqueeze(1)  # b d -> b 1 d for clip

            prompt_attention_mask = prompt_attention_mask.repeat(num_samples_per_prompt, 1)

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)


        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_samples_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_samples_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            negative_prompt_attention_mask = uncond_input.attention_mask.to(device)
            negative_prompt_embeds = text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=negative_prompt_attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]
            if text_encoder_index == 1:
                negative_prompt_embeds = negative_prompt_embeds.unsqueeze(1)  # b d -> b 1 d for clip
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_samples_per_prompt, 1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_samples_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_samples_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds, prompt_attention_mask, negative_prompt_attention_mask

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        label,
        num_frames,
        height,
        width,
        callback_on_step_end_tensor_inputs=None,
    ):
        if (num_frames - 1) % 4 != 0:
            raise ValueError(f"`num_frames - 1` have to be divisible by 4 but is {num_frames}.")
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            (int(num_frames) - 1) // self.vae_scale_factor[0] + 1, 
            int(height) // self.vae_scale_factor[1],
            int(width) // self.vae_scale_factor[2],
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        if not isinstance(self.scheduler, FlowMatchEulerScheduler):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        label: Union[int, List[int]] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = 50,
        timesteps: List[int] = None,
        guidance_scale: Optional[float] = 5.0,
        num_samples_per_label: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        guidance_rescale: float = 0.0,
        use_linear_quadratic_schedule: bool = False, 
        device = None, 
    ):
        
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. default height and width
        num_frames = num_frames or (self.transformer.config.sample_size_t - 1) * self.vae.vae_scale_factor[0] + 1
        height = height or self.transformer.config.sample_size[0] * self.vae.vae_scale_factor[1]
        width = width or self.transformer.config.sample_size[1] * self.vae.vae_scale_factor[2]

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            label,
            num_frames, 
            height,
            width,
            callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._interrupt = False

        # 2. Define call parameters
        if label is not None and isinstance(label, int):
            batch_size = 1
            label = [label]
        elif label is not None and isinstance(label, list):
            batch_size = len(label)

        device = device or getattr(self, '_execution_device', None) or getattr(self, 'device', None) or torch.device('cuda')


        # 4. Prepare timesteps
        if not isinstance(self.scheduler, FlowMatchEulerScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            self._num_timesteps = len(timesteps)
        else:
            # timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
            sigmas = None
            if use_linear_quadratic_schedule:
                approximate_steps = min(max(num_inference_steps*10, 250), 1000)
                sigmas = flowworld_linear_quadratic_schedule(
                    num_inference_steps=num_inference_steps, 
                    approximate_steps=approximate_steps
                    )
                sigmas = np.array(sigmas)
                print(f"Using linear quadratic schedule, sigmas: {sigmas}, num_inference_steps: {num_inference_steps}, approximate_steps: {approximate_steps}")
            sigmas = self.scheduler.set_sigmas(num_inference_steps=num_inference_steps, device=device, sigmas=sigmas)
            print(len(sigmas), sigmas)
            timesteps = sigmas.clone() * 1000
            timesteps = timesteps[:-1]
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
            self._num_timesteps = len(timesteps)
        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_samples_per_label,
            num_channels_latents,
            num_frames, 
            height,
            width,
            self.transformer.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = {}
        # 7 create image_rotary_emb, style embedding & time ids
        label = torch.LongTensor(label).to(device)
        label = label.repeat(num_samples_per_label)
        if self.do_classifier_free_guidance:
            negative_label = [1000] * len(label)
            negative_label = torch.LongTensor(negative_label).to(device)
            label = torch.cat([negative_label, label])

        # 8. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                timestep = t.expand(latent_model_input.shape[0])

                # ==================prepare my shape=====================================
                # predict the noise residual
                if latent_model_input.ndim == 5:
                    latent_model_input = rearrange(latent_model_input, 'b c 1 h w -> (b 1) c h w')
                # ==================prepare my shape=====================================


                noise_pred = self.transformer(
                    latent_model_input,
                    t=timestep,
                    y=label
                )
                assert not torch.any(torch.isnan(noise_pred))
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)


                # ==================prepare my shape=====================================
                # predict the noise residual
                if noise_pred.ndim == 4:
                    noise_pred = rearrange(noise_pred, '(b 1) c h w -> b c 1 h w')
                # ==================prepare my shape=====================================
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, i, latents, **extra_step_kwargs, return_dict=False)[0]


                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    label = callback_outputs.pop("label", label)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            videos = self.decode_latents(latents)
            videos = videos[:, :num_frames, :height, :width]
        else:
            videos = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (videos, )

        return FlowWorldPipelineOutput(videos=videos)

    
    def decode_latents(self, latents):
        print(f'before vae decode {latents.shape}', torch.max(latents).item(), torch.min(latents).item(), torch.mean(latents).item(), torch.std(latents).item())
        if latents.ndim == 5:
            latents = latents.squeeze(2)
        video = self.vae.decode(latents / 0.18215).sample
        print(f'after vae decode {latents.shape}', torch.max(video).item(), torch.min(video).item(), torch.mean(video).item(), torch.std(video).item())
        video = ((video / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu()
        video = rearrange(video, 'b c h w -> b 1 h w c').contiguous() # b t h w c
        return video