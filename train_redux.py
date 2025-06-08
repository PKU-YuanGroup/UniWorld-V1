import torch._dynamo
torch._dynamo.config.optimize_ddp = False
from univa.training.configuration_denoise import UnivaTrainingDenoiseConfig
from pathlib import Path
import os
from typing import List, Dict, Callable, Optional
import math
import random
import shutil
from einops import rearrange, repeat
import copy
import json
import deepspeed
from accelerate import Accelerator, DistributedDataParallelKwargs, DistributedType
from accelerate.utils import ProjectConfiguration, set_seed
from univa.models import MODEL_TYPE, UnivaQwen2ForCausalLM
from univa.models.modeling_univa_denoise_tower import UnivaDenoiseTower
from univa.dataset import ReduxDataset
from univa.dataset.data_collator import DataCollator, pad_list_of_tensors
from univa.utils.prompter import PROMPT_TYPE, Qwen2Prompter
from univa.utils.constant import SPACIAL_TOKEN, GENERATE_TOKEN
from univa.utils.denoiser_prompt_embedding_flux import encode_prompt, _encode_prompt_with_t5
from univa.utils.get_ocr import ocr_with_paddle, draw_boxes, get_ocr_result
from univa.utils.flux_pipeline import FluxPipeline
from univa.utils.create_ema import EMAModel, _z3_params_to_fetch
from univa.utils.anyres_util import dynamic_resize
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers.integrations import HfDeepSpeedConfig
from transformers import (
    CLIPTextModel,
    T5EncoderModel,
    CLIPTokenizer,
    T5TokenizerFast,
    AutoImageProcessor,
    PreTrainedTokenizer,
    AutoTokenizer,
    AutoProcessor, 
)
from torchvision import transforms
import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from tqdm import tqdm
from PIL import Image
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    # FluxPipeline,
)
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from contextlib import nullcontext
import wandb
GB = 1024 * 1024 * 1024

def get_trainable_params(
    layers_to_train: int = list(range(57)), num_transformer_blocks: int = 19, only_img_branch: bool = True
):
    components = [
        # "x_embedder"
        ]
    transformer_components = [
        "attn.norm_q", 
        "attn.norm_k", 
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out",
        "norm1.linear",
    ]
    single_transformer_components = [
        "attn.norm_q", 
        "attn.norm_k", 
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "norm.linear",
    ]
    if not only_img_branch:
        components.extend(
            [
                # "context_embedder"
            ]
        )
        transformer_components.extend(
            [
                "norm1_context.linear", "attn.norm_added_q", "attn.norm_added_k", "ff.net", "ff_context.net"
            ]
        )
        single_transformer_components.extend(
            [
                "proj_mlp", "proj_out"
            ]
        )
    for layer in layers_to_train:
        if layer < num_transformer_blocks:
            prefix = f"denoiser.transformer_blocks.{layer}"
            base_components = transformer_components
        else:
            prefix = f"denoiser.single_transformer_blocks.{layer - num_transformer_blocks}"
            base_components = single_transformer_components
        components.extend([f"{prefix}.{comp}" for comp in base_components])

    return components

class DataCollator:
    def __call__(self, instances: List[Dict]) -> Dict:
        image_0 = torch.stack([instance["image_0"][0] for instance in instances])
        # image_1 = torch.stack([instance["image_1"][0] for instance in instances])
        generated_image = torch.stack([instance["generated_image"] for instance in instances])

        return {
            "image_0": image_0,
            # "image_1": image_1,
            "generated_image": generated_image, 
        }

def check_param_is_in_components(name: str, components: List[str]) -> bool:
    return any(component in name for component in components)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
                raise NotImplementedError
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def gather_zero3ema(accelerator, ema_model):
    model_to_save = ema_model.model.module if hasattr(ema_model.model, "module") else ema_model.model
    model_state_dict = {}
    for k, v in model_to_save.named_parameters():
        # only gather z3 params
        params_to_fetch = _z3_params_to_fetch([v])
        with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
            vv = v.data.cpu()
            # if accelerator.process_index == 0:
            model_state_dict[k] = vv
    return model_state_dict


def pad_x_and_mask(model_input, attention_mask=None, max_h=None, max_w=None):
    if attention_mask is None:
        attention_mask = [None] * len(model_input)
    batch_attention_mask = None
    max_h = max(t.shape[2] for t in model_input) if max_h is None else max_h
    max_w = max(t.shape[3] for t in model_input) if max_w is None else max_w

    padded_list = []
    padded_mask_list = []
    for t, m in zip(model_input, attention_mask):
        _, _, h, w = t.shape
        pad_h = max_h - h
        pad_w = max_w - w
        # pad 的顺序是 (left, right, top, bottom)
        # 这里只在右边和下边 pad
        padded = F.pad(t, pad=(0, pad_w, 0, pad_h), mode='constant', value=0)
        padded_list.append(padded)
        # import ipdb;ipdb.set_trace()
        if m is not None:
            m = m[:, :1]
            m = F.pad(m, pad=(0, pad_w, 0, pad_h), mode='constant', value=0)
        padded_mask_list.append(m)
    batch_model_input = torch.cat(padded_list, dim=0) 
    if padded_mask_list[0] is not None:
        batch_attention_mask = torch.cat(padded_mask_list, dim=0) 
    return batch_model_input, batch_attention_mask
    
def build_validation_info(args):
    base_eval_prompts = []
    base_eval_image_paths = []
    base_phase_names = []
    if args.dataset_config.validation_t2i_prompt:
        base_eval_prompts.append(args.dataset_config.validation_t2i_prompt)
        base_eval_image_paths.append(None)
        base_phase_names.append('vlm->generate image')
    if args.dataset_config.validation_it2i_prompt:
        base_eval_prompts.append(args.dataset_config.validation_it2i_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_image_path)
        base_phase_names.append('vlm->reconstruct image')
    if args.dataset_config.validation_iit2i_prompt:
        base_eval_prompts.append(args.dataset_config.validation_iit2i_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_iit2i_path)
        base_phase_names.append('vlm->fusion 2 images')
    if args.dataset_config.validation_cannyt2i_prompt:
        base_eval_prompts.append(args.dataset_config.validation_cannyt2i_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_cannyt2i_path)
        base_phase_names.append('vlm->generate image based on canny')
    if args.dataset_config.validation_it2canny_prompt:
        base_eval_prompts.append(args.dataset_config.validation_it2canny_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_it2canny_path)
        base_phase_names.append('vlm->generate canny')
    if args.dataset_config.validation_poset2i_prompt:
        base_eval_prompts.append(args.dataset_config.validation_poset2i_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_poset2i_path)
        base_phase_names.append('vlm->generate image based on pose')
    if args.dataset_config.validation_it2pose_prompt:
        base_eval_prompts.append(args.dataset_config.validation_it2pose_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_it2pose_path)
        base_phase_names.append('vlm->generate pose')
    if args.dataset_config.validation_NIKEit2i_prompt:
        base_eval_prompts.append(args.dataset_config.validation_NIKEit2i_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_NIKEit2i_path)
        base_phase_names.append('vlm->edit nike')
    
    if args.dataset_config.validation_TRANSFERit2i_prompt:
        base_eval_prompts.append(args.dataset_config.validation_TRANSFERit2i_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_TRANSFERit2i_path)
        base_phase_names.append('vlm->transfer')

    if args.dataset_config.validation_EXTRACTit2i_prompt:
        base_eval_prompts.append(args.dataset_config.validation_EXTRACTit2i_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_EXTRACTit2i_path)
        base_phase_names.append('vlm->extract')
    if args.dataset_config.validation_TRYONit2i_prompt:
        base_eval_prompts.append(args.dataset_config.validation_TRYONit2i_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_TRYONit2i_path)
        base_phase_names.append('vlm->try on')

    if args.dataset_config.validation_REPLACEit2i_prompt:
        base_eval_prompts.append(args.dataset_config.validation_REPLACEit2i_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_REPLACEit2i_path)
        base_phase_names.append('vlm->replace')

    if args.dataset_config.validation_DETit2i_prompt:
        base_eval_prompts.append(args.dataset_config.validation_DETit2i_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_DETit2i_path)
        base_phase_names.append('vlm->detect')

    if args.dataset_config.validation_SEGit2i_prompt:
        base_eval_prompts.append(args.dataset_config.validation_SEGit2i_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_SEGit2i_path)
        base_phase_names.append('vlm->segment')
    
    if args.dataset_config.validation_REFiit2i_prompt:
        base_eval_prompts.append(args.dataset_config.validation_REFiit2i_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_REFiit2i_path)
        base_phase_names.append('vlm->transfer based on ref-style ')
    return base_eval_prompts, base_eval_image_paths, base_phase_names

# deepspeed.init_distributed()
def create_ema_model(
        accelerator, 
        args, 
        resume_checkpoint_path, 
        model_cls,
        model_config,
        ema_model_state_dict,
        ds_config=None, 
        ):
    # model_config = AutoConfig.from_pretrained(model_name_or_path)
    ds_config["train_micro_batch_size_per_gpu"] = args.dataset_config.batch_size
    ds_config["fp16"]["enabled"] = False
    ds_config["bf16"]["enabled"] = False
    ds_config["gradient_accumulation_steps"] = args.training_config.gradient_accumulation_steps
    ds_config["train_batch_size"] = args.dataset_config.batch_size * args.training_config.gradient_accumulation_steps * accelerator.num_processes

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    accelerator.print(f'EMA deepspeed config {ds_config}')
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
            
    if resume_checkpoint_path:
        ema_model_path = os.path.join(resume_checkpoint_path, "model_ema")
        if os.path.exists(ema_model_path):
            ema_model = EMAModel.from_pretrained(ema_model_path, model_cls=model_cls)
            accelerator.print(f'Successully resume EMAModel from {ema_model_path}')
    else:
        # we load weights from original model instead of deepcopy
        # model = model_cls.from_config(model_config)
        # accelerator.print('init model', model)
        # for k, v in model.state_dict().items():
        #     accelerator.print(k, v.shape)
        # model.load_state_dict(ema_model_state_dict, strict=True)
        model = model_cls.from_pretrained(
            args.model_config.ema_pretrained_lvlm_name_or_path,
            # config=lvlm_model.config,
            # deepspeed=dschf.to_dict(),    # 关键参数
            torch_dtype=torch.float32,           # fp32
        )
        accelerator.print(f"model_cls.from_pretrained finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")
        model.eval().requires_grad_(False)
        model.to(accelerator.device)
        # model.config.hidden_size = 4096
        ema_model = EMAModel(
            model, decay=args.training_config.ema_decay,
            model_cls=model_cls, model_config=model_config
            )
        accelerator.print(f"EMAModel finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")
        accelerator.print(f'Successully deepcopy EMAModel from model')
    # from deepspeed.runtime.zero import Init as DSZeroInit
    # with DSZeroInit(config=ds_config):
    ema_model.model, _, _, _ = deepspeed.initialize(model=ema_model.model, config_params=ds_config)
    return ema_model

def main(args: UnivaTrainingDenoiseConfig, attn_implementation='sdpa'):
    # Prepare accelerator
    logging_dir = Path(
        args.training_config.output_dir, args.training_config.logging_dir
    )
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.training_config.output_dir, logging_dir=logging_dir
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.training_config.gradient_accumulation_steps,
        mixed_precision=args.training_config.mixed_precision,
        log_with=args.training_config.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    # Set seed
    set_seed(args.training_config.seed, device_specific=True)

    # Create output directory
    if accelerator.is_main_process:
        if args.training_config.output_dir is not None:
            os.makedirs(args.training_config.output_dir, exist_ok=True)

    # Set weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif accelerator.mixed_precision == "no":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Resume from checkpoint
    resume_checkpoint_path = None
    if args.training_config.resume_from_checkpoint:
        if args.training_config.resume_from_checkpoint != "latest":
            path = os.path.basename(args.training_config.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.training_config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.training_config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.training_config.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            # accelerator.load_state(os.path.join(args.training_config.output_dir, path))
            resume_checkpoint_path = os.path.join(args.training_config.output_dir, path)
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            # first_epoch = global_step // num

    else:
        initial_global_step = 0


    # Load models
    lvlm_model = UnivaDenoiseTower.from_pretrained(
        args.model_config.pretrained_lvlm_name_or_path,
        attn_implementation=attn_implementation,
    )
    accelerator.print(f'{lvlm_model}')
    lvlm_tokenizer = None

    siglip_processor, siglip_model = None, None
    from transformers import SiglipImageProcessor, SiglipVisionModel
    siglip_processor = SiglipImageProcessor.from_pretrained(
        args.model_config.pretrained_siglip_name_or_path
        )
    siglip_processor.size = {"height": args.dataset_config.height, "width": args.dataset_config.width}
    siglip_model = SiglipVisionModel.from_pretrained(
        args.model_config.pretrained_siglip_name_or_path, 
        attn_implementation=attn_implementation,
        )
    accelerator.print(f'{siglip_model}')

    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.model_config.pretrained_denoiser_name_or_path,
        subfolder="tokenizer",
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.model_config.pretrained_denoiser_name_or_path,
        subfolder="tokenizer_2",
    )

    text_encoder_cls_one = CLIPTextModel.from_pretrained(
        args.model_config.pretrained_denoiser_name_or_path,
        subfolder="text_encoder",
    )
    text_encoder_cls_two = T5EncoderModel.from_pretrained(
        args.model_config.pretrained_denoiser_name_or_path,
        subfolder="text_encoder_2",
    )

    vae = AutoencoderKL.from_pretrained(
        args.model_config.pretrained_denoiser_name_or_path,
        subfolder="vae",
    )

    accelerator.print(f'{text_encoder_cls_two}')
    accelerator.print(f'{vae}')
    # Load scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_config.pretrained_denoiser_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # Move models to device and set grad
    vae_dtype = torch.float32 if args.model_config.vae_fp32 else weight_dtype
    vae.to(accelerator.device, dtype=vae_dtype)
    accelerator.print(f"Load vae model finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")

    text_encoder_cls_one.to(accelerator.device, dtype=weight_dtype)
    accelerator.print(f"Load text_encoder_cls_one model finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")

    text_encoder_cls_two.to(accelerator.device, dtype=weight_dtype)
    accelerator.print(f"Load text_encoder_cls_two model finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")

    lvlm_model.to(accelerator.device, dtype=weight_dtype)
    accelerator.print(f"Load main model finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")

    siglip_model.to(accelerator.device, dtype=weight_dtype)
    accelerator.print(f"Load siglip model finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")

    vae.requires_grad_(False)
    text_encoder_cls_one.requires_grad_(False)
    text_encoder_cls_two.requires_grad_(False)
    lvlm_model.requires_grad_(False)
    lvlm_model.denoiser.requires_grad_(False)
    siglip_model.requires_grad_(False)
    # siglip_model = torch.compile(siglip_model)
    # lvlm_model = torch.compile(lvlm_model)
    # vae = torch.compile(vae)


    if args.training_config.gradient_checkpointing:
        # lvlm_model._set_gradient_checkpointing()
        lvlm_model.denoiser.enable_gradient_checkpointing()

    # Setup model saving and loading
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                if isinstance(accelerator.unwrap_model(model), UnivaDenoiseTower):
                    accelerator.unwrap_model(model).save_pretrained(
                        os.path.join(output_dir, "univa"),
                    )
                    # processor.save_pretrained(
                    #     os.path.join(output_dir, "univa"),
                    # )
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")
                if weights:
                    weights.pop()
        if args.training_config.ema_deepspeed_config_file is not None:
            ema_model.save_pretrained(os.path.join(output_dir, "model_ema"))
            # if accelerator.is_main_process:
            #     processor.save_pretrained(os.path.join(output_dir, "model_ema"))

    # def load_model_hook(models, input_dir):
    #     for _ in range(len(models)):
    #         model = models.pop()
    #         if isinstance(accelerator.unwrap_model(model), model_class):
    #             load_model = model_class.from_pretrained(
    #                 input_dir, subfolder="univa", 
    #                 attn_implementation=attn_implementation,
    #             )
    #             # model.register_to_config(**load_model.config)
    #             model.load_state_dict(load_model.state_dict())
    #         else:
    #             raise ValueError(f"Unsupported model found: {type(model)=}")
    #         del load_model

    # if accelerator.distributed_type == DistributedType.MULTI_GPU:
    accelerator.register_save_state_pre_hook(save_model_hook)
    # accelerator.register_load_state_pre_hook(load_model_hook)

    '''
    all 

    pos_embed time_text_embed context_embedder

    norm1.linear
    norm1_context.linear
    attn.to_q attn.to_k attn.to_v attn.to_out
    attn.add_k_proj attn.add_v_proj attn.add_q_proj attn.to_add_out
    ff.net ff_context.net

    norm_out.linear proj_out
    '''
    # denoise_key_to_train for sd3.5 
    # context_embedder attn.to_q attn.to_k attn.to_v attn.to_out norm1.linear norm1_context.linear
    '''
    all 

    pos_embed time_text_embed context_embedder

    norm1.linear
    norm1_context.linear
    attn.to_q attn.to_k attn.to_v attn.to_out attn.norm_q attn.norm_k 
    attn.add_k_proj attn.add_v_proj attn.add_q_proj attn.to_add_out attn.norm_added_q attn.norm_added_k
    attn2.to_q attn2.to_k attn2.to_v attn2.to_out attn2.norm_q attn2.norm_k
    ff.net ff_context.net

    norm_out.linear proj_out
    '''
    # denoise_key_to_train for flux
    # context_embedder attn.norm_q attn.norm_k attn.to_q attn.to_k attn.to_v attn.to_out norm1.linear norm1_context.linear norm.linear
    '''
    all

    x_embedder time_text_embed context_embedder
    -----------↓↓↓ for two-branch----------------
    norm1.linear
    norm1_context.linear
    attn.norm_q attn.norm_k attn.to_q attn.to_k attn.to_v attn.to_out
    attn.add_k_proj attn.add_v_proj attn.add_q_proj attn.to_add_out attn.norm_added_q attn.norm_added_k
    ff.net ff_context.net
    -----------↓↓↓ for single-branch----------------
    norm.linear
    attn.norm_q attn.norm_k attn.to_q attn.to_k attn.to_v
    proj_mlp proj_out
    -----------------
    norm_out.linear proj_out
    '''
    
    for name, param in lvlm_model.named_parameters():
        if 'model.siglip_projector' in name:
            param.requires_grad_(False)


    if args.model_config.pretrained_siglip_mlp_path is not None:
        pretrained_siglip_mlp = torch.load(args.model_config.pretrained_siglip_mlp_path)
        pretrained_siglip_mlp = {k.replace('denoise_tower.', ''): v for k, v in pretrained_siglip_mlp.items()}
        if accelerator.is_main_process:
            accelerator.print(f'Load {[k for k in pretrained_siglip_mlp.keys()]} from {args.model_config.pretrained_siglip_mlp_path}')
        msg = lvlm_model.load_state_dict(pretrained_siglip_mlp, strict=False)
        assert len(msg[1]) == 0, msg

    if args.model_config.only_tune_siglip_mlp:
        lvlm_model.requires_grad_(False)
        if args.model_config.only_tune_siglip_mlp:
            for name, param in lvlm_model.named_parameters():
                if 'model.siglip_projector' in name:
                    param.requires_grad_(True)
        else:
            raise ValueError('NOT both support only_tune_mlp2 and only_tune_mlp3 and only_tune_siglip_mlp')
    else:
        if args.model_config.flux_train_layer_idx is not None:
            trainable_components = get_trainable_params(
                layers_to_train=args.model_config.flux_train_layer_idx, 
                only_img_branch=args.model_config.only_tune_image_branch, 
            )
            for name, module in lvlm_model.named_modules():
                if check_param_is_in_components(
                    name, trainable_components
                ):
                    module.requires_grad_(True)

    if args.model_config.with_tune_siglip_mlp:
        for name, param in lvlm_model.named_parameters():
            if 'siglip_projector' in name:
                param.requires_grad_(True)
        

    # =======================================================================================================
    # STEP 6: Create EMAModel
    if args.training_config.ema_deepspeed_config_file is not None:
        ema_model_state_dict = lvlm_model.state_dict()
        with open(args.training_config.ema_deepspeed_config_file, 'r') as f:
            ds_config = json.load(f)
        ema_model = create_ema_model(
            accelerator, args, resume_checkpoint_path, model_cls=UnivaDenoiseTower, model_config=lvlm_model.config, 
            ema_model_state_dict=ema_model_state_dict, ds_config=ds_config
            )

        accelerator.print(f"Load ema model finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")
    # =======================================================================================================


    # Load optimizer
    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    trainable_names = []
    trainable_params = []
    for name, param in lvlm_model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            trainable_names.append(name)
    if accelerator.is_main_process:
        with open("trainable_params.txt", "w") as f:
            for name in trainable_names:
                f.write(f"{name}\n")
                print(f"{name}.requires_grad: True")
        accelerator.print("Trainable params:", len(trainable_params))
    if use_deepspeed_optimizer:
        from accelerate.utils import DummyOptim

        optimizer = DummyOptim(
            trainable_params,
            lr=args.training_config.learning_rate,
            betas=(args.training_config.adam_beta1, args.training_config.adam_beta2),
            eps=args.training_config.adam_epsilon,
        )
    else:
        if args.training_config.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=args.training_config.learning_rate,
                betas=(args.training_config.adam_beta1, args.training_config.adam_beta2),
                eps=args.training_config.adam_epsilon,
                weight_decay=args.training_config.adam_weight_decay,
            )
        elif args.training_config.optimizer.lower() == "prodigy":
            try:
                import prodigyopt
            except ImportError:
                raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

            if args.training_config.learning_rate <= 0.1:
                raise ValueError(
                    "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
                )

            optimizer = prodigyopt.Prodigy(
                trainable_params,
                betas=(args.training_config.adam_beta1, args.training_config.adam_beta2),
                beta3=args.training_config.prodigy_beta3,
                weight_decay=args.training_config.adam_weight_decay,
                eps=args.training_config.adam_epsilon,
                decouple=args.training_config.prodigy_decouple,
                use_bias_correction=args.training_config.prodigy_use_bias_correction,
                safeguard_warmup=args.training_config.prodigy_safeguard_warmup,
                d_coef=args.training_config.prodigy_d_coef,
            )

    # Load dataset
    transform = transforms.Compose(
        transforms=[
            transforms.Resize((args.dataset_config.height, args.dataset_config.width)),
            transforms.Normalize([0.5], std=[0.5]),
        ]
    )
    data_collator = DataCollator()
    dataset = ReduxDataset(
        data_txt=args.dataset_config.data_txt,
        image_processors=[siglip_processor],
        image_transform=transform,
    )

    # Create dataloader
    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.dataset_config.batch_size,
        shuffle=True,
        pin_memory=args.dataset_config.pin_memory,
        num_workers=args.dataset_config.num_workers,
        collate_fn=data_collator,
        prefetch_factor=None if args.dataset_config.num_workers == 0 else 4, 
        # prefetch_factor=None, 
        # persistent_workers=True, 
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.training_config.gradient_accumulation_steps
    )
    if args.training_config.max_train_steps is None:
        args.training_config.max_train_steps = (
            args.training_config.num_train_epochs * num_update_steps_per_epoch
        )
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler

        lr_scheduler = DummyScheduler(
            name=args.training_config.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=args.training_config.max_train_steps
            * accelerator.num_processes,
            num_warmup_steps=args.training_config.lr_warmup_steps
            * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            args.training_config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.training_config.lr_warmup_steps
            * accelerator.num_processes,
            num_training_steps=args.training_config.max_train_steps
            * accelerator.num_processes,
            num_cycles=args.training_config.lr_num_cycles,
            power=args.training_config.lr_power,
        )

    # Prepare training
    device_placement = None
    if accelerator.distributed_type != DistributedType.DEEPSPEED:
        device_placement = [True, True, True, False]
    lvlm_model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        lvlm_model, optimizer, lr_scheduler, train_dataloader, 
        device_placement=device_placement
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.training_config.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.training_config.max_train_steps = (
            args.training_config.num_train_epochs * num_update_steps_per_epoch
        )
    # Afterwards we recalculate our number of training epochs
    args.training_config.num_train_epochs = math.ceil(
        args.training_config.max_train_steps / num_update_steps_per_epoch
    )
    print('OmegaConf.to_container(args, resolve=True)', OmegaConf.to_container(args, resolve=True))
    if accelerator.is_main_process:
        accelerator.init_trackers(
            args.training_config.wandb_project,
            init_kwargs={"wandb": {"name": args.training_config.wandb_name}},
            config=OmegaConf.to_container(args, resolve=True)
        )

    total_batch_size = (
        args.dataset_config.batch_size
        * accelerator.num_processes
        * args.training_config.gradient_accumulation_steps
    )

    accelerator.print("***** Running training *****")
    accelerator.print(f"  Num examples = {len(train_dataloader)}")
    accelerator.print(f"  Num batches each epoch = {len(train_dataloader)}")
    accelerator.print(f"  Num Epochs = {args.training_config.num_train_epochs}")
    accelerator.print(f"  Instantaneous batch size per device = {args.dataset_config.batch_size}")
    accelerator.print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    accelerator.print(
        f"  Gradient Accumulation steps = {args.training_config.gradient_accumulation_steps}"
    )
    accelerator.print(f"  Total optimization steps = {args.training_config.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Resume from checkpoint
    if resume_checkpoint_path is not None:
        accelerator.load_state(resume_checkpoint_path)
        first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.training_config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [
        text_encoder_cls_one,
        text_encoder_cls_two,
    ]
    empty_t5_prompt_embeds, empty_pooled_prompt_embeds = encode_prompt(
        text_encoders,
        tokenizers,
        prompt="",
        max_sequence_length=256,
        device=accelerator.device,
        num_images_per_prompt=1,
    )
    empty_pooled_prompt_embeds = empty_pooled_prompt_embeds.repeat(
        args.dataset_config.batch_size, 1
    )

    if args.training_config.drop_t5_rate == 1.0:
        del text_encoders, text_encoder_cls_one, text_encoder_cls_two
        free_memory()

    prof = None
    if args.training_config.profile_out_dir is not None:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
                ], 
            schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=2, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.training_config.profile_out_dir),
            profile_memory=True,
            with_stack=True,
            record_shapes=True
            )
    

    latent_image_ids_dict = {}
    for epoch in range(first_epoch, args.training_config.num_train_epochs):
        lvlm_model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(lvlm_model):
                generated_image = batch["generated_image"].to(
                    accelerator.device, dtype=vae.dtype
                )    
                
                siglip_pixel_values = batch["image_0"].to(
                    device=accelerator.device, dtype=siglip_model.dtype, non_blocking=True
                    )
                batch_size, channels, height, width = siglip_pixel_values.shape
                siglip_pixel_values = rearrange(
                    siglip_pixel_values, "b c (h2 h) (w2 w) -> (b h2 w2) c h w", h2=2, w2=2
                )
                
                with torch.no_grad():
                    siglip_hidden_states = siglip_model(siglip_pixel_values).last_hidden_state
                
                siglip_hidden_states = rearrange(
                    siglip_hidden_states,
                    "(b h2 w2) (h w) c -> b (h2 h w2 w) c",
                    h2=2,
                    w2=2,
                    h=height // siglip_model.config.patch_size // 2,
                    w=width // siglip_model.config.patch_size // 2,
                )
                # VAE encode
                def vae_encode(x):
                    with torch.no_grad():
                        model_input = vae.encode(x).latent_dist.sample()
                    model_input = (
                        model_input - vae.config.shift_factor
                    ) * vae.config.scaling_factor
                    model_input = model_input
                    return model_input


                model_input = vae_encode(generated_image)

                vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

                if model_input.shape in latent_image_ids_dict:
                    latent_image_ids = latent_image_ids_dict[model_input.shape]
                else:
                    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                        model_input.shape[0],
                        model_input.shape[2] // 2,
                        model_input.shape[3] // 2,
                        accelerator.device,
                        weight_dtype,
                    )
                    latent_image_ids_dict[model_input.shape] = latent_image_ids

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]


                def calculate_shift(
                    image_seq_len,
                    base_seq_len: int = 256,
                    max_seq_len: int = 4096,
                    base_shift: float = 0.5,
                    max_shift: float = 1.16,
                ):
                    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
                    b = base_shift - m * base_seq_len
                    mu = image_seq_len * m + b
                    return mu

                def apply_flux_schedule_shift(sigmas, noise):
                    # Resolution-dependent shifting of timestep schedules as per section 5.3.2 of SD3 paper
                    # Resolution-dependent shift value calculation used by official Flux inference implementation
                    image_seq_len = (noise.shape[-1] * noise.shape[-2]) // 4
                    mu = calculate_shift(
                        image_seq_len,
                        noise_scheduler_copy.config.base_image_seq_len,
                        noise_scheduler_copy.config.max_image_seq_len,
                        noise_scheduler_copy.config.base_shift,
                        noise_scheduler_copy.config.max_shift,
                    )
                    shift = math.exp(mu)
                    sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
                    return sigmas

                sigmas = torch.sigmoid(
                    1.0 * torch.randn((bsz,), device=model_input.device, dtype=torch.float32)
                )
                sigmas = apply_flux_schedule_shift(sigmas, noise)
                timesteps = sigmas * 1000.0  # rescale to [0, 1000.0)
                while sigmas.ndim < model_input.ndim:
                    sigmas = sigmas.unsqueeze(-1)
                # print(f'STEP[{global_step}]-RANK[{accelerator.process_index}], sigmas={sigmas}, noise: max {noise.max()}, min {noise.min()}, mean {noise.mean()}, std {noise.std()}')
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )
                # 优化写法，避免 Python list 构造
                guidance = torch.full(
                    (model_input.shape[0],),  # 直接一次性创建所有
                    fill_value=args.model_config.guidance_scale,
                    device=accelerator.device,
                )
                siglip_hidden_states = lvlm_model.module.siglip_projector(siglip_hidden_states)
                encoder_hidden_states = torch.cat([empty_t5_prompt_embeds, siglip_hidden_states], dim=1)
                
                denoiser_kwargs = {
                        "encoder_hidden_states": encoder_hidden_states,
                        "hidden_states": packed_noisy_model_input.to(weight_dtype),
                        "timestep": (timesteps / 1000).to(weight_dtype),
                        "guidance": guidance,
                        "pooled_projections": empty_pooled_prompt_embeds,
                        "img_ids": latent_image_ids,
                        "joint_attention_kwargs": {}
                    }
                
                with torch.amp.autocast("cuda", dtype=weight_dtype):
                    model_pred = lvlm_model(**denoiser_kwargs)

                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    height=model_input.shape[2] * vae_scale_factor,
                    width=model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                target = noise - model_input
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.training_config.weighting_scheme,
                    sigmas=sigmas,
                )

                loss = (
                        weighting.float() * (model_pred.float() - target.float()) ** 2
                    ).reshape(target.shape[0], -1)
                
                loss = loss.mean()
                avg_loss_list = accelerator.gather(loss)
                accelerator.backward(loss)
                # import ipdb;ipdb.set_trace()
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params, args.training_config.max_grad_norm
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:        
                if args.training_config.ema_deepspeed_config_file is not None and global_step % args.training_config.ema_update_freq == 0:
                    ema_model.step(lvlm_model.parameters())

                progress_bar.update(1)
                global_step += 1

                if (
                    accelerator.is_main_process
                    or accelerator.distributed_type == DistributedType.DEEPSPEED
                ) and global_step % args.training_config.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.training_config.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.training_config.output_dir)
                        checkpoints = [
                            d for d in checkpoints if d.startswith("checkpoint")
                        ]
                        checkpoints = sorted(
                            checkpoints, key=lambda x: int(x.split("-")[1])
                        )
                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if (
                            len(checkpoints)
                            >= args.training_config.checkpoints_total_limit
                        ):
                            num_to_remove = (
                                len(checkpoints)
                                - args.training_config.checkpoints_total_limit
                                + 1
                            )
                            removing_checkpoints = checkpoints[0:num_to_remove]
                            accelerator.print(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            accelerator.print(
                                f"removing checkpoints: {', '.join(removing_checkpoints)}"
                            )
                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(
                                    args.training_config.output_dir,
                                    removing_checkpoint,
                                )
                                shutil.rmtree(removing_checkpoint)
                    save_path = os.path.join(
                        args.training_config.output_dir, f"checkpoint-{global_step}"
                    )
                    accelerator.save_state(save_path)
                    if args.model_config.only_tune_siglip_mlp or args.model_config.with_tune_siglip_mlp:
                        keys_to_match = ['siglip_projector']
                        weight_siglip_mlp = get_mm_adapter_state_maybe_zero_3(lvlm_model.named_parameters(), keys_to_match)
                        weight_siglip_mlp = {k.replace('module.', ''): v for k, v in weight_siglip_mlp.items()}
                        torch.save(weight_siglip_mlp, os.path.join(save_path, 'siglip_projector.bin'))
                    accelerator.print(f"Saved state to {save_path}")

                # num_machines = accelerator.state.num_processes // 8
                # node_rank = accelerator.process_index // 8
                # print(f'node_rank: {node_rank}, num_machines: {num_machines}')
                # num_run_per_node = 8 // num_run_per_node
                if global_step % args.training_config.validation_steps == 0:
                    base_eval_image_paths = args.dataset_config.validation_redux_path
                if accelerator.is_main_process and global_step % args.training_config.validation_steps == 0:
                    if len(base_eval_image_paths) > 0:

                        unwrapped_lvlm_model = accelerator.unwrap_model(lvlm_model)
                        pipe = FluxPipeline.from_pretrained(
                            args.model_config.pretrained_denoiser_name_or_path,
                            transformer=None,
                            vae=vae,
                            text_encoder=None,
                            text_encoder_2=None,
                            torch_dtype=weight_dtype,
                        )
                        pipe.to(accelerator.device)
                        pipe.transformer = unwrapped_lvlm_model.denoiser

                        for image_path in base_eval_image_paths:
                            log_validation(
                                accelerator=accelerator,
                                weight_dtype=weight_dtype,
                                args=args,
                                siglip_model=siglip_model,
                                siglip_processor=siglip_processor,
                                pipe=pipe,
                                unwrapped_lvlm_model=unwrapped_lvlm_model,
                                image_path=image_path,
                                title="rec",
                            )
                    

                    if len(base_eval_image_paths) > 0:
                        
                        # if args.training_config.ema_deepspeed_config_file is not None:
                        #     ema_model.restore(lvlm_model.parameters())
                        del pipe
                        free_memory()

            if prof is not None:
                prof.step()

            log_interval = 1
            if global_step % log_interval == 0:
                logs = {
                    # "loss": loss.detach().item(),
                    "loss": avg_loss_list.mean().detach().item(), 
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                if args.training_config.optimizer.lower() == "prodigy":
                    d = optimizer.param_groups[0]['d']
                    beta1, beta2 = optimizer.param_groups[0]['betas']
                    k = optimizer.param_groups[0]['k']
                    lr = max(group['lr'] for group in optimizer.param_groups)
                    d_lr = d * lr
                    bias_correction = ((1 - beta2**(k+1))**0.5) / (1 - beta1**(k+1))
                    d_lr_bias_corr = d_lr * bias_correction
                    prodigy_log = {"d*lr": d_lr, "d*lr*bias_corr": d_lr_bias_corr}
                    logs.update(prodigy_log)
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step >= args.training_config.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()

    
@torch.no_grad()
def log_validation(
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    args: UnivaTrainingDenoiseConfig,
    siglip_model: Optional[Callable] = None,
    siglip_processor: Optional[Callable] = None,
    pipe: Optional[Callable] = None,
    unwrapped_lvlm_model: Optional[Callable] = None,
    image_path: Optional[str] = None,
    title: str = "",
):
    image = Image.open(image_path).convert('RGB')
    pixel_values = siglip_processor(images=image, return_tensors='pt', do_resize=True, do_center_crop=True).pixel_values
    pixel_values = pixel_values.to(accelerator.device, dtype=siglip_model.dtype)
    
    batch_size, channels, height, width = pixel_values.shape

    pixel_values = rearrange(
        pixel_values, "b c (h2 h) (w2 w) -> (b h2 w2) c h w", h2=2, w2=2
    )
    features = siglip_model(pixel_values=pixel_values).last_hidden_state
    features = features.to(dtype=weight_dtype)
            
    features = rearrange(
        features,
        "(b h2 w2) (h w) c -> b (h2 h w2 w) c",
        h2=2,
        w2=2,
        h=height // siglip_model.config.patch_size // 2,
        w=width // siglip_model.config.patch_size // 2,
    )
    
    encoder_hidden_states = unwrapped_lvlm_model.siglip_projector(features)
        
    empty_t5_prompt_embeds = torch.zeros(1, 512, 4096, device=accelerator.device)
    empty_pooled_prompt_embeds = torch.zeros(1, 768, device=accelerator.device)
    
    encoder_hidden_states = torch.cat([empty_t5_prompt_embeds, encoder_hidden_states], dim=1)
    
    generator = (
        torch.Generator(device=accelerator.device).manual_seed(
            args.training_config.seed,
        )
        if args.training_config.seed is not None
        else None
    )
    autocast_ctx = torch.amp.autocast("cuda", dtype=weight_dtype)
    with autocast_ctx:
        images = [
            pipe(
                prompt_embeds=encoder_hidden_states,
                num_inference_steps=15,
                pooled_prompt_embeds=empty_pooled_prompt_embeds,
                height=args.dataset_config.height,
                width=args.dataset_config.width,
                generator=generator
            ).images[0]
            for _ in range(args.training_config.num_validation_images)
        ]

    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            phase_name = f"validation_{title}"
            if tracker.name == "wandb":
                tracker.log(
                    {
                        phase_name: [
                            wandb.Image(image, caption=f"{i}")
                            for i, image in enumerate(images)
                        ]
                    }
                )

    # del pipe
    # free_memory()

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    schema = OmegaConf.structured(UnivaTrainingDenoiseConfig)
    conf = OmegaConf.merge(schema, config)
    # main(conf, attn_implementation='sdpa')
    main(conf, attn_implementation='flash_attention_2')
