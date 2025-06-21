from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TrainingConfig:
    seed: int = 42
    wandb_project: str = "univa-denoiser"
    wandb_name: str = "default_config"
    output_dir: str = "./output"
    logging_dir: str = "./logs"
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    adam_weight_decay: float = 1e-2
    mixed_precision: str = "bf16"
    report_to: str = "wandb"
    gradient_checkpointing: bool = False
    num_train_epochs: int = 1
    max_train_steps: Optional[int] = None
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    resume_from_checkpoint: Optional[str] = None
    weighting_scheme: Optional[str] = (
        "logit_normal"  # ["sigma_sqrt", "logit_normal", "mode", "cosmap", "null"]
    )
    logit_mean: float = 0.0
    logit_std: float = 1.0
    mode_scale: float = 1.29
    max_grad_norm: Optional[float] = None
    checkpointing_steps: int = 100
    checkpoints_total_limit: Optional[int] = 500
    drop_condition_rate: float = 0.0
    drop_t5_rate: float = 1.0
    validation_steps: int = 100
    num_validation_images: int = 1

    noise_reference_images: bool = False 
    mask_weight_type: Optional[str] = None     # ['log', 'exp']
    sigmas_as_weight: bool = False  # Used in Flux

    discrete_timestep: bool = True  # Used in Flux

    optimizer: str = 'adamw'  # ['adamw', 'prodigy']

    prodigy_use_bias_correction: bool = True
    prodigy_safeguard_warmup: bool = True
    prodigy_decouple: bool = True
    prodigy_beta3: Optional[float] = None 
    prodigy_d_coef: float = 1.0

    profile_out_dir: Optional[str] = None

    ema_deepspeed_config_file: Optional[str] = None
    ema_update_freq: int = 1
    ema_decay: float = 0.999

    drop_lora_rate: float = 0.0
    drop_lora_layer_wise: bool = False

    ce_loss_weight: float = 1.0

    ema_siglip_lora: bool = False

@dataclass
class DatasetConfig:
    dataset_type: str
    data_txt: str
    batch_size: int = 16
    num_workers: int = 4
    height: int = 512
    width: int = 512
    min_pixels: int = 448*448
    max_pixels: int = 448*448
    anyres: str = 'any_1ratio'
    ocr_enhancer: bool = False
    random_data: bool = False
    padding_side: str = 'right'
    validation_t2i_prompt: Optional[str] = None
    validation_it2i_prompt: Optional[str] = None
    validation_image_path: Optional[str] = None
    pin_memory: bool = True
    validation_iit2i_prompt: Optional[str] = None
    validation_iit2i_path: Optional[List[str]] = None

    validation_REFiit2i_prompt: Optional[str] = None
    validation_REFiit2i_path: Optional[List[str]] = None


    validation_cannyt2i_prompt: Optional[str] = None
    validation_cannyt2i_path: Optional[str] = None
    validation_poset2i_prompt: Optional[str] = None
    validation_poset2i_path: Optional[str] = None
    
    validation_it2pose_prompt: Optional[str] = None
    validation_it2pose_path: Optional[str] = None
    validation_it2canny_prompt: Optional[str] = None
    validation_it2canny_path: Optional[str] = None

    validation_NIKEit2i_prompt: Optional[str] = None
    validation_NIKEit2i_path: Optional[str] = None

    validation_TRANSFERit2i_prompt: Optional[str] = None
    validation_TRANSFERit2i_path: Optional[str] = None

    validation_EXTRACTit2i_prompt: Optional[str] = None
    validation_EXTRACTit2i_path: Optional[str] = None

    validation_TRYONit2i_prompt: Optional[str] = None
    validation_TRYONit2i_path: Optional[str] = None

    validation_REPLACEit2i_prompt: Optional[str] = None
    validation_REPLACEit2i_path: Optional[str] = None

    validation_DETit2i_prompt: Optional[str] = None
    validation_DETit2i_path: Optional[str] = None

    validation_SEGit2i_prompt: Optional[str] = None
    validation_SEGit2i_path: Optional[str] = None

    validation_redux_path: Optional[List[str]] = None

@dataclass
class ModelConfig:
    pretrained_lvlm_name_or_path: str
    pretrained_denoiser_name_or_path: str
    pretrained_siglip_name_or_path: Optional[str] = None

    pretrained_lora_siglip_name_or_path: Optional[str] = None
    ema_pretrained_lora_siglip_name_or_path: Optional[str] = None

    train_vision_tower_mm_projector: bool = False

    guidance_scale: float = 1.0  # Used in Flux
    tune_mlp1_only: bool = False
    pretrained_mlp1_path: Optional[str] = None

    with_tune_mlp2: bool = False
    only_tune_mlp2: bool = False
    pretrained_mlp2_path: Optional[str] = None

    only_tune_image_branch: bool = True  # Used in SD3

    with_tune_mlp3: bool = False
    only_tune_mlp3: bool = False
    pretrained_mlp3_path: Optional[str] = None

    flux_train_layer_idx: Optional[list] = None

    with_tune_siglip_mlp: bool = False
    only_tune_siglip_mlp: bool = False
    pretrained_siglip_mlp_path: Optional[str] = None

    joint_ref_feature: bool = False
    joint_ref_feature_as_condition: bool = False
    only_use_t5: bool = False

    vlm_residual_image_factor: float = 0.0

    vae_fp32: bool = True
    compile_flux: bool = False
    compile_qwen2p5vl: bool = False

    ema_pretrained_lvlm_name_or_path: Optional[str] = None

    lora_r: int = -1
    lora_alpha: int = -1
    lora_dropout: float = 0.0
    lora_target_modules: Optional[List[str]] = field(
        default_factory=lambda: ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']
    )

    with_tune_pretrained_redux_mlp: bool = False
    with_tune_aspect_ratio_embedder: bool = True

    
    lora_r_for_vlm: int = -1
    lora_alpha_for_vlm: int = -1
    lora_dropout_for_vlm: float = 0.0
    lora_target_modules_for_vlm: Optional[List[str]] = field(
        default_factory=lambda: ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']
    )

@dataclass
class UnivaTrainingDenoiseConfig:
    training_config: TrainingConfig
    dataset_config: DatasetConfig
    model_config: ModelConfig
