from univa.model.language_model.univa_qwen import UnivaQwen2ForCausalLM
from univa.model.multimodal_denoiser.builder import build_denoise_tower
from univa.model.multimodal_projector.builder import build_denoise_projector
from transformers import AutoTokenizer

model = UnivaQwen2ForCausalLM.from_pretrained("/mnt/data/checkpoints/Qwen/Qwen2.5-3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("/mnt/data/checkpoints/Qwen/Qwen2.5-3B-Instruct")

class DummyModelArguments():
    # ???
    mm_anyres: bool = False
    mm_anyres_max_pixels: int = 864 * 864
    mm_anyres_min_pixels: int = 320 * 320
    image_aspect_ratio: str = "pad"
    im_start_token: int = 151645
    img_token: int = 151645
    # Vision Input
    vision_tower: str = "/mnt/data/checkpoints/google/siglip-so400m-patch14-384"
    mm_vision_select_layer: int = -2
    mm_vision_select_feature: str = "patch"
    pretrain_mm_mlp_adapter: str = None
    mm_pixel_decoder: bool = False
    pretrain_mm_inv_mlp_adapter: str = None
    mm_eye_adapter: bool = False
    mm_eye_depth: int = 0
    mm_eye_weight: float = 1.0
    mm_eye_model_path: str = None
    pretrain_mm_eye_mlp_adapter: str = None
    mm_mask_adapter: bool = False
    pretrain_mm_mask_mlp_adapter: str = None
    mm_mask_ratio: float = 0.0
    mm_mask_depth: int = 0
    mm_mask_weight: float = 1.0
    use_mm_proj: bool = True
    mm_projector_type: str = "mlp2x_gelu"
    # Denoiser
    denoise_enable: bool = True
    mm_denoise_weight: float = 1.0
    denoise_tower: str = "/mnt/data/checkpoints/black-forest-labs/FLUX.1-dev"
    mm_denoise_projector_type: str = "mlp2x_gelu"
    pretrain_mm_denoise_mlp_adapter: str = None

model_args = DummyModelArguments()
model.get_model().initialize_vision_modules(model_args)

model.save_pretrained("models/univa_qwen2.5-3b-instruct")
tokenizer.save_pretrained("models/univa_qwen2.5-3b-instruct")