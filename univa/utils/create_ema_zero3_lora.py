import sys
sys.path.append(".")
import os
import json
import copy
import time
import torch
import deepspeed
from peft import LoraConfig, get_peft_model, PeftModel
from univa.utils.create_ema_zero3 import EMAModel_Zero3, _z3_params_to_fetch

# Adapted from diffusers-style ema https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py#L263
class EMAModel_Zero3_LoRA(EMAModel_Zero3):
    """
    Exponential Moving Average of models weights
    """
    def __init__(
        self,
        *args, 
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, path, model_cls, pretrained_lvlm_name_or_path) -> "EMAModel_Zero3_LoRA":

        model = model_cls.from_pretrained(pretrained_lvlm_name_or_path)
        model_lora = PeftModel.from_pretrained(model, os.path.join(path, 'lora'))
        ema_model = cls(model_lora, model_cls=model_cls, model_config=model_lora.config)

        with open(os.path.join(path, 'ema_kwargs.json'), 'r') as f:
            ema_kwargs = json.load(f)
        ema_model.load_state_dict(ema_kwargs)

        return ema_model

    def save_pretrained(self, path, pretrained_lvlm_name_or_path,  lora_config, lvlm_model_cpu=None):
        if self.model_cls is None:
            raise ValueError("`save_pretrained` can only be used if `model_cls` was defined at __init__.")

        if self.model_config is None:
            raise ValueError("`save_pretrained` can only be used if `model_config` was defined at __init__.")

        rank = int(os.getenv("RANK", "0"))
        state_dict = self.state_dict()
        state_dict.pop("model")

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_state_dict = {}
        for k, v in model_to_save.named_parameters():
            # only gather z3 params
            params_to_fetch = _z3_params_to_fetch([v])
            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
                vv = v.data.cpu()
                if rank == 0:
                    model_state_dict[k] = vv

        if rank == 0:
            os.makedirs(path, exist_ok=True)
            print(f'state_dict, {state_dict.keys()}')
            t_start = time.perf_counter()
            print(f"[{t_start:.4f}] self.model_cls.from_pretrained")

            print('self.model_cls', self.model_cls)
            if lvlm_model_cpu is None:
                model = self.model_cls.from_pretrained(pretrained_lvlm_name_or_path, torch_dtype=torch.float32)
                model = get_peft_model(model, lora_config)
            else:
                model = copy.deepcopy(lvlm_model_cpu)
            t1 = time.perf_counter()
            print(f"[{t1:.4f}] after self.model_cls.from_pretrained (耗时 {t1-t_start:.4f} 秒)")

            miss, unexp = model.load_state_dict(model_state_dict, strict=False)
            assert len(unexp) == 0, f'miss: {miss}; unexp: {unexp}'
            for i in miss:
                assert 'lm_head' in i
            model.save_pretrained(os.path.join(path, 'lora'))
            model_merge = model.merge_and_unload()
            model_merge.save_pretrained(path)
            t2 = time.perf_counter()      
            print(f"[{t2:.4f}] after save_pretrained (耗时 {t2-t1:.4f} 秒)")

            print(f"[{t2:.4f}] 总耗时 {t2-t_start:.4f} 秒")

            with open(os.path.join(path, 'ema_kwargs.json'), 'w') as f:
                json.dump(state_dict, f, indent=2)

        print(f'rank {rank}')
        return model_state_dict



if __name__ == "__main__":
    import sys
    import ipdb
    import json
    import deepspeed
    from transformers.integrations import HfDeepSpeedConfig
    from accelerate import Accelerator, DistributedDataParallelKwargs, DistributedType
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    from transformers import AutoProcessor

    deepspeed.init_distributed()
    GB = 1024 * 1024 * 1024
    def create_ema_model(
        accelerator, 
        model_cls,
        model_config,
        ema_model_state_dict,
        ds_config=None, 
        lora_config=None, 
        ):
        ds_config["train_micro_batch_size_per_gpu"] = 1
        ds_config["fp16"]["enabled"] = False
        ds_config["bf16"]["enabled"] = False
        ds_config["gradient_accumulation_steps"] = 1
        ds_config["train_batch_size"] = 1 * accelerator.num_processes

        # Note: dschf is defined in function scope to avoid global effects
        # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
        accelerator.print(f'EMA deepspeed config {ds_config}')
        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            dschf = HfDeepSpeedConfig(ds_config)
        else:
            dschf = None
                
        model = model_cls.from_pretrained(
            pretrained_name_or_path,
            torch_dtype=torch.float32,           # fp32
        )
        if lora_config is not None:
            model = get_peft_model(model, lora_config)
        ema_model = EMAModel_Zero3_LoRA(
            model, decay=0.99,
            model_cls=model_cls, model_config=model_config
            )
        accelerator.print(f"EMAModel_Zero3 finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")
        accelerator.print(f'Successully deepcopy EMAModel_Zero3 from model')

        ema_model.model, _, _, _ = deepspeed.initialize(model=ema_model.model, config_params=ds_config)
        accelerator.print(f"deepspeed.initialize finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")
        return ema_model

    ema_deepspeed_config_file = "scripts/accelerate_configs/zero3.json"
    pretrained_name_or_path = "/mnt/data/checkpoints/Qwen/Qwen2.5-VL-3B-Instruct"
    accelerator = Accelerator()
    model_cls = Qwen2_5_VLForConditionalGeneration

    lvlm_model = model_cls.from_pretrained(
        pretrained_name_or_path,
    )
    processor = AutoProcessor.from_pretrained(
        pretrained_name_or_path,
    )
    tokenizer = processor.tokenizer

    spacial_token = {
        'image_token': '<|image_pad|>', 
        'image_begin_token': '<|vision_start|>', 
        'image_end_token': '<|vision_end|>', 
        'think_begin_token': '<think_begin>', 
        'think_end_token': '<think_end>', 
        'think_token': '<think>', 
        'no_think_token': '<no_think>', 
    }
    for k, v in spacial_token.items():
        tokenizer.add_tokens([v], special_tokens=True)
    special_tokens = [tokenizer.convert_tokens_to_ids(v) for v in spacial_token.values()]
    
    lora_config_for_vlm = LoraConfig(
        r=256,
        lora_alpha=256,         
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        trainable_token_indices={'embed_tokens': special_tokens}, 
        lora_dropout=0.0,
    )   
    lvlm_model = get_peft_model(lvlm_model, lora_config_for_vlm)
    
    lvlm_model.to(device=accelerator.device, dtype=torch.bfloat16)
    accelerator.print(f"Load model finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")

    ema_model_state_dict = lvlm_model.state_dict()
    with open(ema_deepspeed_config_file, 'r') as f:
        ds_config = json.load(f)
    ema_model = create_ema_model(
        accelerator, model_cls=model_cls, model_config=lvlm_model.config, 
        ema_model_state_dict=ema_model_state_dict, ds_config=ds_config, 
        lora_config=lora_config_for_vlm, 
        )

'''

CUDA_VISIBLE_DEVICES=0 python univa/utils/create_ema_zero3_lora.py
accelerate launch --num_processes 2 --num_machines 1 univa/utils/create_ema_zero3_lora.py
'''
# if __name__ == "__main__":
#     import sys
#     sys.path.append('../..')
#     from univa.models.qwen2p5vl.modeling_univa_qwen2p5vl_v1_1 import UnivaQwen2p5VLForConditionalGeneration_V1_1
#     import ipdb
#     import json
#     import deepspeed
#     from transformers.integrations import HfDeepSpeedConfig
#     from accelerate import Accelerator, DistributedDataParallelKwargs, DistributedType
#     from accelerate import init_empty_weights

#     deepspeed.init_distributed()
#     GB = 1024 * 1024 * 1024
#     def create_ema_model(
#         accelerator, 
#         model_cls,
#         model_config,
#         ema_model_state_dict,
#         ds_config=None, 
#         lora_config=None, 
#         ):
#         # model_config = AutoConfig.from_pretrained(model_name_or_path)
#         ds_config["train_micro_batch_size_per_gpu"] = 1
#         ds_config["fp16"]["enabled"] = False
#         ds_config["bf16"]["enabled"] = False
#         ds_config["gradient_accumulation_steps"] = 1
#         ds_config["train_batch_size"] = 1 * accelerator.num_processes

#         # Note: dschf is defined in function scope to avoid global effects
#         # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
#         accelerator.print(f'EMA deepspeed config {ds_config}')
#         if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
#             dschf = HfDeepSpeedConfig(ds_config)
#         else:
#             dschf = None
                
#         # we load weights from original model instead of deepcopy
#         # model = model_cls.from_config(model_config)
#         # model.eval().requires_grad_(False)
#         # print('init model', model)
#         # print('model.device', model.device)
#         # accelerator.print(f"model_cls.from_config(model_config) finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")
#         # for k, v in model.state_dict().items():
#         #     print(k, v.shape)
            
#         # model.load_state_dict(ema_model_state_dict, strict=True)


#         model = model_cls.from_pretrained(
#             pretrained_lvlm_name_or_path,
#             # config=lvlm_model.config,
#             # deepspeed=dschf.to_dict(),    # 关键参数
#             torch_dtype=torch.float32,           # fp32
#         )
#         if lora_config is not None:
#             model = get_peft_model(model, lora_config)
#         # print('load_state_dict')
#         # print('after model.device', model.device)
#         accelerator.print(f"load_state_dict finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")
#         ema_model = EMAModel_Zero3_LoRA(
#             model, decay=0.99,
#             model_cls=model_cls, model_config=model_config
#             )
#         accelerator.print(f"EMAModel_Zero3 finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")
#         accelerator.print(f'Successully deepcopy EMAModel_Zero3 from model')

#         ema_model.model, _, _, _ = deepspeed.initialize(model=ema_model.model, config_params=ds_config)
#         accelerator.print(f"deepspeed.initialize finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")
#         return ema_model

#     ema_deepspeed_config_file = "scripts/accelerate_configs/zero3.json"
#     pretrained_lvlm_name_or_path = "/mnt/data/checkpoints/UniWorld_V1_1_Qwen2.5-VL-3B-Instruct_FLUX.1-dev"
#     accelerator = Accelerator()
#     model_cls = UnivaQwen2p5VLForConditionalGeneration_V1_1

#     # lvlm_model_ema_from_pt = EMAModel_Zero3_LoRA.from_pretrained('ema_model', model_cls, pretrained_lvlm_name_or_path)
#     lvlm_model = model_cls.from_pretrained(
#         pretrained_lvlm_name_or_path,
#         # 'ema_model',
#     )

#     regex = (
#         r"^(?:"
#         # q_proj/k_proj/v_proj/o_proj
#         r"model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
#         r"|"
#         # gate_proj/up_proj/down_proj
#         r"model\.layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)"
#         # r"|"
#         # embed_tokens
#         # r"model.embed_tokens"
#         r")$"
#     )
#     # special_tokens = [lvlm_tokenizer.convert_tokens_to_ids(v) for v in spacial_token.values()]
    
#     lora_config_for_vlm = LoraConfig(
#         r=256,
#         lora_alpha=256,         
#         target_modules=regex,
#         # modules_to_save=['embed_tokens', 'lm_head'], 
#         # trainable_token_indices={'embed_tokens': special_tokens}, 
#         lora_dropout=0.0,
#     )   
#     lvlm_model = get_peft_model(lvlm_model, lora_config_for_vlm)
    
#     print('after load', lvlm_model.dtype)
#     lvlm_model.to(device=accelerator.device, dtype=torch.bfloat16)
#     accelerator.print(f"Load model finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")

#     ema_model_state_dict = lvlm_model.state_dict()
#     with open(ema_deepspeed_config_file, 'r') as f:
#         ds_config = json.load(f)
#     ema_model = create_ema_model(
#         accelerator, model_cls=model_cls, model_config=lvlm_model.config, 
#         ema_model_state_dict=ema_model_state_dict, ds_config=ds_config, 
#         lora_config=lora_config_for_vlm, 
#         )
#     accelerator.print(f"Load ema model finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")
    
#     ema_model.save_pretrained('ema_model', pretrained_lvlm_name_or_path, lora_config_for_vlm)
#     print(ema_model)


#     '''

#     CUDA_VISIBLE_DEVICES=0 python univa/utils/create_ema_zero3_lora.py
#     accelerate launch --num_processes 8 --num_machines 1 univa/utils/create_ema_zero3_lora.py
#     '''