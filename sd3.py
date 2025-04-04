import torch
import numpy as np
import random
from diffusers import StableDiffusion3Pipeline
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

pipe = StableDiffusion3Pipeline.from_pretrained("/mnt/data/checkpoints/stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

image = pipe(
    # prompt='A cat holding a sign that says hello world',
    prompt='',
    prompt_2='',
    # prompt_2='A cat holding a sign that says hello world', 
    prompt_3='A cat holding a sign that says hello world', 
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
).images[0]
image.save('sd3_p1_.jpg')
pipe.save_pretrained('test_save_pipeline')
