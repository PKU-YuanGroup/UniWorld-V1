import argparse
import torch

from univa.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from univa.conversation import conv_templates, SeparatorStyle
from univa.model.builder import load_pretrained_model
from univa.utils import disable_torch_init
from univa.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from transformers.cache_utils import DynamicCache

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def main(args):
    # Model
    disable_torch_init()

    # torch_dtype = torch.bfloat16
    torch_dtype = torch.float16
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device, torch_dtype=torch_dtype
        )
    if "qwen2" in model_name.lower():
        conv_mode = "qwen_chatml"
    else:
        conv_mode = "v1"

    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    image = load_image(args.image_file)
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, model.config)
    image_h, image_w = image_tensor.shape[-2:]
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch_dtype) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch_dtype)

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if getattr(model.config, 'mm_use_im_start_end', False):
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            image = None
        
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        # print('prompt', prompt)
        print('input_ids', input_ids.shape, input_ids)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=False)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[[image_h, image_w]],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True)
        print('output_ids', output_ids)
        outputs = tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    # args.model_path = "/mnt/data/lb/logs/univa/univa-siglip-qwen2p5-3p0b-pt558k-sft737k-mmtag-0403-stage3-ft-debug-debug"
    args.model_path = "/mnt/data/lb/logs/univa/univa-siglip-qwen2p5-3p0b-pt558k-sft737k-mmtag-0403-stage3-ft-imend"
    args.image_file = "/mnt/data/datasets/LLaVA/llava_image_tune/coco/train2017/000000033471.jpg"
    main(args)

    '''
    CUDA_VISIBLE_DEVICES=0 python cli.py
    '''