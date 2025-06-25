import sys
sys.path.append("..")
import torch
from einops import rearrange
from PIL import Image
from qwen_vl_utils import process_vision_info
from univa.utils.get_ocr import get_ocr_result
from univa.utils.constant import SPACIAL_TOKEN, PREFILL_TOKEN
from univa.utils.anyres_util import dynamic_resize

def update_size(i1, i2, default_height=1024, default_width=1024):
    shapes = []
    anchor_pixels = default_height * default_width
    for p in (i1, i2):
        if p:
            im = Image.open(p)
            w, h = im.size
            shapes.append((w, h))
    if not shapes:
        return default_height, default_width
    if len(shapes) == 1:
        w, h = shapes[0]
    else:
        w = sum(s[0] for s in shapes) / len(shapes)
        h = sum(s[1] for s in shapes) / len(shapes)
    new_h, new_w = dynamic_resize(int(h), int(w), 'any_11ratio', anchor_pixels=anchor_pixels)
    return new_h, new_w

def prepare_step(
        convo, 
        image1, 
        image2, 
        text,
        think_mode, 
        ocr_enhancer, 
        device, 
        processor, 
        min_pixels=448*448, 
        max_pixels=448*448, 
        think_token='<think>', 
        no_think_token='<no_think>', 
        ):
    content = []
    if text:
        ocr_text = ''
        if ocr_enhancer and content:
            ocr_texts = []
            for img in (image1, image2):
                if img:
                    ocr_texts.append(get_ocr_result(img, cur_ocr_i))
                    cur_ocr_i += 1
            ocr_text = '\n'.join(ocr_texts)
        blanket_text = f"{text} {ocr_text} {think_token if think_mode else no_think_token}"
        remove_blanket_text = ' '.join(blanket_text.split())
        content.append({'type':'text','text': remove_blanket_text})
    for img in (image1, image2):
        if img:
            content.append({'type':'image','image':img,'min_pixels':min_pixels,'max_pixels':max_pixels})

    if len(content) > 0:
        convo.append({'role':'user','content':content})
    # print(convo)
    # Prepare inputs
    chat_text = processor.apply_chat_template(
        convo,
        tokenize=False, 
        add_generation_prompt=True
        )
    image_inputs, video_inputs = process_vision_info(convo)
    inputs = processor(
        text=[chat_text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors='pt'
    ).to(device)
    return inputs, convo


def think(model, tokenizer, processor, inputs, image_begin_token='<|vision_start|>'):
    image_begin_token_id = tokenizer.convert_tokens_to_ids(image_begin_token)

    eos_ids = [
        tokenizer.eos_token_id, 
        image_begin_token_id
        ]
    generated_ids = model.generate(**inputs, max_new_tokens=512, eos_token_id=eos_ids, early_stopping=True)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]
    # print('output_text', output_text)
    return output_text



def clean_spacial_tokens(string):
    for i in SPACIAL_TOKEN['qwen2p5vl_v1_1'].values():
        string = string.replace(i, '')
    string = string.replace(PREFILL_TOKEN, '')
    return ' '.join(string.split())


def siglip_model_1024_encode(siglip_model, siglip_processor, image_paths):
    siglip_pixel_values = torch.concat([
        siglip_processor.preprocess(
            images=Image.open(i).convert('RGB'), 
            do_resize=True, return_tensors="pt", do_convert_rgb=True
            ).pixel_values  for i in image_paths
            ])
    batch_size, channels, height, width = siglip_pixel_values.shape
    siglip_pixel_values = siglip_pixel_values.to(
        device="cuda", dtype=siglip_model.dtype, non_blocking=True
        )
    # len(siglip_pixel_values) == 0 means t2i data
    # B is data parallel number, b is image number in a sequence.
    # siglip_pixel_values Bb c h w, flatten in collator
    siglip_pixel_values = rearrange(
        siglip_pixel_values, "b c (h2 h) (w2 w) -> (b h2 w2) c h w", h2=2, w2=2
    )
    with torch.no_grad():
        siglip_hidden_states = siglip_model(siglip_pixel_values).last_hidden_state
    siglip_hidden_states = rearrange(
        siglip_hidden_states,
        "(b h2 w2) (h w) c -> b (h2 h) (w2 w) c",
        h2=2,
        w2=2,
        h=(height // siglip_model.config.patch_size) // 2,
        w=(width // siglip_model.config.patch_size) // 2,
    )
    return siglip_hidden_states