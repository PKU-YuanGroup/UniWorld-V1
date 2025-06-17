
SPACIAL_TOKEN = {
    'qwen2p5vl': {
        'image_token': '<|image_pad|>', 
        'image_begin_token': '<|vision_start|>', 
        'image_end_token': '<|vision_end|>', 
    }, 
    'qwen2p5vl_v1_1': {
        'image_token': '<|image_pad|>', 
        'image_begin_token': '<|vision_start|>', 
        'image_end_token': '<|vision_end|>', 
        'think_begin_token': '<think_begin>', 
        'think_end_token': '</think_end>', 
        'think_token': '<think>', 
        'no_think_token': '<no_think>', 
    }, 
}
GENERATE_TOKEN = '<gen_image>'