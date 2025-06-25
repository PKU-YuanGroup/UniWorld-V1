
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
        'think_end_token': '<think_end>', 
        'think_token': '<think>', 
        'no_think_token': '<no_think>', 
    }, 
}
PREFILL_TOKEN = '<image>'
GENERATE_TOKEN = '<gen_image>'
SYSTEM_PROMPT = """
You are UniWorld, a friendly multimodal assistant from Peking University. 
You accept multimodal inputs—text, images, and video—and generate outputs across modalities, 
including text-to-image synthesis, image editing, and reference-based image generation. 
You genuinely understand user directives and contemplate their deeper intent.
"""