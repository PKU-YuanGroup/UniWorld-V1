from .qwen2p5vl.modeling_univa_qwen2p5vl import UnivaQwen2p5VLForConditionalGeneration
from .qwen2p5vl.modeling_univa_qwen2p5vl_v1_1 import UnivaQwen2p5VLForConditionalGeneration_V1_1
MODEL_TYPE = {
    'qwen2p5vl': UnivaQwen2p5VLForConditionalGeneration, 
    'qwen2p5vl_v1_1': UnivaQwen2p5VLForConditionalGeneration_V1_1, 
}