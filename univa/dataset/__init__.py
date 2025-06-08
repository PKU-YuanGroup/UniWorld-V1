from .llava_dataset import LlavaDataset
from .qwen2vl_dataset import Qwen2VLDataset
from .redux_dataset import ReduxDataset

DATASET_TYPE = {
    'llava': LlavaDataset, 
    'qwen2vl': Qwen2VLDataset, 
    'qwen2p5vl': Qwen2VLDataset, 
}