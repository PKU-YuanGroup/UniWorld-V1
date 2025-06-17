
from .qwen2p5vl_dataset import Qwen2p5VLDataset
from .redux_dataset import ReduxDataset
from .qwen2p5vl_dataset_v1_1 import Qwen2p5VLDataset_V1_1

DATASET_TYPE = {
    'qwen2p5vl': Qwen2p5VLDataset, 
    'qwen2p5vl_v1_1': Qwen2p5VLDataset_V1_1, 
    'redux': ReduxDataset, 
}