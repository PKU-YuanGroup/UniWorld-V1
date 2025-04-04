import os
from .sd3_decoder import SD3DenoiseTower
from .flux_decoder import FluxDenoiseTower

def build_denoise_tower(denoise_tower_cfg, **kwargs):
    denoise_tower = getattr(denoise_tower_cfg, 'mm_denoise_tower', getattr(denoise_tower_cfg, 'denoise_tower', None))
    if 'stable-diffusion-3' in denoise_tower.lower():
        return SD3DenoiseTower(denoise_tower, args=denoise_tower_cfg, **kwargs)
    elif 'flux' in denoise_tower.lower():
        return FluxDenoiseTower(denoise_tower, args=denoise_tower_cfg, **kwargs)
    
    raise ValueError(f'Unknown denoise tower: {denoise_tower}')
