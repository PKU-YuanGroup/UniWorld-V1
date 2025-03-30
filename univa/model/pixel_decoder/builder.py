from .flux_decoder import FluxDecoder


def build_pixel_decoder(config, **kwargs):
    return FluxDecoder(config, **kwargs)
