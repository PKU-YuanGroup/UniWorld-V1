from .marvae import MAR_VAE
from .vavae import VA_VAE
from .sdvae import SD_VAE
from .flowautoencoder import FlowVAE_models
from .mmflowautoencoder import MMFlowVAE_models

VAE_Models = {
    'marvae': MAR_VAE, 
    'vavae': VA_VAE, 
    'sdvae': SD_VAE, 
}
VAE_Models.update(FlowVAE_models)
VAE_Models.update(MMFlowVAE_models)