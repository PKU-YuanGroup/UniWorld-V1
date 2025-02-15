# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# LightningDiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
from .dit import DiT_models
from .tad import TAD_models
Models = {}
Models.update(DiT_models)
Models.update(TAD_models)