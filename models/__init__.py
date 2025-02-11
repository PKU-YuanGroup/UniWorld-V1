# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# LightningDiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
from .did import DiD_models
from .dit import DiT_models
Models = {}
Models.update(DiD_models)
Models.update(DiT_models)