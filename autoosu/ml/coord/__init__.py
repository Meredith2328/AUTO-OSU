"""Coordinate diffusion model (DiT) and sampler, vendored from osu-diffusion (MIT, OliBomby).

The network and the Gaussian diffusion code are unchanged; AUTO-OSU trains it from scratch on the
full noise schedule and drives it from `autoosu.ml.coord_infer`.
"""
from .diffusion import create_diffusion
from .models import DiT, DiT_models
from .positional_embedding import timestep_embedding

__all__ = ["create_diffusion", "DiT", "DiT_models", "timestep_embedding"]
