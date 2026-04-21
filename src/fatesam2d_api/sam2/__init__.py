"""Vendored SAM2 implementation used by fatesam2d_api.FATESAM2D."""

import sys

# Keep upstream import paths working for Hydra targets like sam2.*
sys.modules.setdefault("sam2", sys.modules[__name__])
