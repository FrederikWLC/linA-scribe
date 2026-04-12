"""Vendored SAM2 implementation used by model.scribe_sam."""

import sys

# Keep upstream import paths working for Hydra targets like sam2.*
sys.modules.setdefault("sam2", sys.modules[__name__])
