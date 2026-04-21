__all__ = [
    "FATESAM2D",
    "FATESAM2DAutoPoint",
    "ModalFATESAM2D",
    "ModalFATESAM2DAutoPoint",
]


def __getattr__(name):
    if name == "FATESAM2D":
        from fatesam2d_api.FATESAM2D import FATESAM2D

        return FATESAM2D
    if name == "FATESAM2DAutoPoint":
        from fatesam2d_api.FATESAM2D import FATESAM2DAutoPoint

        return FATESAM2DAutoPoint
    if name == "ModalFATESAM2D":
        from fatesam2d_api.ModalFATESAM2D import ModalFATESAM2D

        return ModalFATESAM2D
    if name == "ModalFATESAM2DAutoPoint":
        from fatesam2d_api.ModalFATESAM2D import ModalFATESAM2DAutoPoint

        return ModalFATESAM2DAutoPoint
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
