__all__ = ["GFSAM", "ModalGFSAM"]


def __getattr__(name):
    if name == "GFSAM":
        from gfsam_api.GFSAM import GFSAM

        return GFSAM
    if name == "ModalGFSAM":
        from gfsam_api.ModalGFSAM import ModalGFSAM

        return ModalGFSAM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
