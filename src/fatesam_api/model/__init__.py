try:
	from fatesam_api.model.scribe_sam import FATESAM2D
	ScribeSAM = FATESAM2D
except Exception:
	FATESAM2D = None
	ScribeSAM = None

try:
	from fatesam_api.model.modal_scribe_sam import ModalFATESAM2D
	ModalScribeSAM = ModalFATESAM2D
except Exception:
	ModalFATESAM2D = None
	ModalScribeSAM = None
