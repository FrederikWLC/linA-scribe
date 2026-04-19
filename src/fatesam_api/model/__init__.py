from fatesam_api.model.scribe_sam import ScribeSAM

try:
	from fatesam_api.model.modal_scribe_sam import ModalScribeSAM
except Exception:
	ModalScribeSAM = None
