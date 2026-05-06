from sam_api.modal_sam import build_best_modal_sam_variant
from gfsam_api.ModalGFSAM import build_modal_gfsam
from fatesam2d_api.ModalFATESAM2D import build_default_modal_fatesam2d, build_all_modal_fatesam2d_variants
from data.split import get_test_data_by_difficulty
from evaluation.evaluate_them import run_full_evaluation

def evaluate_sam_variants():
    SAM_MODELS = [
        build_best_modal_sam_variant(),
        build_modal_gfsam(),
    ] + build_all_modal_fatesam2d_variants()
    print("Starting evaluation of SAM models...")
    evaluation_data = get_test_data_by_difficulty()
    run_full_evaluation(evaluation_data=evaluation_data, models=SAM_MODELS, csv_path="data/results/evaluation_sam")