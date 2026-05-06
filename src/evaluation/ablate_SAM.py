from sam_api.modal_sam import build_all_modal_sam_variants
from data.split import get_test_data_by_difficulty
from evaluation.evaluate_them import run_full_evaluation

def run_ablation():
    SAM_MODELS = build_all_modal_sam_variants()
    print("Starting ablation study of SAM models...")
    evaluation_data = get_test_data_by_difficulty()
    run_full_evaluation(evaluation_data=evaluation_data, models=SAM_MODELS, csv_path="data/results/ablation_sam")
