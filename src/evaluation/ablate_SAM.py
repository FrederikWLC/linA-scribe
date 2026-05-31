from sam_api.modal_sam import build_all_modal_sam_variants
from data.split import get_test_data_by_difficulty
from evaluation.evaluate_them import run_full_evaluation

def run_ablation():
    SAM_MODELS = build_all_modal_sam_variants()
    print("Starting ablation study of SAM models...")
    # we do binarize the images ideally for the evaluation
    # (unlike tuning, as it cannot be redone since ideal binarization was first implemented after the interactive experiment with Ester)
    evaluation_data = get_test_data_by_difficulty(binarized=True)
    run_full_evaluation(evaluation_data=evaluation_data, models=SAM_MODELS, csv_path="data/results/ablation_sam",short_names=True)
