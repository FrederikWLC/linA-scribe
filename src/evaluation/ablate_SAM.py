from evaluation.baselines.sam import MobileSAMv2AutoPointBilateralFilter, MobileSAMv2AutoPointBilateralFilterBestOfThree, MobileSAMv2AutoPointNoFilter, MobileSAMv2AutoPointNoFilterBestOfThree, MobileSAMv2BilateralFilter, MobileSAMv2BilateralFilterBestOfThree, MobileSAMv2NoFilterBestOfThree, MobileSAMv2NoFilter
from data.split import get_evaluation_data
from evaluation.evaluate_them import run_full_evaluation

SAM_MODELS = [
    MobileSAMv2AutoPointBilateralFilter(),
    MobileSAMv2AutoPointBilateralFilterBestOfThree(),
    MobileSAMv2AutoPointNoFilter(),
    MobileSAMv2AutoPointNoFilterBestOfThree(),
    MobileSAMv2BilateralFilter(),
    MobileSAMv2BilateralFilterBestOfThree(),
    MobileSAMv2NoFilterBestOfThree(),
    MobileSAMv2NoFilter()
]


def run_ablation():
    print("Starting ablation study of SAM models...")
    evaluation_data = get_evaluation_data()
    run_full_evaluation(evaluation_data=evaluation_data, models=SAM_MODELS, csv_path="data/results/ablation_sam")
