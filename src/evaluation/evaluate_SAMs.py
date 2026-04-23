from scribe.baselines.sam import BestMobileSAMv2Implementation,MobileSAMv2BilateralFilterBestOfThree
from gfsam_api.ModalGFSAM import ModalGFSAM
from fatesam2d_api.ModalFATESAM2D import ModalFATESAM2D, ModalFATESAM2DAutoPoint, ModalFATESAM2DBlank
from data.split import get_test_data_by_difficulty
from evaluation.evaluate_them import run_full_evaluation

SAM_MODELS = [
    BestMobileSAMv2Implementation(),
    MobileSAMv2BilateralFilterBestOfThree(),
    ModalGFSAM(),
    ModalFATESAM2D(),
    ModalFATESAM2DAutoPoint(),
    ModalFATESAM2DBlank()
]


def evaluate_sam_variants():
    print("Starting evaluation of SAM models...")
    evaluation_data = get_test_data_by_difficulty()
    run_full_evaluation(evaluation_data=evaluation_data, models=SAM_MODELS, csv_path="data/results/evaluation_sam")
