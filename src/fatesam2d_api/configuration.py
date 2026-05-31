import numpy as np

from config import config
from data.split import get_support_data
from scribe.baselines.sam import AUTOPOINT_SPECS
from scribe.tunable import TunableConfiguration


class FATESAM2DConfiguration(TunableConfiguration):
    def __init__(self, support_data_root=config.DATA_DIR, top_n_supports=3, use_autopoints=False, is_blank=False):
        self.support_data_root = support_data_root
        self.top_n_supports = top_n_supports
        self.use_autopoints = use_autopoints
        self.is_blank = is_blank

        self.checkpoint_path = config.FATESAM_CHECKPOINT_PATH
        self.config_file = config.FATESAM_CONFIG

        name = "FATESAM2D+pts" if use_autopoints else "FATESAM2D-blank" if is_blank else "FATESAM2D"
        short_name = "FATESAM2D+pts" if use_autopoints else "FATESAM2D-blank" if is_blank else "FATESAM2D"
        hyperparameter_specs = AUTOPOINT_SPECS if use_autopoints else []

        super().__init__(
            name=name,
            short_name=short_name,
            hyperparameter_specs=hyperparameter_specs,
        )

    def get_support_data(self):
        # we do binarize the images ideally for the few-shot inference and evaluation
        # (unlike tuning, as it cannot be redone since ideal binarization was first implemented after the interactive experiment with Ester)
        support_images, support_labels, _ = get_support_data(data_root=self.support_data_root, binarized=True)
        if self.is_blank:
            rng = np.random.default_rng(42)
            support_labels = [
                np.where(
                    rng.random(label.shape) < 0.05,
                    0,
                    255,
                ).astype(np.uint8)
                for label in support_labels
            ]
        return support_images, support_labels


def get_default_fatesam2d_configuration(support_data_root: str = config.DATA_DIR) -> FATESAM2DConfiguration:
    return FATESAM2DConfiguration(
        support_data_root=support_data_root,
        top_n_supports=3,
        use_autopoints=False,
        is_blank=False,
    )


def get_all_fatesam2d_configurations(data_root: str = config.DATA_DIR) -> list[FATESAM2DConfiguration]:
    return [
        FATESAM2DConfiguration(support_data_root=data_root, top_n_supports=3, use_autopoints=False, is_blank=False),
        FATESAM2DConfiguration(support_data_root=data_root, top_n_supports=3, use_autopoints=True, is_blank=False),
        FATESAM2DConfiguration(support_data_root=data_root, top_n_supports=3, use_autopoints=False, is_blank=True),
    ]


def get_all_tunable_fatesam2d_configurations(data_root: str = config.DATA_DIR) -> list[FATESAM2DConfiguration]:
    return [conf for conf in get_all_fatesam2d_configurations(data_root=data_root) if conf.is_tunable()]
