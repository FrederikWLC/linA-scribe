from contextlib import contextmanager
from pathlib import Path
import random
import shutil
import tempfile

import cv2

DIFFICULTIES = ("easy", "medium", "hard")
RAW_ROOT = Path("data/raw")
GROUND_TRUTH_ROOT = Path("data/ground_truth/registered")


def get_difficulty_image_paths(difficulty: str) -> list[Path]:
	return sorted((RAW_ROOT / difficulty).glob("*.jpg"))


def iter_difficulty_image_paths():
	for difficulty in DIFFICULTIES:
		yield difficulty, get_difficulty_image_paths(difficulty)


def load_difficulty_set(difficulty: str):
	image_paths = get_difficulty_image_paths(difficulty)
	return load_paths(image_paths)


def get_evaluation_data(seed: int = 42):
	_, test_paths = get_train_test_paths(seed=seed)
	evaluation_data = {difficulty: {"images": [], "ground_truths": [], "labels": [], "paths": []} for difficulty in DIFFICULTIES}

	for difficulty in DIFFICULTIES:
		difficulty_paths = [path for path in test_paths if path.parent.name == difficulty]
		images, ground_truths, labels = load_paths(difficulty_paths)
		evaluation_data[difficulty] = {
			"images": images,
			"ground_truths": ground_truths,
			"labels": labels,
			"paths": difficulty_paths,
		}

	return evaluation_data


def load_paths(image_paths: list[Path]):
	images = [cv2.imread(path.as_posix(), cv2.IMREAD_GRAYSCALE) for path in image_paths]
	ground_truths = [
		cv2.imread((GROUND_TRUTH_ROOT / path.name).as_posix(), cv2.IMREAD_GRAYSCALE)
		for path in image_paths
	]
	labels = [path.stem for path in image_paths]
	return images, ground_truths, labels


def get_train_test_paths(seed: int = 42):
	rng = random.Random(seed)

	train_paths = []
	test_paths = []
	train_present = set()
	test_present = set()

	for difficulty in DIFFICULTIES:
		paths = get_difficulty_image_paths(difficulty)
		if not paths:
			continue

		n_train = max(1, len(paths) // 2)
		train_subset = sorted(rng.sample(paths, k=n_train))
		train_lookup = {path.name for path in train_subset}
		test_subset = [path for path in paths if path.name not in train_lookup]

		train_paths.extend(train_subset)
		test_paths.extend(test_subset)

		if train_subset:
			train_present.add(difficulty)
		if test_subset:
			test_present.add(difficulty)

	assert train_present == test_present, (
		"Train/test difficulty mismatch. "
		f"Train has {sorted(train_present)}, test has {sorted(test_present)}. "
		"Each represented difficulty must appear in both splits."
	)

	return train_paths, test_paths


def get_training_data(seed: int = 42):
	train_paths, _ = get_train_test_paths(seed=seed)
	images, ground_truths, labels = load_paths(train_paths)
	return images, ground_truths, labels


def get_test_data(seed: int = 42):
	_, test_paths = get_train_test_paths(seed=seed)
	images, ground_truths, labels = load_paths(test_paths)
	return images, ground_truths, labels


def get_train_test_data(seed: int = 42):
	train_paths, test_paths = get_train_test_paths(seed=seed)
	train_images, train_ground_truths, train_labels = load_paths(train_paths)
	test_images, test_ground_truths, test_labels = load_paths(test_paths)

	return {
		"train": {
			"images": train_images,
			"ground_truths": train_ground_truths,
			"labels": train_labels,
			"paths": train_paths,
		},
		"test": {
			"images": test_images,
			"ground_truths": test_ground_truths,
			"labels": test_labels,
			"paths": test_paths,
		},
	}


def materialize_split_folders(seed: int = 42, output_root: Path | None = None):
	split_paths = get_train_test_data(seed=seed)
	base_dir = Path(output_root) if output_root is not None else Path(tempfile.mkdtemp(prefix="split_data_"))

	for subset_name in ("train", "test"):
		image_dir = base_dir / subset_name / "images"
		label_dir = base_dir / subset_name / "labels"
		image_dir.mkdir(parents=True, exist_ok=True)
		label_dir.mkdir(parents=True, exist_ok=True)

		for image_path in split_paths[subset_name]["paths"]:
			shutil.copy2(image_path, image_dir / image_path.name)
			shutil.copy2(GROUND_TRUTH_ROOT / image_path.name, label_dir / image_path.name)

	return {
		"root": base_dir,
		"train_images": base_dir / "train" / "images",
		"train_labels": base_dir / "train" / "labels",
		"test_images": base_dir / "test" / "images",
		"test_labels": base_dir / "test" / "labels",
	}


@contextmanager
def temporary_split_folders(seed: int = 42):
	split_dirs = materialize_split_folders(seed=seed)
	try:
		yield split_dirs
	finally:
		shutil.rmtree(split_dirs["root"], ignore_errors=True)
