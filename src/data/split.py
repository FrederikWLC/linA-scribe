from pathlib import Path
import random

import cv2

DIFFICULTIES = ("easy", "medium", "hard")
DATA_ROOT = Path("data")
get_raw_root = lambda data_root: data_root / "raw"
get_ground_truth_root = lambda data_root: data_root / "ground_truth" / "registered"


def get_difficulty_paths(difficulty: str, data_root: Path = DATA_ROOT) -> list[Path]:
	return sorted((get_raw_root(data_root) / difficulty).glob("*.jpg"))


def iter_difficulty_image_paths(data_root: Path = DATA_ROOT):
	for difficulty in DIFFICULTIES:
		yield difficulty, get_difficulty_paths(difficulty, data_root=data_root)


def get_evaluation_paths(seed: int = 42, data_root: Path = DATA_ROOT):
	"""Get test paths organized by difficulty without loading images."""
	_, test_paths = get_train_test_paths(seed=seed, data_root=data_root)
	evaluation_paths = {difficulty: [] for difficulty in DIFFICULTIES}
	
	for difficulty in DIFFICULTIES:
		difficulty_paths = [path for path in test_paths if path.parent.name == difficulty]
		evaluation_paths[difficulty] = difficulty_paths
	
	return evaluation_paths


def load_difficulty_set(difficulty: str, data_root: Path = DATA_ROOT):
	image_paths = get_difficulty_paths(difficulty, data_root=data_root)
	images, ground_truths, labels = load_paths(image_paths, data_root=data_root)
	return images, ground_truths, labels


def get_test_data_by_difficulty(seed: int = 42, data_root: Path = DATA_ROOT):
	_, test_paths = get_train_test_paths(seed=seed, data_root=data_root)
	evaluation_data = {difficulty: {"images": [], "ground_truths": [], "labels": [], "paths": []} for difficulty in DIFFICULTIES}

	for difficulty in DIFFICULTIES:
		difficulty_paths = [path for path in test_paths if path.parent.name == difficulty]
		images, ground_truths, labels = load_paths(difficulty_paths, data_root=data_root)
		evaluation_data[difficulty] = {
			"images": images,
			"ground_truths": ground_truths,
			"labels": labels,
			"paths": difficulty_paths,
		}

	return evaluation_data


def load_paths(image_paths: list[Path], data_root: Path = DATA_ROOT):
	images = [cv2.imread(path.as_posix(), cv2.IMREAD_GRAYSCALE) for path in image_paths]
	ground_truths = [
		cv2.imread((get_ground_truth_root(data_root) / path.name).as_posix(), cv2.IMREAD_GRAYSCALE)
		for path in image_paths
	]
	labels = [path.stem for path in image_paths]
	return images, ground_truths, labels


def get_train_test_paths(seed: int = 42, data_root: Path = DATA_ROOT):
	rng = random.Random(seed)

	train_paths = []
	test_paths = []
	train_present = set()
	test_present = set()

	for difficulty in DIFFICULTIES:
		paths = get_difficulty_paths(difficulty, data_root=data_root)
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


def get_training_paths(seed: int = 42, data_root: Path = DATA_ROOT):
	train_paths, _ = get_train_test_paths(seed=seed, data_root=data_root)
	label_paths = [get_ground_truth_root(data_root) / path.name for path in train_paths]
	return train_paths, label_paths


def get_training_data(seed: int = 42, data_root: Path = DATA_ROOT):
	train_paths, _ = get_train_test_paths(seed=seed, data_root=data_root)
	images, ground_truths, labels = load_paths(train_paths, data_root=data_root)
	return images, ground_truths, labels


def get_test_data(seed: int = 42, data_root: Path = DATA_ROOT):
	_, test_paths = get_train_test_paths(seed=seed, data_root=data_root)
	images, ground_truths, labels = load_paths(test_paths, data_root=data_root)
	return images, ground_truths, labels


def get_train_test_data(seed: int = 42, data_root: Path = DATA_ROOT):
	train_paths, test_paths = get_train_test_paths(seed=seed, data_root=data_root)
	train_images, train_ground_truths, train_labels = load_paths(train_paths, data_root=data_root)
	test_images, test_ground_truths, test_labels = load_paths(test_paths, data_root=data_root)

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


