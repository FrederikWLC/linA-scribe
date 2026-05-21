from pathlib import Path
import random
import cv2

DIFFICULTIES = ("easy", "medium", "hard")
DATA_ROOT = Path("data")
get_raw_root = lambda data_root: data_root / "raw"
get_ground_truth_root = lambda data_root: data_root / "ground_truth" / "registered0"
get_binarized_ground_truth_root = lambda data_root: data_root / "ground_truth" / "registered2"

def get_difficulty_paths(difficulty: str, data_root: Path | str = DATA_ROOT, ) -> list[Path]:
	data_root = Path(data_root)
	return sorted((get_raw_root(data_root) / difficulty).glob("*.jpg"))

def iter_difficulty_image_paths(data_root: Path | str = DATA_ROOT):
	data_root = Path(data_root)
	for difficulty in DIFFICULTIES:
		yield difficulty, get_difficulty_paths(difficulty, data_root=data_root)

def load_paths(image_paths: list[Path], data_root: Path | str = DATA_ROOT, binarized=False):
	data_root = Path(data_root)
	images = [cv2.imread(path.as_posix(), cv2.IMREAD_GRAYSCALE) for path in image_paths]
	ground_truth_root = get_binarized_ground_truth_root(data_root) if binarized else get_ground_truth_root(data_root)
	ground_truths = [
		cv2.imread((ground_truth_root / path.stem).with_suffix(".png").as_posix(), cv2.IMREAD_GRAYSCALE)
		if binarized else cv2.imread((ground_truth_root / path.name).as_posix(), cv2.IMREAD_GRAYSCALE)
		for path in image_paths
	]
	labels = [path.stem for path in image_paths]
	return images, ground_truths, labels

# ====================================|
# Sampling of support/val/test splits | (for generalizability)
# ====================================|

# 60 datapoints in total
# 10% (6) to support
# 45% (27) to val
# 45% (27) to test

# with equal difficulty sample sizes all the way through
# n(easy) = n(medium) = n(hard)

def get_support_val_test_paths(seed: int = 42, data_root: Path | str = DATA_ROOT):
	data_root = Path(data_root)
	rng = random.Random(seed)

	support_paths = []
	val_paths = []
	test_paths = []

	support_present = set()
	val_present = set()
	test_present = set()

	for difficulty in DIFFICULTIES:
		paths = get_difficulty_paths(difficulty, data_root=data_root)
		if not paths:
			continue
		
		n = len(paths)
		n_support = max(1, n // 10)
		n_val_test = n - n_support
		n_val = n_test = max(1, n_val_test // 2)
		
		support_subset = sorted(rng.sample(paths, k=n_support))
		val_test_subset = [p for p in paths if p not in support_subset]

		val_subset = rng.sample(val_test_subset, k=n_val)
		test_subset = [path for path in val_test_subset if path not in val_subset]

		support_paths.extend(support_subset)
		val_paths.extend(val_subset)
		test_paths.extend(test_subset)

		if support_subset:
			support_present.add(difficulty)
		if val_subset:
			val_present.add(difficulty)
		if test_subset:
			test_present.add(difficulty)

	assert val_present == test_present, (
		"Val/test difficulty mismatch. "
		f"Val has {sorted(val_present)}, test has {sorted(test_present)}. "
		"Each represented difficulty must appear in both splits."
	)

	return support_paths, val_paths, test_paths

def get_support_data(seed: int = 42, data_root: Path | str = DATA_ROOT, binarized=False):
	data_root = Path(data_root)
	support_paths, _, _ = get_support_val_test_paths(seed=seed, data_root=data_root)
	images, ground_truths, labels = load_paths(support_paths, data_root=data_root, binarized=binarized)
	return images, ground_truths, labels

def get_val_data(seed: int = 42, data_root: Path | str = DATA_ROOT, binarized=False):
	data_root = Path(data_root)
	_, val_paths, _ = get_support_val_test_paths(seed=seed, data_root=data_root)
	images, ground_truths, labels = load_paths(val_paths, data_root=data_root, binarized=binarized)
	return images, ground_truths, labels

def get_test_data(seed: int = 42, data_root: Path | str = DATA_ROOT, binarized=False):
	data_root = Path(data_root)
	_, _, test_paths = get_support_val_test_paths(seed=seed, data_root=data_root)
	images, ground_truths, labels = load_paths(test_paths, data_root=data_root, binarized=binarized)
	return images, ground_truths, labels

def get_test_data_by_difficulty(seed: int = 42, data_root: Path | str = DATA_ROOT, binarized=False):
	data_root = Path(data_root)
	_, _, test_paths = get_support_val_test_paths(seed=seed, data_root=data_root)
	evaluation_data = {difficulty: {"images": [], "ground_truths": [], "labels": [], "paths": []} for difficulty in DIFFICULTIES}

	for difficulty in DIFFICULTIES:
		difficulty_paths = [path for path in test_paths if path.parent.name == difficulty]
		images, ground_truths, labels = load_paths(difficulty_paths, data_root=data_root, binarized=binarized)
		evaluation_data[difficulty] = {
			"images": images,
			"ground_truths": ground_truths,
			"labels": labels,
			"paths": difficulty_paths,
		}

	return evaluation_data