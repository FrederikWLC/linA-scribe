import fnmatch
import os

from app.utils.supabase_storage import get_storage

SUPABASE_EVAL_PREFIX = os.getenv("SUPABASE_EVAL_PREFIX", "evaluation-exports").strip("/")
SIGNED_URL_TTL = int(os.getenv("SUPABASE_SIGNED_URL_TTL", "3600"))


def _signed_file_entry(storage, path: str) -> dict[str, str] | None:
    try:
        signed = storage.create_signed_url(path, SIGNED_URL_TTL)
    except Exception:
        return None

    signed_url = signed.get("signedURL") or signed.get("signedUrl")
    if not signed_url:
        return None
    return {"path": path, "download_url": signed_url}


def _list_paths_under_prefix(storage, prefix: str) -> list[str]:
    found: set[str] = set()

    def visit(current: str) -> None:
        try:
            items = storage.list(current)
        except Exception:
            return

        for item in items or []:
            name = item.get("name")
            if not name:
                continue

            candidate = f"{current.rstrip('/')}/{name}" if current else name
            lower_name = name.lower()
            if lower_name.endswith((".csv", ".jpg", ".jpeg", ".png")):
                found.add(candidate)
            else:
                visit(candidate)

    visit(prefix)
    return sorted(found)


def _filter_and_sign(patterns: tuple[str, ...], path_predicate=None) -> list[dict[str, str]]:
    storage = get_storage()
    base_paths = _list_paths_under_prefix(storage, SUPABASE_EVAL_PREFIX)
    rows: list[dict[str, str]] = []

    for path in base_paths:
        filename = path.rsplit("/", 1)[-1]
        if not any(fnmatch.fnmatch(filename, pattern) for pattern in patterns):
            if not (path_predicate and path_predicate(path)):
                continue

        entry = _signed_file_entry(storage, path)
        if entry:
            rows.append(entry)

    return rows


def get_tuning_files() -> list[dict[str, str]]:
    return _filter_and_sign(("tuning-*.csv",))


def get_evaluation_files() -> list[dict[str, str]]:
    return _filter_and_sign(("evaluation-*.csv",))


def get_ablation_files() -> list[dict[str, str]]:
    return _filter_and_sign(("ablation_sam-*.csv",))


def get_scribing_files() -> list[dict[str, str]]:
    return _filter_and_sign(
        ("*.jpg", "*.jpeg", "*.png"),
        path_predicate=lambda path: "/scribing/" in f"/{path.lower()}",
    )
