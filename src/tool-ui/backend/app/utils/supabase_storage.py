import os

from fastapi import HTTPException
from supabase import create_client


# Build and return Supabase storage client for configured bucket.
def get_storage():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    bucket = os.getenv("SUPABASE_BUCKET")
    if not url or not key or not bucket:
        raise HTTPException(
            status_code=500,
            detail="Missing SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, or SUPABASE_BUCKET",
        )

    client = create_client(url, key)
    return client.storage.from_(bucket)


# Collect CSV paths under a storage prefix with one nested folder level.
def csv_files_under_prefix(storage, prefix: str) -> list[str]:
    csv_paths: list[str] = []
    root_items = storage.list(prefix)

    for item in root_items or []:
        name = item.get("name")
        if not name:
            continue

        candidate = f"{prefix.rstrip('/')}/{name}"
        if name.lower().endswith(".csv"):
            csv_paths.append(candidate)
            continue

        nested_items = storage.list(candidate)
        for nested in nested_items or []:
            nested_name = nested.get("name")
            if not nested_name:
                continue
            if nested_name.lower().endswith(".csv"):
                csv_paths.append(f"{candidate.rstrip('/')}/{nested_name}")

    return sorted(set(csv_paths))
