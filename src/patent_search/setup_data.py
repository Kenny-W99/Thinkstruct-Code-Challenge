"""Unzip the small dataset that ships with the repo (data/patent_data_small.zip).

Safe by design:
- only the *file name* of each member is used, so a hostile path like
  "../../x.json" can never write outside the target folder ("zip slip");
- files are unzipped into a temporary folder first and renamed at the end,
  so an interrupted unzip never leaves a half-filled data folder behind.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path, PurePosixPath

from .config import DATA_DIRS, SMALL_DATA_ZIP


def extract_json_files(zip_path: Path, dest_dir: Path) -> int:
    """Unzip every *.json file into dest_dir (flat). Returns how many were written.
    Raises ValueError if two members share a file name (they would overwrite)."""
    tmp = dest_dir.with_name(dest_dir.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    seen: set[str] = set()
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                name = PurePosixPath(info.filename).name
                if info.is_dir() or not name.endswith(".json") or name.startswith("._") or "__MACOSX" in info.filename:
                    continue
                if name in seen:
                    raise ValueError(f"two files named {name!r} in {zip_path.name}")
                seen.add(name)
                with zf.open(info) as src, open(tmp / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        shutil.rmtree(dest_dir, ignore_errors=True)
        tmp.rename(dest_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return len(seen)


def ensure_small_data(zip_path: Path = SMALL_DATA_ZIP, data_dir: Path = DATA_DIRS["small"]) -> tuple[Path, int]:
    """Unzip the small data unless data_dir already has JSON files.
    Returns (data_dir, number of files unzipped; 0 = it was already there)."""
    if data_dir.is_dir() and any(data_dir.glob("*.json")):
        return data_dir, 0
    if not zip_path.is_file():
        raise FileNotFoundError(f"Small data zip not found: {zip_path} (see README: 'Get the data')")
    return data_dir, extract_json_files(zip_path, data_dir)
