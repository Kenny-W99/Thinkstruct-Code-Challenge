import zipfile

import pytest

from patent_search.setup_data import ensure_small_data, extract_json_files


def make_zip(path, members: dict[str, bytes]):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_extracts_json_files_flat_and_skips_junk(tmp_path):
    z = make_zip(
        tmp_path / "d.zip",
        {
            "data/patent_data_small/a.json": b"[]",
            "data/patent_data_small/b.json": b"[1]",
            "data/patent_data_small/readme.txt": b"not data",
            "__MACOSX/data/._a.json": b"mac junk",
            "data/patent_data_small/._c.json": b"mac junk",
        },
    )
    dest = tmp_path / "out"
    assert extract_json_files(z, dest) == 2
    assert sorted(p.name for p in dest.iterdir()) == ["a.json", "b.json"]
    assert (dest / "b.json").read_bytes() == b"[1]"


def test_zip_slip_paths_cannot_escape_the_target_folder(tmp_path):
    z = make_zip(tmp_path / "evil.zip", {"../../evil.json": b"[]"})
    dest = tmp_path / "a" / "b" / "out"
    extract_json_files(z, dest)
    assert (dest / "evil.json").exists()
    assert not (tmp_path / "evil.json").exists() and not (tmp_path / "a" / "evil.json").exists()


def test_duplicate_names_fail_without_leaving_a_half_filled_folder(tmp_path):
    z = make_zip(tmp_path / "dup.zip", {"x/a.json": b"[]", "y/a.json": b"[]"})
    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="two files"):
        extract_json_files(z, dest)
    assert not dest.exists() and not (tmp_path / "out.partial").exists()


def test_ensure_unzips_only_once(tmp_path):
    z = make_zip(tmp_path / "d.zip", {"d/a.json": b"[]"})
    dest = tmp_path / "small"
    assert ensure_small_data(z, dest) == (dest, 1)
    assert ensure_small_data(z, dest) == (dest, 0)


def test_ensure_missing_zip_has_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="README"):
        ensure_small_data(tmp_path / "nope.zip", tmp_path / "small")
