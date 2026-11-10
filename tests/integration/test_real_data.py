"""Checks against the real small dataset, unzipped from the zip committed to the repo.
Skipped only if that zip is missing."""

import zipfile

import pytest

from patent_search.config import DATA_DIRS, SMALL_DATA_ZIP
from patent_search.load import load_patents
from patent_search.setup_data import ensure_small_data

pytestmark = [
    pytest.mark.realdata,
    pytest.mark.skipif(not SMALL_DATA_ZIP.is_file(), reason="data/patent_data_small.zip not found"),
]


@pytest.fixture(scope="module")
def small():
    ensure_small_data()
    return load_patents(DATA_DIRS["small"])


def test_committed_zip_has_the_64_weekly_files():
    with zipfile.ZipFile(SMALL_DATA_ZIP) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
    assert len(names) == 64 and all("patents_ipa" in n for n in names)


def test_small_dataset_report_matches_what_we_measured(small):
    _, report = small
    assert report.files_read == 64 and not report.bad_files
    assert report.kept == 640 and report.duplicates_dropped == 0 and report.bad_records == 0
    assert report.no_description == 119
    assert report.no_cpc == 0


def test_small_dataset_fields_are_clean(small):
    patents, _ = small
    assert all(p.pub_date is not None and p.cpc is not None for p in patents)
    assert all(p.cpc.subclass.startswith("B60") for p in patents)
    assert all(c.text and "(canceled)" not in c.text.lower() for p in patents for c in p.claims)
    assert all(para.strip() for p in patents for para in p.paragraphs)
