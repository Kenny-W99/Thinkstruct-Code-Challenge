from datetime import date

import pytest

from patent_search.load import (
    BadRecord,
    dedupe,
    find_data_files,
    load_patents,
    parse_pub_date,
    parse_record,
    patent_from_dict,
    patent_to_dict,
)
from tests.conftest import LARGE_FILES, SMALL_FILES


def test_pub_date_from_small_filename_and_year_from_large_doc_number():
    assert parse_pub_date("US20240051333A1-20240215.XML", "20240051333") == (date(2024, 2, 15), 2024)
    assert parse_pub_date("US20150158422A1.XML", "20150158422") == (None, 2015)
    assert parse_pub_date("", "ABC") == (None, None)


def test_parse_small_record():
    raw = SMALL_FILES["patents_ipa240215.json"][0]
    p = parse_record(raw, "patents_ipa240215.json")
    assert p.doc_number == "20240000001" and p.title == "PNEUMATIC TIRE"
    assert p.cpc.subclass == "B60C" and p.pub_date == date(2024, 2, 15)
    assert len(p.paragraphs) == 2 and "in FIG.1 the" in p.paragraphs[1]  # blanks dropped, text cleaned
    assert [c.number for c in p.claims] == [1, 2, 5]  # canceled 3-4 dropped
    assert p.has_description


def test_parse_large_record_turns_na_into_none():
    raw = LARGE_FILES["job_000/patents_batch_0001.json"][0]
    p = parse_record(raw)
    assert p.title is None and p.cpc is None and p.year == 2015 and p.pub_date is None


def test_keep_description_false_still_knows_there_was_one():
    raw = SMALL_FILES["patents_ipa240215.json"][0]
    p = parse_record(raw, keep_description=False)
    assert p.paragraphs == [] and p.has_description


def test_keep_claims_false_skips_claims_but_record_still_counts_as_text():
    raw = SMALL_FILES["patents_ipa240215.json"][0]
    assert parse_record(raw, keep_claims=False).claims == []
    claims_only = {"doc_number": "7", "claims": ["1. A thing."]}
    assert parse_record(claims_only, keep_claims=False, keep_description=False).doc_number == "7"


@pytest.mark.parametrize("raw", ["a string", {"title": "no number"}, {"doc_number": "1", "title": "N/A"}])
def test_bad_records_raise(raw):
    with pytest.raises(BadRecord):
        parse_record(raw)


def test_dedupe_keeps_longest_copy_in_first_seen_order():
    short = parse_record({"doc_number": "1", "abstract": "short"})
    other = parse_record({"doc_number": "2", "abstract": "other"})
    long = parse_record({"doc_number": "1", "abstract": "a much longer abstract"})
    kept, dropped = dedupe([short, other, long])
    assert [p.doc_number for p in kept] == ["1", "2"] and kept[0].abstract == "a much longer abstract"
    assert dropped == 1


def test_find_data_files_skips_sample_json(large_dir):
    names = [str(p.relative_to(large_dir)) for p in find_data_files(large_dir)]
    assert "sample.json" not in names and len(names) == 3


def test_missing_folder_has_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="README"):
        find_data_files(tmp_path / "nope")


def test_load_report_counts_every_problem(large_dir):
    patents, report = load_patents(large_dir)
    assert report.files_read == 2 and len(report.bad_files) == 1  # broken JSON file skipped, not fatal
    assert report.records_read == 6 and report.bad_records == 2
    assert report.duplicates_dropped == 1 and report.kept == 3
    assert report.no_cpc == 1 and report.no_title == 1
    charger = next(p for p in patents if p.doc_number == "20180000002")
    assert "induction" in charger.abstract  # the longer duplicate won


def test_max_files(large_dir):
    _, report = load_patents(large_dir, max_files=1)
    assert report.files_read == 1


def test_serialization_round_trip(small_dir):
    patents, _ = load_patents(small_dir)
    for p in patents:
        assert patent_from_dict(patent_to_dict(p)) == p
