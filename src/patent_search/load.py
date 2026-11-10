"""Read patent JSON files (both dataset formats) into clean `Patent` objects.

Split into small pieces so each can be unit tested:
    find_data_files -> read_json_file -> parse_record -> dedupe -> LoadReport
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from .clean import Claim, clean_text, is_missing, parse_claims
from .cpc import CPCCode, parse_cpc

log = logging.getLogger(__name__)


class BadRecord(ValueError):
    """A record we cannot use (not an object, no document number, no text)."""


@dataclass
class Patent:
    doc_number: str
    title: str | None
    abstract: str | None
    claims: list[Claim]
    paragraphs: list[str]  # non-empty description paragraphs
    cpc: CPCCode | None
    pub_date: date | None  # exact publication date when the file name has it (small data)
    year: int | None  # always set when known (large data only has the year)
    filename: str = ""
    source_file: str = ""
    has_description: bool = False  # from the raw record, even if paragraphs were not kept

    @property
    def text_size(self) -> int:
        """Rough amount of text; used to pick the best copy of a duplicate."""
        return (
            len(self.title or "")
            + len(self.abstract or "")
            + sum(len(c.text) for c in self.claims)
            + sum(len(p) for p in self.paragraphs)
        )


# ---------------------------------------------------------------- parsing

_FILENAME_DATE = re.compile(r"-(\d{8})\.XML$", re.IGNORECASE)


def parse_pub_date(filename: str, doc_number: str) -> tuple[date | None, int | None]:
    """Small data: "US20240051333A1-20240215.XML" -> 2024-02-15.
    Large data: "US20150158422A1.XML" has no date, so use the year in the doc number."""
    match = _FILENAME_DATE.search(filename or "")
    if match:
        s = match.group(1)
        try:
            d = date(int(s[:4]), int(s[4:6]), int(s[6:]))
            return d, d.year
        except ValueError:
            pass
    if doc_number[:4].isdigit() and 1976 <= int(doc_number[:4]) <= 2100:
        return None, int(doc_number[:4])
    return None, None


def parse_record(
    raw: object,
    source_file: str = "",
    keep_description: bool = True,
    keep_claims: bool = True,
) -> Patent:
    if not isinstance(raw, dict):
        raise BadRecord("record is not a JSON object")
    if is_missing(raw.get("doc_number")):
        raise BadRecord("missing doc_number")
    doc_number = str(raw["doc_number"]).strip()

    def text_field(key: str) -> str | None:
        value = raw.get(key)
        return None if is_missing(value) or not isinstance(value, str) else clean_text(value) or None

    raw_claims = raw.get("claims") or []
    raw_claims = raw_claims if isinstance(raw_claims, list) else [raw_claims]
    if keep_claims:
        claims = parse_claims(raw_claims)
        has_claims = bool(claims)
    else:  # parsing ~20 claims per patent is the slow part of loading; skip it when not needed
        claims = []
        has_claims = any(isinstance(c, str) and not is_missing(c) for c in raw_claims)

    desc = raw.get("detailed_description") or []
    desc = desc if isinstance(desc, list) else [desc]
    if keep_description:
        paragraphs = [clean_text(p) for p in desc if isinstance(p, str) and not is_missing(p)]
        paragraphs = [p for p in paragraphs if p]
        has_description = bool(paragraphs)
    else:
        # Don't clean ~80 paragraphs per patent just to learn whether there were any.
        paragraphs = []
        has_description = any(isinstance(p, str) and not is_missing(p) for p in desc)

    title, abstract = text_field("title"), text_field("abstract")
    if not (title or abstract or has_claims or has_description):
        raise BadRecord("record has no text")

    filename = str(raw.get("filename") or "")
    pub_date, year = parse_pub_date(filename, doc_number)
    return Patent(
        doc_number=doc_number,
        title=title,
        abstract=abstract,
        claims=claims,
        paragraphs=paragraphs,
        cpc=parse_cpc(raw.get("classification")),
        pub_date=pub_date,
        year=year,
        filename=filename,
        source_file=source_file,
        has_description=has_description,
    )


def dedupe(patents: list[Patent]) -> tuple[list[Patent], int]:
    """Keep one copy per document number (the one with the most text).
    Keeps first-seen order. Returns (kept, number_dropped)."""
    position: dict[str, int] = {}
    kept: list[Patent] = []
    for p in patents:
        i = position.get(p.doc_number)
        if i is None:
            position[p.doc_number] = len(kept)
            kept.append(p)
        elif p.text_size > kept[i].text_size:
            kept[i] = p
    return kept, len(patents) - len(kept)


# ---------------------------------------------------------------- files


def find_data_files(root: Path | str) -> list[Path]:
    """All patent JSON files under root, sorted. Skips the large set's sample.json."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data folder not found: {root} (see README: 'Get the data')")
    return sorted(p for p in root.rglob("*.json") if p.name != "sample.json")


def read_json_file(path: Path) -> list:
    """Raises OSError / ValueError for unreadable or invalid files."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("top-level JSON value is not a list")
    return data


@dataclass
class LoadReport:
    files_read: int = 0
    bad_files: list[str] = field(default_factory=list)
    records_read: int = 0
    bad_records: int = 0
    duplicates_dropped: int = 0
    kept: int = 0
    no_description: int = 0
    no_cpc: int = 0
    no_title: int = 0
    seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"files {self.files_read} (bad {len(self.bad_files)}) · records {self.records_read:,} "
            f"· bad records {self.bad_records} · duplicates dropped {self.duplicates_dropped:,} "
            f"· kept {self.kept:,} · no description {self.no_description:,} "
            f"· no classification {self.no_cpc:,} · no title {self.no_title:,} "
            f"· {self.seconds:.1f}s"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def load_patents(
    root: Path | str,
    *,
    max_files: int | None = None,
    keep_description: bool = True,
    keep_claims: bool = True,
) -> tuple[list[Patent], LoadReport]:
    """Load, clean and dedupe every file under root. One bad file or record never
    stops the load; it is counted in the report instead."""
    start = time.perf_counter()
    root = Path(root)
    files = find_data_files(root)[:max_files] if max_files else find_data_files(root)
    report = LoadReport()
    parsed: list[Patent] = []
    for path in files:
        try:
            records = read_json_file(path)
        except (OSError, ValueError) as err:  # JSONDecodeError / UnicodeDecodeError are ValueErrors
            report.bad_files.append(f"{path.relative_to(root)}: {err}")
            log.warning("skipping bad file %s: %s", path, err)
            continue
        report.files_read += 1
        source = str(path.relative_to(root))
        for raw in records:
            report.records_read += 1
            try:
                parsed.append(parse_record(raw, source, keep_description, keep_claims))
            except BadRecord:
                report.bad_records += 1

    patents, report.duplicates_dropped = dedupe(parsed)
    report.kept = len(patents)
    report.no_description = sum(not p.has_description for p in patents)
    report.no_cpc = sum(p.cpc is None for p in patents)
    report.no_title = sum(p.title is None for p in patents)
    report.seconds = time.perf_counter() - start
    return patents, report


# ---------------------------------------------------------------- (de)serialization


def patent_to_dict(p: Patent) -> dict:
    return {
        "doc_number": p.doc_number,
        "title": p.title,
        "abstract": p.abstract,
        "claims": [[c.number, c.text, list(c.depends_on)] for c in p.claims],
        "paragraphs": p.paragraphs,
        "cpc": p.cpc.raw if p.cpc else None,
        "pub_date": p.pub_date.isoformat() if p.pub_date else None,
        "year": p.year,
        "filename": p.filename,
        "source_file": p.source_file,
        "has_description": p.has_description,
    }


def patent_from_dict(d: dict) -> Patent:
    return Patent(
        doc_number=d["doc_number"],
        title=d["title"],
        abstract=d["abstract"],
        claims=[Claim(n, text, tuple(deps)) for n, text, deps in d["claims"]],
        paragraphs=d["paragraphs"],
        cpc=parse_cpc(d["cpc"]),
        pub_date=date.fromisoformat(d["pub_date"]) if d["pub_date"] else None,
        year=d["year"],
        filename=d["filename"],
        source_file=d["source_file"],
        has_description=d["has_description"],
    )
