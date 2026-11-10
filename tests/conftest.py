"""Shared test fixtures: tiny fake datasets in BOTH real formats.

The records copy the real quirks: blank paragraphs, "(canceled)" claims,
claims without numbers, "N/A" values, duplicates, a broken file and the
large set's sample.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from patent_search.chunk import ChunkConfig
from patent_search.embed import HashingEmbedder
from patent_search.index import PatentIndex
from patent_search.load import load_patents
from patent_search.search import SearchEngine

SMALL_FILES = {
    "patents_ipa240215.json": [
        {
            "title": "PNEUMATIC TIRE",
            "doc_number": "20240000001",
            "filename": "US20240000001A1-20240215.XML",
            "abstract": "A pneumatic tire has a tread with circumferential grooves that reduce road noise.",
            "detailed_description": [
                "",
                "The tread portion includes noise reducing grooves arranged around the tire.",
                "",
                "As shown inFIG.1the sidewall protects the carcass.",
            ],
            "claims": [
                "a tread portion having circumferential grooves; and a sidewall portion",
                "2 . The pneumatic tire according to claim 1 , wherein the grooves reduce road noise.",
                "3 - 4 . (canceled)",
                "5 . The pneumatic tire according to claim 1, wherein the bead core is steel.",
            ],
            "bibtex": "@patent{20240000001}",
            "classification": "B60C1103FI",
        },
        {
            "title": "SPOKE",
            "doc_number": "20240000002",
            "filename": "US20240000002A1-20240215.XML",
            "abstract": "A spoke for a bicycle wheel has an axle body and friction enhancing connecting elements.",
            "detailed_description": [""],
            "claims": ["an axle body having a middle segment; and two connecting elements"],
            "bibtex": "@patent{20240000002}",
            "classification": "B60B104FI",
        },
    ],
    "patents_ipa240222.json": [
        {
            "title": "TRAILER HITCH",
            "doc_number": "20240000003",
            "filename": "US20240000003A1-20240222.XML",
            "abstract": "A trailer hitch coupling with a ball mount and a locking pin.",
            "detailed_description": ["The ball mount is inserted into the receiver and locked with a pin."],
            "claims": ["a ball mount; a receiver; and a locking pin"],
            "bibtex": "@patent{20240000003}",
            "classification": "B60D100FI",
        },
        {
            "title": "WHEEL RIM",
            "doc_number": "20240000004",
            "filename": "US20240000004A1-20240222.XML",
            "abstract": "A light alloy wheel rim with spokes made of carbon fiber.",
            "detailed_description": ["The carbon fiber spokes connect the rim to the hub."],
            "claims": ["1. A wheel rim comprising carbon fiber spokes.", "2. The wheel rim of claim 1, wherein the hub is aluminum."],
            "bibtex": "@patent{20240000004}",
            "classification": "B60B2700FI",
        },
    ],
}

_BATTERY = {
    "title": "N/A",
    "doc_number": "20150000001",
    "filename": "US20150000001A1.XML",
    "abstract": "A battery pack with cooling plates between battery cells.",
    "detailed_description": ["Cooling plates carry liquid coolant between the cells."],
    "claims": ["1. A battery pack comprising cooling plates.", "2. The battery pack of claim 1, wherein the coolant is glycol."],
    "classification": "N/A",
}
_CHARGER_SHORT = {
    "title": "WIRELESS CHARGING PAD",
    "doc_number": "20180000002",
    "filename": "US20180000002A1.XML",
    "abstract": "A charging pad.",
    "detailed_description": [],
    "claims": ["1. A charging pad."],
    "classification": "H02J50/10",
}
_CHARGER_LONG = {**_CHARGER_SHORT, "abstract": "A wireless charging pad with a coil that charges a phone by induction."}
_WHEEL = {
    "title": "VEHICLE WHEEL",
    "doc_number": "20100000003",
    "filename": "US20100000003A1.XML",
    "abstract": "A vehicle wheel with a rim and an air hole cover.",
    "detailed_description": ["The wall portion partially closes the air holes between spokes."],
    "claims": ["1. A vehicle wheel comprising a rim and a wall portion."],
    "classification": "B60B3/02",
}

LARGE_FILES = {
    "job_000/patents_batch_0001.json": [_BATTERY, _CHARGER_SHORT, _WHEEL, "not a record", {"title": "no number"}],
    "job_001/patents_batch_0001.json": [_CHARGER_LONG],
}


def write_dataset(root: Path, files: dict[str, list]) -> Path:
    for rel, records in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records), encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def small_dir(tmp_path_factory) -> Path:
    return write_dataset(tmp_path_factory.mktemp("small"), SMALL_FILES)


@pytest.fixture(scope="session")
def large_dir(tmp_path_factory) -> Path:
    root = write_dataset(tmp_path_factory.mktemp("large"), LARGE_FILES)
    (root / "job_001" / "patents_batch_0002.json").write_text('[{"title": "trunc', encoding="utf-8")  # broken
    (root / "sample.json").write_text(json.dumps(_WHEEL), encoding="utf-8")  # must be ignored
    return root


@pytest.fixture(scope="session")
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=128)


@pytest.fixture(scope="session")
def small_index(small_dir, embedder) -> PatentIndex:
    patents, _ = load_patents(small_dir)
    return PatentIndex.build(patents, embedder, ChunkConfig())


@pytest.fixture(scope="session")
def engine(small_index, embedder) -> SearchEngine:
    return SearchEngine(small_index, embedder)
