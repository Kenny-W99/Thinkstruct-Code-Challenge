"""Settings that are written in more than one place must agree with config.py."""

import re

import pytest

from patent_search import config

ROOT = config.PROJECT_ROOT
# The API Docker image contains only src/ and tests/, so these checks run in a checkout.
needs_web = pytest.mark.skipif(not (ROOT / "web").is_dir(), reason="web/ is not in the API Docker image")
needs_docker = pytest.mark.skipif(not (ROOT / "docker").is_dir(), reason="docker/ is not in the API Docker image")


@needs_docker
def test_docker_image_bakes_in_the_configured_models():
    dockerfile = (ROOT / "docker" / "api.Dockerfile").read_text()
    args = dict(re.findall(r"^ARG (\w+)=(\S+)$", dockerfile, re.MULTILINE))
    assert args["EMBEDDING_MODEL"] == config.EMBEDDING_MODEL
    assert args["RERANK_MODEL"] == config.RERANK_MODEL


@needs_web
def test_web_app_uses_the_same_auto_rerank_cutoff():
    app = (ROOT / "web" / "src" / "App.tsx").read_text()
    assert f"RERANK_AUTO_MAX_WORDS = {config.RERANK_AUTO_MAX_WORDS};" in app


@needs_web
def test_web_app_uses_the_same_default_rerank_candidates():
    query = (ROOT / "web" / "src" / "lib" / "query.ts").read_text()
    assert f"rerankCandidates: {config.RERANK_CANDIDATES}," in query


@needs_web
def test_web_types_list_the_same_choices_as_the_api():
    from patent_search.chunk import KINDS
    from patent_search.search import MODES, RERANK_CHOICES, STRATEGIES

    types = (ROOT / "web" / "src" / "api" / "types.ts").read_text()

    def union(name: str) -> set[str]:
        return set(re.findall(r'"(\w+)"', re.search(rf"export type {name} = ([^;]+);", types).group(1)))

    assert union("Kind") == set(KINDS)
    assert union("Mode") == set(MODES)
    assert union("Strategy") == set(STRATEGIES)
    assert union("RerankChoice") == set(RERANK_CHOICES)


def test_saved_paths_are_project_relative(tmp_path):
    from patent_search.loader import portable_path

    assert portable_path(ROOT / "data" / "patent_data_small") == "data/patent_data_small"
    assert portable_path(tmp_path) == str(tmp_path)  # outside the project: kept as is
