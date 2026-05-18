from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from partoguard.core.extraction import gemma_adapter as ga
from partoguard.core.extraction.gemma_adapter import (
    RemoteGemmaExtractor,
    _build_remote_extraction_prompt,
    _build_remote_verify_prompt,
    _remote_invalidate_marker_cache,
    build_verifier,
)
from partoguard.core.schemas.contracts import DilationPoint, ExtractionResult, TemplateID

_FAKE_MARKER = "<__media_FAKE_MARKER_FOR_TESTS__>"


def _extraction(points: list[DilationPoint] | None = None) -> ExtractionResult:
    return ExtractionResult(
        template_id=TemplateID.MODIFIED_WHO_V1,
        chart_present=True,
        registered=True,
        points=points
        or [
            DilationPoint(x_hours=4.0, dilation_cm=6.0, confidence=0.8),
            DilationPoint(x_hours=5.0, dilation_cm=7.0, confidence=0.8),
        ],
        overall_confidence=0.8,
    )


class _FakeResponse:
    def __init__(self, payload: object, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload) if isinstance(payload, (dict, list)) else str(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"status={self.status_code}")

    def json(self) -> object:
        return self._payload


class _FakeRequests:
    """Captures HTTP traffic so tests can assert on URLs, payloads, and counts."""

    def __init__(
        self,
        props_payload: dict[str, Any] | None = None,
        completion_payload: dict[str, Any] | None = None,
        completion_error: Exception | None = None,
        props_error: Exception | None = None,
    ):
        self.props_payload = props_payload or {"media_marker": _FAKE_MARKER}
        self.completion_payload = completion_payload or {"content": '{"p":[]}'}
        self.completion_error = completion_error
        self.props_error = props_error
        self.get_calls: list[tuple[str, dict[str, Any]]] = []
        self.post_calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.get_calls.append((url, kwargs))
        if self.props_error is not None:
            raise self.props_error
        return _FakeResponse(self.props_payload)

    def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.post_calls.append((url, kwargs))
        if self.completion_error is not None:
            raise self.completion_error
        return _FakeResponse(self.completion_payload)


@pytest.fixture(autouse=True)
def _clear_marker_cache():
    _remote_invalidate_marker_cache()
    yield
    _remote_invalidate_marker_cache()


@pytest.fixture
def fake_requests(monkeypatch: pytest.MonkeyPatch) -> _FakeRequests:
    fake = _FakeRequests()

    class _RequestException(Exception):
        pass

    fake_module = type(
        "FakeRequestsModule",
        (),
        {
            "get": fake.get,
            "post": fake.post,
            "RequestException": _RequestException,
            "HTTPError": _RequestException,
        },
    )
    monkeypatch.setitem(__import__("sys").modules, "requests", fake_module)
    return fake


@pytest.fixture
def crop_path(tmp_path: Path) -> Path:
    p = tmp_path / "crop.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\nfake-image-bytes")
    return p


def test_build_verifier_remote_returns_extractor() -> None:
    verifier = build_verifier(None, use_remote=True, remote_url="http://example:8080/completion")
    assert isinstance(verifier, RemoteGemmaExtractor)
    assert verifier.url == "http://example:8080/completion"


def test_build_verifier_remote_uses_default_url() -> None:
    verifier = build_verifier(None, use_remote=True)
    assert isinstance(verifier, RemoteGemmaExtractor)
    assert verifier.url == "http://localhost:8080/completion"


@pytest.mark.parametrize(
    "url,expected_base",
    [
        ("http://example:8080/completion", "http://example:8080"),
        ("http://example:8080/", "http://example:8080"),
        ("http://example:8080", "http://example:8080"),
        ("http://example:8080/v1/chat/completions", "http://example:8080"),
        ("http://example:8080/v1/completions", "http://example:8080"),
        ("https://api.example.com/llama/completion", "https://api.example.com/llama"),
    ],
)
def test_base_url_strips_known_suffixes(url: str, expected_base: str) -> None:
    extractor = RemoteGemmaExtractor(url=url)
    assert extractor._base_url() == expected_base


def test_base_url_does_not_chop_completions_when_endpoint_is_completion() -> None:
    """Regression: '/completion' must not match '/v1/completions' or vice versa."""
    extractor = RemoteGemmaExtractor(url="http://example:8080/v1/completions")
    assert extractor._base_url() == "http://example:8080"
    extractor2 = RemoteGemmaExtractor(url="http://example:8080/completion")
    assert extractor2._base_url() == "http://example:8080"


def test_extract_from_image_missing_crop_returns_manual_review(tmp_path: Path) -> None:
    extractor = RemoteGemmaExtractor(url="http://localhost:8080/completion")
    result = extractor.extract_from_image(tmp_path / "nope.png")
    assert result.points == []
    assert "manual_review" in result.warnings
    assert "gemma_remote_extract_missing_crop" in result.warnings


def test_extract_from_image_happy_path(fake_requests: _FakeRequests, crop_path: Path) -> None:
    fake_requests.completion_payload = {
        "content": '{"p":[[4.0,6.0,0.9],[5.0,7.0,0.85]]}'
    }
    extractor = RemoteGemmaExtractor(url="http://srv:8080/completion")
    result = extractor.extract_from_image(crop_path)

    assert len(result.points) == 2
    assert result.points[0].x_hours == 4.0
    assert result.points[0].dilation_cm == 6.0
    assert result.points[1].x_hours == 5.0
    assert result.points[1].dilation_cm == 7.0
    assert "gemma_remote_extracted" in result.warnings
    assert "manual_review" not in result.warnings


def test_extract_from_image_empty_payload_routes_clean_empty(
    fake_requests: _FakeRequests, crop_path: Path
) -> None:
    fake_requests.completion_payload = {"content": '{"p":[]}'}
    extractor = RemoteGemmaExtractor(url="http://srv:8080/completion")
    result = extractor.extract_from_image(crop_path)

    assert result.points == []
    assert "gemma_extracted_no_marks" in result.warnings
    assert "manual_review" not in result.warnings


def test_extract_from_image_invalid_json_routes_manual_review(
    fake_requests: _FakeRequests, crop_path: Path
) -> None:
    fake_requests.completion_payload = {"content": "not json at all {{{"}
    extractor = RemoteGemmaExtractor(url="http://srv:8080/completion")
    result = extractor.extract_from_image(crop_path)

    assert result.points == []
    assert "manual_review" in result.warnings
    assert "gemma_remote_extract_invalid_json" in result.warnings


def test_extract_from_image_server_unreachable_routes_manual_review(
    monkeypatch: pytest.MonkeyPatch, crop_path: Path
) -> None:
    boom_msg = "connection refused"

    class _Boom(Exception):
        pass

    def _post_boom(*_a: Any, **_k: Any) -> _FakeResponse:
        raise _Boom(boom_msg)

    def _get_ok(*_a: Any, **_k: Any) -> _FakeResponse:
        return _FakeResponse({"media_marker": _FAKE_MARKER})

    fake_module = type(
        "FakeRequestsModule",
        (),
        {"get": _get_ok, "post": _post_boom, "RequestException": _Boom, "HTTPError": _Boom},
    )
    monkeypatch.setitem(__import__("sys").modules, "requests", fake_module)

    extractor = RemoteGemmaExtractor(url="http://srv:8080/completion")
    result = extractor.extract_from_image(crop_path)

    assert result.points == []
    assert "manual_review" in result.warnings
    assert any(w.startswith("gemma_remote_extract_unavailable:") for w in result.warnings)


def test_extract_from_image_props_unreachable_routes_manual_review(
    monkeypatch: pytest.MonkeyPatch, crop_path: Path
) -> None:
    class _Boom(Exception):
        pass

    def _get_boom(*_a: Any, **_k: Any) -> _FakeResponse:
        raise _Boom("props unreachable")

    fake_module = type(
        "FakeRequestsModule",
        (),
        {"get": _get_boom, "post": lambda *a, **k: _FakeResponse({}), "RequestException": _Boom, "HTTPError": _Boom},
    )
    monkeypatch.setitem(__import__("sys").modules, "requests", fake_module)

    extractor = RemoteGemmaExtractor(url="http://srv:8080/completion")
    result = extractor.extract_from_image(crop_path)

    assert result.points == []
    assert "manual_review" in result.warnings
    assert any(w.startswith("gemma_remote_extract_unavailable:") for w in result.warnings)


def test_marker_cached_across_multiple_extractions(
    fake_requests: _FakeRequests, crop_path: Path
) -> None:
    fake_requests.completion_payload = {"content": '{"p":[]}'}
    extractor = RemoteGemmaExtractor(url="http://srv:8080/completion")
    for _ in range(5):
        _ = extractor.extract_from_image(crop_path)
    assert len(fake_requests.get_calls) == 1
    assert len(fake_requests.post_calls) == 5


def test_marker_cache_isolates_distinct_servers(
    fake_requests: _FakeRequests, crop_path: Path
) -> None:
    fake_requests.completion_payload = {"content": '{"p":[]}'}
    extractor_a = RemoteGemmaExtractor(url="http://a:8080/completion")
    extractor_b = RemoteGemmaExtractor(url="http://b:8080/completion")
    _ = extractor_a.extract_from_image(crop_path)
    _ = extractor_b.extract_from_image(crop_path)
    _ = extractor_a.extract_from_image(crop_path)
    _ = extractor_b.extract_from_image(crop_path)
    assert len(fake_requests.get_calls) == 2
    seen_hosts = sorted({url for url, _ in fake_requests.get_calls})
    assert seen_hosts == ["http://a:8080/props", "http://b:8080/props"]


def test_extract_payload_sends_canonical_format(
    fake_requests: _FakeRequests, crop_path: Path
) -> None:
    fake_requests.completion_payload = {"content": '{"p":[]}'}
    extractor = RemoteGemmaExtractor(url="http://srv:8080/completion")
    _ = extractor.extract_from_image(crop_path)
    assert len(fake_requests.post_calls) == 1
    url, kwargs = fake_requests.post_calls[0]
    assert url == "http://srv:8080/completion"
    body = kwargs.get("json", {})
    assert isinstance(body["prompt"], dict), "Must send prompt as object, not legacy string"
    assert "prompt_string" in body["prompt"]
    assert "multimodal_data" in body["prompt"]
    assert isinstance(body["prompt"]["multimodal_data"], list)
    assert len(body["prompt"]["multimodal_data"]) == 1
    assert isinstance(body["prompt"]["multimodal_data"][0], str)
    prompt_string = body["prompt"]["prompt_string"]
    assert _FAKE_MARKER in prompt_string
    assert "<|turn>user" in prompt_string
    assert "<|turn>model" in prompt_string
    assert not prompt_string.startswith("<bos>"), (
        "Tokenizer adds BOS automatically; literal <bos> would double-tokenize."
    )
    assert "[img-" not in prompt_string, "Legacy LLaVA-style markers must not appear"
    assert "image_data" not in body, "Legacy image_data array must not be sent"


def test_verify_no_points_passes_through_extraction(fake_requests: _FakeRequests) -> None:
    extraction = ExtractionResult(
        template_id=TemplateID.MODIFIED_WHO_V1,
        chart_present=True,
        registered=True,
        points=[],
        overall_confidence=0.0,
    )
    extractor = RemoteGemmaExtractor(url="http://srv:8080/completion")
    result = extractor.verify(extraction, chart_crop_path=None)
    assert result == extraction
    assert len(fake_requests.post_calls) == 0


def test_verify_missing_crop_routes_manual_review(fake_requests: _FakeRequests) -> None:
    extractor = RemoteGemmaExtractor(url="http://srv:8080/completion")
    result = extractor.verify(_extraction(), chart_crop_path=None)
    assert "manual_review" in result.warnings
    assert "gemma_remote_verify_missing_crop" in result.warnings
    assert len(fake_requests.post_calls) == 0


def test_verify_happy_path(fake_requests: _FakeRequests, crop_path: Path) -> None:
    fake_requests.completion_payload = {
        "content": '{"accepted_points":[{"x_hours":4.0,"dilation_cm":6.0,"confidence":0.7}]}'
    }
    extractor = RemoteGemmaExtractor(url="http://srv:8080/completion")
    result = extractor.verify(_extraction(), chart_crop_path=crop_path)
    assert len(result.points) == 1
    assert "gemma_remote_verified" in result.warnings


def test_remote_extraction_prompt_describes_chart_and_schema() -> None:
    prompt = _build_remote_extraction_prompt()
    assert "0-12" in prompt
    assert "0-10" in prompt or "0.0-10.0" in prompt
    assert '"p"' in prompt
    assert "cervicograph" in prompt.lower()
    assert "ignore" in prompt.lower() or "Ignore" in prompt


def test_remote_verify_prompt_includes_candidates() -> None:
    extraction = _extraction()
    prompt = _build_remote_verify_prompt(extraction)
    assert "accepted_points" in prompt
    assert "4.0" in prompt
    assert "6.0" in prompt
