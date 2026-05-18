import sys

from pathlib import Path

from partoguard.core.extraction.gemma_adapter import LiteRTGemmaE2BVerifier, LocalGemmaVerifier, StubVerifier, _bounded_extracted_points_from_payload, _build_litert_extraction_prompt, _build_litert_prompt, _normalize_extraction_payload, build_verifier
from partoguard.core.schemas.contracts import DilationPoint, ExtractionResult, TemplateID


def _extraction() -> ExtractionResult:
    return ExtractionResult(
        template_id=TemplateID.MODIFIED_WHO_V1,
        chart_present=True,
        registered=True,
        points=[DilationPoint(x_hours=4.0, dilation_cm=6.0, confidence=0.8)],
        overall_confidence=0.8,
    )


def _multi_point_extraction() -> ExtractionResult:
    return ExtractionResult(
        template_id=TemplateID.MODIFIED_WHO_V1,
        chart_present=True,
        registered=True,
        points=[
            DilationPoint(x_hours=4.0, dilation_cm=6.0, confidence=0.8),
            DilationPoint(x_hours=5.0, dilation_cm=7.0, confidence=0.8),
        ],
        overall_confidence=0.8,
    )


def test_stub_verifier_passes_through_extraction():
    extraction = _extraction()
    assert StubVerifier().verify(extraction) == extraction


def test_local_gemma_verifier_accepts_valid_json():
    command = [
        sys.executable,
        "-c",
        'print("{\\\"accepted_points\\\":[{\\\"x_hours\\\":4.0,\\\"dilation_cm\\\":6.0,\\\"confidence\\\":0.7}]}")',
    ]
    result = LocalGemmaVerifier(command=command).verify(_extraction())

    assert result.points[0].source == "cv_gemma_verified"
    assert result.overall_confidence == 0.7
    assert "gemma_verified" in result.warnings


def test_local_gemma_verifier_rejects_invented_points():
    command = [
        sys.executable,
        "-c",
        'print("{\\\"accepted_points\\\":[{\\\"x_hours\\\":9.0,\\\"dilation_cm\\\":9.0,\\\"confidence\\\":0.9}]}")',
    ]
    result = LocalGemmaVerifier(command=command).verify(_extraction())

    assert "manual_review" in result.warnings
    assert "gemma_verifier_invalid_json" in result.warnings
    assert result.points == []
    assert result.overall_confidence == 0.0


def test_local_gemma_verifier_rejects_duplicate_accepted_points():
    command = [
        sys.executable,
        "-c",
        'print("{\\\"accepted_points\\\":[{\\\"x_hours\\\":4.0,\\\"dilation_cm\\\":6.0},{\\\"x_hours\\\":4.0,\\\"dilation_cm\\\":6.0}]}")',
    ]
    result = LocalGemmaVerifier(command=command).verify(_multi_point_extraction())

    assert "manual_review" in result.warnings
    assert "gemma_verifier_invalid_json" in result.warnings
    assert result.points == []
    assert result.overall_confidence == 0.0


def test_local_gemma_verifier_invalid_json_triggers_manual_review():
    command = [sys.executable, "-c", "print('not json')"]
    result = LocalGemmaVerifier(command=command).verify(_extraction())

    assert "manual_review" in result.warnings
    assert result.points == []
    assert result.overall_confidence == 0.0


def test_local_gemma_verifier_command_failure_triggers_manual_review():
    command = [sys.executable, "-c", "raise SystemExit(2)"]
    result = LocalGemmaVerifier(command=command).verify(_extraction())

    assert "gemma_verifier_failed" in result.warnings
    assert "manual_review" in result.warnings


def test_build_verifier_defaults_to_stub():
    assert isinstance(build_verifier(None), StubVerifier)


def test_build_verifier_creates_local_verifier():
    verifier = build_verifier(f"{sys.executable} -c print")
    assert isinstance(verifier, LocalGemmaVerifier)


def test_build_verifier_creates_litert_e2b_verifier():
    verifier = build_verifier(
        None, use_litert_e2b=True, use_daemon=False, litert_bin=Path("/tmp/litert-lm")
    )
    assert isinstance(verifier, LiteRTGemmaE2BVerifier)
    assert verifier.litert_bin == Path("/tmp/litert-lm")


def test_build_verifier_defaults_to_daemon_for_litert_e2b():
    from partoguard.core.extraction.gemma_adapter import LiteRTGemmaDaemonExtractor

    verifier = build_verifier(None, use_litert_e2b=True)
    assert isinstance(verifier, LiteRTGemmaDaemonExtractor)
    assert verifier.model_reference == "gemma-4-E2B-it.litertlm"


def test_build_verifier_defaults_to_daemon_for_litert_e4b():
    from partoguard.core.extraction.gemma_adapter import LiteRTGemmaDaemonExtractor

    verifier = build_verifier(None, use_litert_e4b=True)
    assert isinstance(verifier, LiteRTGemmaDaemonExtractor)
    assert verifier.model_reference == "gemma-4-E4B-it.litertlm"
    assert verifier.huggingface_repo == "litert-community/gemma-4-E4B-it-litert-lm"


def test_build_verifier_rejects_command_plus_litert_e2b():
    try:
        build_verifier("fake", use_litert_e2b=True)
    except ValueError as exc:
        assert "--gemma-command" in str(exc) and "--gemma-litert" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_litert_e2b_unavailable_triggers_manual_review():
    result = LiteRTGemmaE2BVerifier(litert_bin=Path("/tmp/missing-litert-lm"), timeout_seconds=0.1).verify(_extraction())

    assert "manual_review" in result.warnings
    assert any(warning.startswith("gemma_e2b_unavailable") for warning in result.warnings)
    assert result.points == []


def test_litert_prompt_without_image_omits_image_clause():
    prompt = _build_litert_prompt(_extraction(), with_image=False)
    assert "attached image" not in prompt
    assert "accepted_points" in prompt


def test_litert_prompt_with_image_includes_image_clause():
    prompt = _build_litert_prompt(_extraction(), with_image=True)
    assert "attached image" in prompt
    assert "visually confirm" in prompt
    assert "accepted_points" in prompt


def test_litert_e2b_with_nonexistent_chart_crop_still_runs():
    result = LiteRTGemmaE2BVerifier(
        litert_bin=Path("/tmp/missing-litert-lm"),
        timeout_seconds=0.1,
    ).verify(_extraction(), chart_crop_path=Path("/tmp/does-not-exist-xyz.png"))
    assert "manual_review" in result.warnings


def test_litert_e2b_passes_attachment_when_crop_exists(tmp_path: Path) -> None:
    fake_crop = tmp_path / "crop.png"
    _ = fake_crop.write_bytes(b"\x89PNG\r\n\x1a\n")

    captured_args_path = tmp_path / "args.txt"
    fake_litert = tmp_path / "litert-lm"
    fake_litert_text = (
        "#!/usr/bin/env python3\n"
        "import sys, json\n"
        f"open({str(captured_args_path)!r}, 'w').write(json.dumps(sys.argv))\n"
        "print('{\"accepted_points\":[{\"x_hours\":4.0,\"dilation_cm\":6.0,\"confidence\":0.6}]}')\n"
    )
    _ = fake_litert.write_text(fake_litert_text)
    fake_litert.chmod(0o755)

    result = LiteRTGemmaE2BVerifier(
        litert_bin=fake_litert,
        timeout_seconds=10.0,
        hf_home=tmp_path / "hf",
    ).verify(_extraction(), chart_crop_path=fake_crop)

    args = captured_args_path.read_text()
    assert "--attachment" in args
    assert str(fake_crop.resolve()) in args
    assert "--vision-backend" in args
    assert "cpu" in args
    assert result.points[0].source == "cv_gemma_verified"
    assert "gemma_e2b_verified" in result.warnings


def test_litert_e2b_omits_attachment_when_crop_missing(tmp_path: Path) -> None:
    captured_args_path = tmp_path / "args.txt"
    fake_litert = tmp_path / "litert-lm"
    fake_litert_text = (
        "#!/usr/bin/env python3\n"
        "import sys, json\n"
        f"open({str(captured_args_path)!r}, 'w').write(json.dumps(sys.argv))\n"
        "print('{\"accepted_points\":[]}')\n"
    )
    _ = fake_litert.write_text(fake_litert_text)
    fake_litert.chmod(0o755)

    _ = LiteRTGemmaE2BVerifier(
        litert_bin=fake_litert,
        timeout_seconds=10.0,
        hf_home=tmp_path / "hf",
    ).verify(_extraction(), chart_crop_path=None)

    args = captured_args_path.read_text()
    assert "--attachment" not in args
    assert "--vision-backend" not in args


def test_extraction_prompt_describes_chart_and_schema():
    prompt = _build_litert_extraction_prompt()
    assert "0-12" in prompt or "0 to 12" in prompt
    assert "0-10" in prompt or "0 to 10" in prompt
    assert '"p"' in prompt
    assert "Do not invent" in prompt or "Do NOT invent" in prompt
    assert "blank" in prompt.lower() or "empty" in prompt.lower()
    assert "cervicograph" in prompt.lower()


def test_bounded_extracted_points_accepts_valid_payload():
    payload = {"points": [
        {"x_hours": 4.0, "dilation_cm": 6.0, "confidence": 0.8},
        {"x_hours": 5.0, "dilation_cm": 7.0, "confidence": 0.7},
    ]}
    points = _bounded_extracted_points_from_payload(payload)
    assert [(p.x_hours, p.dilation_cm) for p in points] == [(4.0, 6.0), (5.0, 7.0)]
    assert all(p.source == "gemma_e2b_extracted" for p in points)


def test_normalize_extraction_payload_compact_to_verbose():
    compact = {"p": [[4.0, 6.0, 0.8], [5.0, 7.0, 0.7]]}
    out = _normalize_extraction_payload(compact)
    assert out == {"points": [
        {"x_hours": 4.0, "dilation_cm": 6.0, "confidence": 0.8},
        {"x_hours": 5.0, "dilation_cm": 7.0, "confidence": 0.7},
    ]}


def test_normalize_extraction_payload_compact_without_confidence():
    compact = {"p": [[4.0, 6.0]]}
    out = _normalize_extraction_payload(compact)
    assert out == {"points": [{"x_hours": 4.0, "dilation_cm": 6.0, "confidence": 0.5}]}


def test_normalize_extraction_payload_passes_through_verbose():
    verbose = {"points": [{"x_hours": 4.0, "dilation_cm": 6.0, "confidence": 0.8}]}
    assert _normalize_extraction_payload(verbose) == verbose


def test_normalize_extraction_payload_rejects_bad_compact():
    for bad in [{"p": "x"}, {"p": [42]}, {"p": [[4.0]]}]:
        try:
            _ = _normalize_extraction_payload(bad)
        except (ValueError, TypeError):
            continue
        raise AssertionError(f"expected error for {bad}")


def test_bounded_extracted_points_rounds_to_half_units():
    payload = {"points": [
        {"x_hours": 3.7, "dilation_cm": 5.2, "confidence": 0.6},
        {"x_hours": 4.6, "dilation_cm": 6.4, "confidence": 0.6},
    ]}
    points = _bounded_extracted_points_from_payload(payload)
    assert points[0].x_hours == 3.5
    assert points[0].dilation_cm == 5.0
    assert points[1].x_hours == 4.5
    assert points[1].dilation_cm == 6.5


def test_bounded_extracted_points_rejects_out_of_range():
    for bad in [
        {"x_hours": -1.0, "dilation_cm": 5.0, "confidence": 0.5},
        {"x_hours": 13.0, "dilation_cm": 5.0, "confidence": 0.5},
        {"x_hours": 4.0, "dilation_cm": 11.0, "confidence": 0.5},
        {"x_hours": 4.0, "dilation_cm": 5.0, "confidence": 1.5},
    ]:
        try:
            _ = _bounded_extracted_points_from_payload({"points": [bad]})
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_bounded_extracted_points_rejects_duplicates():
    payload = {"points": [
        {"x_hours": 4.0, "dilation_cm": 6.0, "confidence": 0.8},
        {"x_hours": 4.0, "dilation_cm": 6.0, "confidence": 0.7},
    ]}
    try:
        _ = _bounded_extracted_points_from_payload(payload)
    except ValueError:
        return
    raise AssertionError("expected ValueError for duplicates")


def test_bounded_extracted_points_rejects_implausible_trajectory():
    payload = {"points": [
        {"x_hours": 4.0, "dilation_cm": 7.0, "confidence": 0.8},
        {"x_hours": 5.0, "dilation_cm": 5.0, "confidence": 0.8},
    ]}
    try:
        _ = _bounded_extracted_points_from_payload(payload)
    except ValueError:
        return
    raise AssertionError("expected ValueError for implausible trajectory")


def test_bounded_extracted_points_rejects_too_many():
    payload = {"points": [
        {"x_hours": float(i) * 0.5, "dilation_cm": 4.0 + i * 0.1, "confidence": 0.5}
        for i in range(25)
    ]}
    try:
        _ = _bounded_extracted_points_from_payload(payload)
    except ValueError:
        return
    raise AssertionError("expected ValueError for too many points")


def test_bounded_extracted_points_rejects_bad_schema():
    for bad in [None, [], {"points": "x"}, {"points": [42]}, {"points": [{"x_hours": "a", "dilation_cm": 5.0}]}]:
        try:
            _ = _bounded_extracted_points_from_payload(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_extract_from_image_missing_crop_returns_manual_review(tmp_path: Path) -> None:
    verifier = LiteRTGemmaE2BVerifier(litert_bin=tmp_path / "litert-lm", hf_home=tmp_path / "hf")
    result = verifier.extract_from_image(tmp_path / "nope.png")
    assert result.points == []
    assert "manual_review" in result.warnings
    assert "gemma_e2b_extract_missing_crop" in result.warnings


def test_extract_from_image_runs_litert_and_returns_points(tmp_path: Path) -> None:
    fake_crop = tmp_path / "crop.png"
    _ = fake_crop.write_bytes(b"\x89PNG\r\n\x1a\n")
    captured_args_path = tmp_path / "args.txt"
    fake_litert = tmp_path / "litert-lm"
    payload = '{"points":[{"x_hours":4.0,"dilation_cm":6.0,"confidence":0.8},{"x_hours":5.0,"dilation_cm":7.0,"confidence":0.7}]}'
    fake_litert_text = (
        "#!/usr/bin/env python3\n"
        "import sys, json\n"
        f"open({str(captured_args_path)!r}, 'w').write(json.dumps(sys.argv))\n"
        f"print({payload!r})\n"
    )
    _ = fake_litert.write_text(fake_litert_text)
    fake_litert.chmod(0o755)

    result = LiteRTGemmaE2BVerifier(
        litert_bin=fake_litert,
        timeout_seconds=10.0,
        hf_home=tmp_path / "hf",
    ).extract_from_image(fake_crop)

    args = captured_args_path.read_text()
    assert "--attachment" in args
    assert str(fake_crop.resolve()) in args
    assert "--vision-backend" in args
    assert len(result.points) == 2
    assert result.points[0].source == "gemma_raw"
    assert "gemma_e2b_extracted" in result.warnings
    assert result.template_id == TemplateID.MODIFIED_WHO_V1


def test_extract_from_image_invalid_json_routes_manual_review(tmp_path: Path) -> None:
    fake_crop = tmp_path / "crop.png"
    _ = fake_crop.write_bytes(b"\x89PNG\r\n\x1a\n")
    fake_litert = tmp_path / "litert-lm"
    _ = fake_litert.write_text("#!/usr/bin/env python3\nprint('not json')\n")
    fake_litert.chmod(0o755)

    result = LiteRTGemmaE2BVerifier(
        litert_bin=fake_litert,
        timeout_seconds=10.0,
        hf_home=tmp_path / "hf",
    ).extract_from_image(fake_crop)

    assert result.points == []
    assert "manual_review" in result.warnings
    assert "gemma_e2b_extract_invalid_json" in result.warnings


def test_extract_from_image_empty_points_is_not_manual_review(tmp_path: Path) -> None:
    fake_crop = tmp_path / "crop.png"
    _ = fake_crop.write_bytes(b"\x89PNG\r\n\x1a\n")
    fake_litert = tmp_path / "litert-lm"
    _ = fake_litert.write_text("#!/usr/bin/env python3\nprint('{\"points\":[]}')\n")
    fake_litert.chmod(0o755)

    result = LiteRTGemmaE2BVerifier(
        litert_bin=fake_litert,
        timeout_seconds=10.0,
        hf_home=tmp_path / "hf",
    ).extract_from_image(fake_crop)

    # Phase 1: empty extraction is a legitimate "no marks visible" reading from the
    # model, not a guardrail-forced manual_review. The pipeline still routes to
    # MANUAL_REVIEW (via the <2 points rule), but the extraction itself does not
    # carry the manual_review warning.
    assert result.points == []
    assert "manual_review" not in result.warnings
    assert "gemma_extracted_no_marks" in result.warnings


def test_extract_from_image_subprocess_failure_routes_manual_review(tmp_path: Path) -> None:
    fake_crop = tmp_path / "crop.png"
    _ = fake_crop.write_bytes(b"\x89PNG\r\n\x1a\n")
    fake_litert = tmp_path / "litert-lm"
    _ = fake_litert.write_text("#!/usr/bin/env python3\nraise SystemExit(2)\n")
    fake_litert.chmod(0o755)

    result = LiteRTGemmaE2BVerifier(
        litert_bin=fake_litert,
        timeout_seconds=10.0,
        hf_home=tmp_path / "hf",
    ).extract_from_image(fake_crop)

    assert result.points == []
    assert "manual_review" in result.warnings
    assert "gemma_e2b_extract_failed" in result.warnings


def test_extract_from_image_unavailable_binary_routes_manual_review(tmp_path: Path) -> None:
    fake_crop = tmp_path / "crop.png"
    _ = fake_crop.write_bytes(b"\x89PNG\r\n\x1a\n")

    result = LiteRTGemmaE2BVerifier(
        litert_bin=tmp_path / "missing-litert-lm",
        timeout_seconds=0.1,
        hf_home=tmp_path / "hf",
    ).extract_from_image(fake_crop)

    assert result.points == []
    assert "manual_review" in result.warnings
    assert any(w.startswith("gemma_e2b_extract_unavailable") for w in result.warnings)


# ---------------------------------------------------------------------------
# LiteRTGemmaDaemonExtractor.extract_from_image tests (mocked engine).
# These verify the success path, all failure paths, and daemon cache reuse
# without loading a real Gemma model.
# ---------------------------------------------------------------------------


class _FakeConversation:
    def __init__(self, response):
        self._response = response
        self.closed = False
        self.received = None

    def send_message(self, message):
        self.received = message
        if isinstance(self._response, Exception):
            raise self._response
        return self._response

    def close(self):
        self.closed = True


class _FakeEngine:
    def __init__(self, response):
        self._response = response
        self.last_conv: _FakeConversation | None = None

    def create_conversation(self, sampler_config=None):  # noqa: ARG002
        conv = _FakeConversation(self._response)
        self.last_conv = conv
        return conv


def _daemon_response(text: str) -> dict[str, object]:
    return {"content": [{"type": "text", "text": text}]}


def _patch_engine(monkeypatch, response):
    from partoguard.core.extraction import gemma_adapter as ga
    engine = _FakeEngine(response)
    monkeypatch.setattr(ga, "_get_daemon_engine", lambda *a, **k: engine)
    # Stub the litert_lm import so SamplerConfig is available
    import types
    fake_litert = types.SimpleNamespace(
        SamplerConfig=lambda top_k=1, temperature=0.0: object(),
    )
    monkeypatch.setitem(sys.modules, "litert_lm", fake_litert)
    return engine


def test_daemon_extractor_success_parses_points(monkeypatch, tmp_path):
    from partoguard.core.extraction.gemma_adapter import LiteRTGemmaDaemonExtractor

    crop = tmp_path / "crop.png"
    crop.write_bytes(b"fakepng")
    response_text = (
        '{"points": [{"x_hours": 4.0, "dilation_cm": 5.0, "confidence": 0.9}, '
        '{"x_hours": 6.0, "dilation_cm": 7.0, "confidence": 0.85}]}'
    )
    engine = _patch_engine(monkeypatch, _daemon_response(response_text))

    result = LiteRTGemmaDaemonExtractor().extract_from_image(crop)

    assert len(result.points) == 2
    assert result.points[0].x_hours == 4.0
    assert result.points[0].dilation_cm == 5.0
    assert "gemma_daemon_extracted" in result.warnings
    assert "manual_review" not in result.warnings
    # Confirm the conversation was actually used and closed
    assert engine.last_conv is not None
    assert engine.last_conv.closed
    # Confirm the image path was sent
    received = engine.last_conv.received
    assert received is not None
    assert any(
        isinstance(item, dict) and item.get("type") == "image" and item.get("path") == str(crop.resolve())
        for item in received["content"]
    )


def test_daemon_extractor_missing_crop_returns_manual_review(monkeypatch, tmp_path):
    from partoguard.core.extraction.gemma_adapter import LiteRTGemmaDaemonExtractor

    # Do NOT patch engine — must short-circuit before engine acquisition
    missing = tmp_path / "does_not_exist.png"

    result = LiteRTGemmaDaemonExtractor().extract_from_image(missing)

    assert result.points == []
    assert "manual_review" in result.warnings
    assert "gemma_daemon_extract_missing_crop" in result.warnings


def test_daemon_extractor_engine_acquisition_failure_returns_manual_review(monkeypatch, tmp_path):
    from partoguard.core.extraction import gemma_adapter as ga
    from partoguard.core.extraction.gemma_adapter import LiteRTGemmaDaemonExtractor

    crop = tmp_path / "crop.png"
    crop.write_bytes(b"fakepng")

    def _boom(*_a, **_k):
        raise FileNotFoundError("model snapshot missing")

    monkeypatch.setattr(ga, "_get_daemon_engine", _boom)

    result = LiteRTGemmaDaemonExtractor().extract_from_image(crop)

    assert result.points == []
    assert "manual_review" in result.warnings
    assert any(w.startswith("gemma_daemon_unavailable:FileNotFoundError") for w in result.warnings)


def test_daemon_extractor_invocation_failure_returns_manual_review(monkeypatch, tmp_path):
    from partoguard.core.extraction.gemma_adapter import LiteRTGemmaDaemonExtractor

    crop = tmp_path / "crop.png"
    crop.write_bytes(b"fakepng")
    _patch_engine(monkeypatch, RuntimeError("inference exploded"))

    result = LiteRTGemmaDaemonExtractor().extract_from_image(crop)

    assert result.points == []
    assert "manual_review" in result.warnings
    assert any(w.startswith("gemma_daemon_invocation_failed:RuntimeError") for w in result.warnings)


def test_daemon_extractor_invalid_json_returns_manual_review(monkeypatch, tmp_path):
    from partoguard.core.extraction.gemma_adapter import LiteRTGemmaDaemonExtractor

    crop = tmp_path / "crop.png"
    crop.write_bytes(b"fakepng")
    _patch_engine(monkeypatch, _daemon_response("not json at all {{{"))

    result = LiteRTGemmaDaemonExtractor().extract_from_image(crop)

    assert result.points == []
    assert "manual_review" in result.warnings
    assert "gemma_daemon_invalid_json" in result.warnings


def test_daemon_extractor_no_points_returns_clean_empty(monkeypatch, tmp_path):
    from partoguard.core.extraction.gemma_adapter import LiteRTGemmaDaemonExtractor

    crop = tmp_path / "crop.png"
    crop.write_bytes(b"fakepng")
    _patch_engine(monkeypatch, _daemon_response('{"points": []}'))

    result = LiteRTGemmaDaemonExtractor().extract_from_image(crop)

    assert result.points == []
    # No-marks is a legitimate blank-chart outcome, NOT manual_review.
    assert "manual_review" not in result.warnings
    assert "gemma_extracted_no_marks" in result.warnings
