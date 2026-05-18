from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from partoguard.core.schemas.contracts import DilationPoint, ExtractionResult, TemplateID


class GemmaVerifier(Protocol):
    def verify(self, extraction: ExtractionResult, chart_crop_path: Path | None = None) -> ExtractionResult:
        ...


class GemmaImageExtractor(Protocol):
    """Verifier that can also extract chart points directly from a normalized chart crop.

    When the pipeline detects this capability it bypasses CV-based mark detection
    and lets Gemma read the chart image. CV remains responsible for normalization
    (quality gating, perspective correction, registration), and the deterministic
    rule engine remains the sole clinical decision-maker.
    """

    def extract_from_image(self, chart_crop_path: Path) -> ExtractionResult:
        ...


@dataclass(frozen=True)
class StubVerifier:
    def verify(self, extraction: ExtractionResult, chart_crop_path: Path | None = None) -> ExtractionResult:
        return extraction


@dataclass(frozen=True)
class LocalGemmaVerifier:
    command: list[str]
    timeout_seconds: float = 30.0

    def verify(self, extraction: ExtractionResult, chart_crop_path: Path | None = None) -> ExtractionResult:
        if not extraction.points:
            return extraction
        try:
            completed = subprocess.run(
                self.command,
                input=_build_prompt(extraction, chart_crop_path),
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return _manual_review_from(extraction, f"gemma_verifier_unavailable:{exc.__class__.__name__}")

        if completed.returncode != 0:
            return _manual_review_from(extraction, "gemma_verifier_failed")

        try:
            payload = json.loads(completed.stdout)
            points = _bounded_points_from_payload(payload, extraction.points)
        except (json.JSONDecodeError, TypeError, ValueError):
            return _manual_review_from(extraction, "gemma_verifier_invalid_json")

        if not points:
            return _manual_review_from(extraction, "gemma_verifier_returned_no_points")

        confidence = min(point.confidence for point in points)
        return ExtractionResult(
            template_id=extraction.template_id,
            chart_present=extraction.chart_present,
            registered=extraction.registered,
            points=points,
            overall_confidence=min(extraction.overall_confidence, confidence),
            warnings=[*extraction.warnings, "gemma_verified"],
        )


@dataclass(frozen=True)
class LiteRTGemmaE2BVerifier:
    litert_bin: Path = Path("/root/partoguard-gemma/local/bin/litert-lm")
    model_reference: str = "gemma-4-E2B-it.litertlm"
    huggingface_repo: str = "litert-community/gemma-4-E2B-it-litert-lm"
    pythonpath: Path = Path("/root/partoguard-gemma/local/lib/python3.11/dist-packages")
    hf_home: Path = Path("/root/partoguard-gemma/hf")
    timeout_seconds: float = 300.0

    def verify(self, extraction: ExtractionResult, chart_crop_path: Path | None = None) -> ExtractionResult:
        if not extraction.points:
            return extraction

        attach_image = chart_crop_path is not None and chart_crop_path.exists() and chart_crop_path.is_file()
        command_args: list[str] = [
            str(self.litert_bin),
            "run",
            f"--from-huggingface-repo={self.huggingface_repo}",
            self.model_reference,
            "--prompt",
            _build_litert_prompt(extraction, with_image=attach_image),
            "--temperature",
            "0.0",
            "--top-k",
            "1",
            "--backend",
            "cpu",
        ]
        if attach_image:
            assert chart_crop_path is not None
            command_args.extend([
                "--attachment",
                str(chart_crop_path.resolve()),
                "--vision-backend",
                "cpu",
            ])

        try:
            completed = subprocess.run(
                command_args,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
                env=_litert_env(self.pythonpath, self.hf_home),
                cwd=str(self.hf_home.parent),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return _manual_review_from(extraction, f"gemma_e2b_unavailable:{exc.__class__.__name__}")

        if completed.returncode != 0:
            return _manual_review_from(extraction, "gemma_e2b_failed")

        try:
            payload = _json_payload_from_text(completed.stdout)
            points = _bounded_points_from_payload(payload, extraction.points)
        except (json.JSONDecodeError, TypeError, ValueError):
            return _manual_review_from(extraction, "gemma_e2b_invalid_json")

        if not points:
            return _manual_review_from(extraction, "gemma_e2b_returned_no_points")

        confidence = min(point.confidence for point in points)
        return ExtractionResult(
            template_id=extraction.template_id,
            chart_present=extraction.chart_present,
            registered=extraction.registered,
            points=points,
            overall_confidence=min(extraction.overall_confidence, confidence),
            warnings=[*extraction.warnings, "gemma_e2b_verified"],
        )

    def extract_from_image(self, chart_crop_path: Path) -> ExtractionResult:
        """Read the normalized chart crop with Gemma and return structured X-mark points.

        On any failure (process error, parse error, schema violation, implausible
        trajectory, out-of-range coordinates), returns a manual_review extraction
        with no points so the deterministic rule engine routes to MANUAL_REVIEW.
        """
        baseline = ExtractionResult(
            template_id=TemplateID.MODIFIED_WHO_V1,
            chart_present=True,
            registered=True,
            points=[],
            overall_confidence=0.0,
            warnings=[],
        )

        if not chart_crop_path.exists() or not chart_crop_path.is_file():
            return _manual_review_from(baseline, "gemma_e2b_extract_missing_crop")

        command_args: list[str] = [
            str(self.litert_bin),
            "run",
            f"--from-huggingface-repo={self.huggingface_repo}",
            self.model_reference,
            "--prompt",
            _build_litert_extraction_prompt(),
            "--temperature",
            "0.0",
            "--top-k",
            "1",
            "--backend",
            "cpu",
            "--attachment",
            str(chart_crop_path.resolve()),
            "--vision-backend",
            "cpu",
        ]

        try:
            completed = subprocess.run(
                command_args,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
                env=_litert_env(self.pythonpath, self.hf_home),
                cwd=str(self.hf_home.parent),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return _manual_review_from(baseline, f"gemma_e2b_extract_unavailable:{exc.__class__.__name__}")

        if completed.returncode != 0:
            return _manual_review_from(baseline, "gemma_e2b_extract_failed")

        try:
            payload = _json_payload_from_text(completed.stdout)
            payload = _normalize_extraction_payload(payload)
            points = _phase1_points_from_payload(payload)
        except (json.JSONDecodeError, TypeError, ValueError):
            return _manual_review_from(baseline, "gemma_e2b_extract_invalid_json")

        if not points:
            return ExtractionResult(
                template_id=TemplateID.MODIFIED_WHO_V1,
                chart_present=True,
                registered=True,
                points=[],
                overall_confidence=0.0,
                warnings=["gemma_extracted_no_marks"],
            )

        confidence = min(point.confidence for point in points)
        return ExtractionResult(
            template_id=TemplateID.MODIFIED_WHO_V1,
            chart_present=True,
            registered=True,
            points=points,
            overall_confidence=confidence,
            warnings=["gemma_e2b_extracted"],
        )


_REMOTE_DEFAULT_BASE_URL = "http://localhost:8080"
_REMOTE_PROPS_PATH = "/props"
_REMOTE_COMPLETION_PATH = "/completion"

_REMOTE_MARKER_CACHE: dict[str, str] = {}
_REMOTE_MARKER_CACHE_LOCK = threading.Lock()


def _remote_fetch_media_marker(base_url: str, timeout: float) -> str:
    import requests  # noqa: PLC0415

    response = requests.get(base_url.rstrip("/") + _REMOTE_PROPS_PATH, timeout=timeout)
    response.raise_for_status()
    marker = response.json().get("media_marker")
    if not isinstance(marker, str) or not marker:
        raise RuntimeError("llama-server /props did not return a media_marker")
    return marker


def _remote_cached_media_marker(base_url: str, timeout: float) -> str:
    key = base_url.rstrip("/")
    with _REMOTE_MARKER_CACHE_LOCK:
        cached = _REMOTE_MARKER_CACHE.get(key)
        if cached is not None:
            return cached
        marker = _remote_fetch_media_marker(base_url, timeout)
        _REMOTE_MARKER_CACHE[key] = marker
        return marker
def _remote_invalidate_marker_cache(base_url: str | None = None) -> None:
    with _REMOTE_MARKER_CACHE_LOCK:
        if base_url is None:
            _REMOTE_MARKER_CACHE.clear()
        else:
            _REMOTE_MARKER_CACHE.pop(base_url.rstrip("/"), None)


def _build_remote_extraction_prompt() -> str:
    return (
        "Find every handwritten X mark inside the cervicograph plot of this WHO "
        "partograph. The cervicograph is the grid where cervical dilation 0-10 cm "
        "is on the y-axis and hours 0-12 is on the x-axis, with two diagonal "
        "Alert and Action lines crossing it. An X mark is two short pen strokes "
        "crossing at one point.\n\n"
        "Ignore: printed grid lines, the diagonal Alert/Action lines, axis labels, "
        "contraction shading, fetal-heart-rate dots, and handwriting outside the "
        "cervicograph.\n\n"
        "For each X mark return [x_hours, dilation_cm, confidence] where x_hours is "
        "integer 0-12, dilation_cm is half-integer 0.0-10.0 (in 0.5 increments), "
        "confidence is 0.0-1.0.\n\n"
        "If the cervicograph is blank (no X marks), return {\"p\":[]}.\n\n"
        "Return strictly compact JSON in this exact schema, no markdown fences and "
        "no commentary:\n"
        "{\"p\":[[x_hours, dilation_cm, confidence], ...]}"
    )


def _build_remote_verify_prompt(extraction: ExtractionResult) -> str:
    candidates = [
        {
            "x_hours": round(float(p.x_hours), 2),
            "dilation_cm": round(float(p.dilation_cm), 2),
            "confidence": round(float(p.confidence), 2),
        }
        for p in extraction.points
    ]
    return (
        "You are PartoGuard's bounded verifier. Look at the partograph image and "
        "decide which of these candidate X marks are real handwritten marks inside "
        "the cervicograph plot. Do not invent new marks; do not change coordinates "
        "more than necessary; round x_hours to integer 0-12 and dilation_cm to "
        "0.5 increments 0.0-10.0.\n\n"
        f"Candidates: {json.dumps(candidates, separators=(',', ':'))}\n\n"
        'Return strictly compact JSON, no markdown fences:\n'
        '{"accepted_points":[{"x_hours":0.0,"dilation_cm":0.0,"confidence":0.0}, ...]}'
    )


@dataclass(frozen=True)
class RemoteGemmaExtractor:
    """HTTP-based remote extractor for llama.cpp ``llama-server``.

    Uses the canonical ``/completion`` multimodal request format
    (``prompt`` is an object with ``prompt_string`` + ``multimodal_data``)
    against the server-advertised ``media_marker`` token from ``/props``.
    """

    url: str = _REMOTE_DEFAULT_BASE_URL + _REMOTE_COMPLETION_PATH
    timeout_seconds: float = 180.0

    def _base_url(self) -> str:
        url = self.url
        for suffix in (_REMOTE_COMPLETION_PATH, "/v1/chat/completions", "/v1/completions"):
            if url.endswith(suffix):
                return url[: -len(suffix)]
        return url.rstrip("/")

    def _post_completion(self, image_bytes: bytes, prompt_body: str, n_predict: int = 400) -> str:
        import base64  # noqa: PLC0415
        import requests  # noqa: PLC0415

        base = self._base_url()
        marker = _remote_cached_media_marker(base, timeout=min(self.timeout_seconds, 15.0))
        prompt_string = (
            f"<|turn>user\n{marker}\n{prompt_body}<turn|>\n<|turn>model\n"
        )
        payload: dict[str, Any] = {
            "prompt": {
                "prompt_string": prompt_string,
                "multimodal_data": [base64.b64encode(image_bytes).decode("utf-8")],
            },
            "n_predict": n_predict,
            "temperature": 0.0,
            "top_k": 1,
            "cache_prompt": False,
        }
        response = requests.post(
            base + _REMOTE_COMPLETION_PATH,
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json().get("content", "")

    def verify(
        self,
        extraction: ExtractionResult,
        chart_crop_path: Path | None = None,
    ) -> ExtractionResult:
        if not extraction.points:
            return extraction

        if chart_crop_path is None or not chart_crop_path.exists() or not chart_crop_path.is_file():
            return _manual_review_from(extraction, "gemma_remote_verify_missing_crop")

        try:
            text = self._post_completion(
                chart_crop_path.read_bytes(),
                _build_remote_verify_prompt(extraction),
            )
        except Exception as exc:
            return _manual_review_from(extraction, f"gemma_remote_unavailable:{exc.__class__.__name__}")

        try:
            json_payload = _json_payload_from_text(text)
            points = _bounded_points_from_payload(json_payload, extraction.points)
        except (json.JSONDecodeError, TypeError, ValueError):
            return _manual_review_from(extraction, "gemma_remote_invalid_json")

        if not points:
            return _manual_review_from(extraction, "gemma_remote_returned_no_points")

        confidence = min(point.confidence for point in points)
        return ExtractionResult(
            template_id=extraction.template_id,
            chart_present=extraction.chart_present,
            registered=extraction.registered,
            points=points,
            overall_confidence=min(extraction.overall_confidence, confidence),
            warnings=[*extraction.warnings, "gemma_remote_verified"],
        )

    def extract_from_image(self, chart_crop_path: Path) -> ExtractionResult:
        baseline = ExtractionResult(
            template_id=TemplateID.MODIFIED_WHO_V1,
            chart_present=True,
            registered=True,
            points=[],
            overall_confidence=0.0,
            warnings=[],
        )

        if not chart_crop_path.exists() or not chart_crop_path.is_file():
            return _manual_review_from(baseline, "gemma_remote_extract_missing_crop")

        try:
            text = self._post_completion(
                chart_crop_path.read_bytes(),
                _build_remote_extraction_prompt(),
            )
        except Exception as exc:
            return _manual_review_from(baseline, f"gemma_remote_extract_unavailable:{exc.__class__.__name__}")

        try:
            json_payload = _json_payload_from_text(text)
            json_payload = _normalize_extraction_payload(json_payload)
            points = _phase1_points_from_payload(json_payload)
        except (json.JSONDecodeError, TypeError, ValueError):
            return _manual_review_from(baseline, "gemma_remote_extract_invalid_json")

        if not points:
            return ExtractionResult(
                template_id=TemplateID.MODIFIED_WHO_V1,
                chart_present=True,
                registered=True,
                points=[],
                overall_confidence=0.0,
                warnings=["gemma_extracted_no_marks"],
            )

        confidence = min(point.confidence for point in points)
        return ExtractionResult(
            template_id=TemplateID.MODIFIED_WHO_V1,
            chart_present=True,
            registered=True,
            points=points,
            overall_confidence=confidence,
            warnings=["gemma_remote_extracted"],
        )


def build_verifier(
    command: str | None,
    *,
    use_litert_e2b: bool = False,
    use_litert_e4b: bool = False,
    use_xpu_e2b: bool = False,
    use_xpu_e4b: bool = False,
    use_xpu_e2b_ft: bool = False,
    use_cuda_e2b_ft: bool = False,
    use_daemon: bool = True,
    use_remote: bool = False,
    remote_url: str | None = None,
    litert_bin: Path | None = None,
    litert_model_path: Path | None = None,
) -> GemmaVerifier:
    if use_remote:
        return RemoteGemmaExtractor(url=remote_url or "http://localhost:8080/completion")
    if use_cuda_e2b_ft:
        if use_xpu_e2b or use_xpu_e4b or use_xpu_e2b_ft:
            raise ValueError("Choose only one --gemma-xpu-* or --gemma-cuda-* variant")
        if command is not None and command.strip() != "":
            raise ValueError("Choose either --gemma-command or --gemma-cuda-*, not both")
        if use_litert_e2b or use_litert_e4b:
            raise ValueError("Choose either --gemma-cuda-* or --gemma-litert-*, not both")
        return CudaGemmaExtractor(
            base_model_id="google/gemma-4-E2B-it",
            adapter_path=os.environ.get(
                "PARTOGUARD_ADAPTER_PATH",
                "/root/partoguard-lora/lora_adapter_v7",
            ),
            preprocess=os.environ.get("PARTOGUARD_PREPROCESS"),
        )
    if use_xpu_e2b_ft:
        if use_xpu_e2b or use_xpu_e4b:
            raise ValueError("Choose only one --gemma-xpu-* variant")
        if command is not None and command.strip() != "":
            raise ValueError("Choose either --gemma-command or --gemma-xpu-*, not both")
        if use_litert_e2b or use_litert_e4b:
            raise ValueError("Choose either --gemma-xpu-* or --gemma-litert-*, not both")
        return XpuGemmaExtractor(model_id="/root/partoguard-lora/merged")
    if use_xpu_e2b or use_xpu_e4b:
        if use_xpu_e2b and use_xpu_e4b:
            raise ValueError("Choose either --gemma-xpu-e2b or --gemma-xpu-e4b, not both")
        if command is not None and command.strip() != "":
            raise ValueError("Choose either --gemma-command or --gemma-xpu-*, not both")
        if use_litert_e2b or use_litert_e4b:
            raise ValueError("Choose either --gemma-xpu-* or --gemma-litert-*, not both")
        if use_xpu_e4b:
            return XpuGemmaExtractor(
                model_id="unsloth/gemma-4-E4B-it-unsloth-bnb-4bit",
                quantize_4bit=True,
            )
        return XpuGemmaExtractor()
    if use_litert_e2b and use_litert_e4b:
        raise ValueError("Choose either --gemma-litert-e2b or --gemma-litert-e4b, not both")
    if use_litert_e2b or use_litert_e4b:
        if command is not None and command.strip() != "":
            raise ValueError("Choose either --gemma-command or --gemma-litert-*, not both")
        bin_path = litert_bin or LiteRTGemmaE2BVerifier.litert_bin
        if use_daemon:
            if litert_model_path is not None:
                return LiteRTGemmaDaemonExtractor(direct_model_path=litert_model_path)
            if use_litert_e4b:
                return LiteRTGemmaDaemonExtractor(
                    model_reference="gemma-4-E4B-it.litertlm",
                    huggingface_repo="litert-community/gemma-4-E4B-it-litert-lm",
                )
            return LiteRTGemmaDaemonExtractor()
        if use_litert_e4b:
            return LiteRTGemmaE2BVerifier(
                litert_bin=bin_path,
                model_reference="gemma-4-E4B-it.litertlm",
                huggingface_repo="litert-community/gemma-4-E4B-it-litert-lm",
            )
        return LiteRTGemmaE2BVerifier(litert_bin=bin_path)
    if command is None or command.strip() == "":
        return StubVerifier()
    import shlex

    return LocalGemmaVerifier(command=shlex.split(command))


_DAEMON_ENGINE_LOCK = threading.Lock()
_DAEMON_ENGINES: dict[str, Any] = {}


def _get_daemon_engine_direct(
    model_path: Path,
    pythonpath: Path,
    hf_home: Path,
) -> Any:
    """Load a LiteRT-LM engine from a direct file path (not HF cache)."""
    cache_key = f"direct::{model_path}"
    with _DAEMON_ENGINE_LOCK:
        cached = _DAEMON_ENGINES.get(cache_key)
        if cached is not None:
            return cached
        if str(pythonpath) not in sys.path:
            sys.path.insert(0, str(pythonpath))
        os.environ.setdefault("HF_HOME", str(hf_home))
        import litert_lm  # noqa: PLC0415

        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
        engine = litert_lm.Engine(
            model_path=str(model_path),
            backend=litert_lm.Backend.CPU,
            vision_backend=litert_lm.Backend.CPU,
        )
        _DAEMON_ENGINES[cache_key] = engine
        return engine


def _litert_generate_with_image(
    engine: Any,
    image_path: Path,
    prompt_text: str,
) -> str:
    """Call LiteRT C API directly to do vision inference, bypassing Jinja."""
    import ctypes  # noqa: PLC0415

    from litert_lm._ffi import InputDataType, LiteRtLmInputData  # noqa: PLC0415

    session = engine.create_session(apply_prompt_template=False)
    try:
        lib = session._lib  # noqa: SLF001
        ptr = session._ptr  # noqa: SLF001

        img_bytes = image_path.resolve().read_bytes()
        img_buf = (ctypes.c_char * len(img_bytes))(*img_bytes)

        raw_prompt = (
            "<bos><|turn>user\n"
            "<|image|>"
            f"{prompt_text}"
            "<turn|>\n<|turn>model\n"
        )
        text_bytes = raw_prompt.encode("utf-8")
        txt_buf = (ctypes.c_char * len(text_bytes))(*text_bytes)

        inputs = (LiteRtLmInputData * 3)()
        inputs[0].type = int(InputDataType.IMAGE)
        inputs[0].data = ctypes.cast(img_buf, ctypes.c_void_p)
        inputs[0].size = len(img_bytes)
        inputs[1].type = int(InputDataType.IMAGE_END)
        inputs[1].data = None
        inputs[1].size = 0
        inputs[2].type = int(InputDataType.TEXT)
        inputs[2].data = ctypes.cast(txt_buf, ctypes.c_void_p)
        inputs[2].size = len(text_bytes)

        resp_ptr = lib.litert_lm_session_generate_content(ptr, inputs, 3)
        if not resp_ptr:
            raise RuntimeError("litert_lm_session_generate_content returned null")

        try:
            num = lib.litert_lm_responses_get_num_candidates(resp_ptr)
            for i in range(num):
                t = lib.litert_lm_responses_get_response_text_at(resp_ptr, i)
                if t:
                    return t.decode("utf-8")
            return ""
        finally:
            lib.litert_lm_responses_delete(resp_ptr)
    finally:
        session.close()


def _resolve_daemon_model_path(huggingface_repo: str, model_reference: str) -> Path:
    repo_slug = huggingface_repo.replace("/", "--")
    snapshots = Path("/root/partoguard-gemma/hf/hub") / f"models--{repo_slug}" / "snapshots"
    if not snapshots.is_dir():
        raise FileNotFoundError(f"Daemon model snapshots dir not found: {snapshots}")
    for snap in snapshots.iterdir():
        candidate = snap / model_reference
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Daemon model file {model_reference} not found under {snapshots}"
    )


def _get_daemon_engine(
    huggingface_repo: str,
    model_reference: str,
    pythonpath: Path,
    hf_home: Path,
) -> Any:
    cache_key = f"{huggingface_repo}::{model_reference}"
    with _DAEMON_ENGINE_LOCK:
        cached = _DAEMON_ENGINES.get(cache_key)
        if cached is not None:
            return cached
        if str(pythonpath) not in sys.path:
            sys.path.insert(0, str(pythonpath))
        os.environ.setdefault("HF_HOME", str(hf_home))
        import litert_lm  # noqa: PLC0415

        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
        model_path = _resolve_daemon_model_path(huggingface_repo, model_reference)
        engine = litert_lm.Engine(
            model_path=str(model_path),
            backend=litert_lm.Backend.CPU,
            vision_backend=litert_lm.Backend.CPU,
        )
        _DAEMON_ENGINES[cache_key] = engine
        return engine


@dataclass(frozen=True)
class LiteRTGemmaDaemonExtractor:
    """Phase-1 image extractor using the LiteRT-LM Python API as a resident daemon.

    Loads the Gemma model into memory once per process (cached at module level)
    and reuses it across every extract_from_image call. Eliminates subprocess
    boot + model load overhead, reducing latency from ~5.5s/image (subprocess)
    to ~4.5s/image (E2B daemon) or ~8.5s/image (E4B daemon).

    On any failure the extractor returns a manual_review extraction so the
    deterministic rule engine routes to MANUAL_REVIEW.
    """

    model_reference: str = "gemma-4-E2B-it.litertlm"
    huggingface_repo: str = "litert-community/gemma-4-E2B-it-litert-lm"
    pythonpath: Path = Path("/root/partoguard-gemma/local/lib/python3.11/dist-packages")
    hf_home: Path = Path("/root/partoguard-gemma/hf")
    direct_model_path: Path | None = None

    def verify(self, extraction: ExtractionResult, chart_crop_path: Path | None = None) -> ExtractionResult:
        # Phase 1: verifier role bypassed (pipeline uses extract_from_image instead).
        return extraction

    def extract_from_image(self, chart_crop_path: Path) -> ExtractionResult:
        baseline = ExtractionResult(
            template_id=TemplateID.MODIFIED_WHO_V1,
            chart_present=True,
            registered=True,
            points=[],
            overall_confidence=0.0,
            warnings=[],
        )

        if not chart_crop_path.exists() or not chart_crop_path.is_file():
            return _manual_review_from(baseline, "gemma_daemon_extract_missing_crop")

        try:
            if self.direct_model_path is not None:
                engine = _get_daemon_engine_direct(
                    self.direct_model_path,
                    self.pythonpath,
                    self.hf_home,
                )
            else:
                engine = _get_daemon_engine(
                    self.huggingface_repo,
                    self.model_reference,
                    self.pythonpath,
                    self.hf_home,
                )
        except (FileNotFoundError, ImportError, OSError) as exc:
            return _manual_review_from(
                baseline, f"gemma_daemon_unavailable:{exc.__class__.__name__}"
            )

        try:
            import litert_lm  # noqa: PLC0415

            sampler = litert_lm.SamplerConfig(top_k=1, temperature=0.0)
            text = ""
            try:
                conv = engine.create_conversation(sampler_config=sampler)
                try:
                    response = conv.send_message({
                        "role": "user",
                        "content": [
                            {"type": "image", "path": str(chart_crop_path.resolve())},
                            {"type": "text", "text": _build_litert_extraction_prompt()},
                        ],
                    })
                finally:
                    conv.close()
                for item in response.get("content", []) if isinstance(response, dict) else []:
                    if isinstance(item, dict) and item.get("type") == "text":
                        value = item.get("text")
                        if isinstance(value, str):
                            text += value
            except RuntimeError as conv_err:
                # Conversation API failed (Jinja template issue with Gemma 4
                # .get() method unsupported by LiteRT Minijinja runtime).
                # Bypass Python Session wrapper and call C API directly which
                # supports IMAGE inputs via LiteRtLmInputData structs.
                try:
                    text = _litert_generate_with_image(
                        engine,
                        chart_crop_path,
                        _build_litert_extraction_prompt(),
                    )
                except Exception:
                    raise conv_err from None
        except Exception as exc:  # noqa: BLE001 - daemon library may raise arbitrary types
            return _manual_review_from(
                baseline, f"gemma_daemon_invocation_failed:{exc.__class__.__name__}"
            )

        try:
            payload = _json_payload_from_text(text)
            payload = _normalize_extraction_payload(payload)
            points = _phase1_points_from_payload(payload)
        except (json.JSONDecodeError, TypeError, ValueError):
            return _manual_review_from(baseline, "gemma_daemon_invalid_json")

        if not points:
            return ExtractionResult(
                template_id=TemplateID.MODIFIED_WHO_V1,
                chart_present=True,
                registered=True,
                points=[],
                overall_confidence=0.0,
                warnings=["gemma_extracted_no_marks"],
            )

        confidence = min(point.confidence for point in points)
        return ExtractionResult(
            template_id=TemplateID.MODIFIED_WHO_V1,
            chart_present=True,
            registered=True,
            points=points,
            overall_confidence=confidence,
            warnings=["gemma_daemon_extracted"],
        )


_XPU_MODEL_LOCK = threading.Lock()
_XPU_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}

_CUDA_MODEL_LOCK = threading.Lock()
_CUDA_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


def _get_xpu_model(model_id: str, *, quantize_4bit: bool = False) -> tuple[Any, Any]:
    cache_key = f"{model_id}::4bit" if quantize_4bit else model_id
    with _XPU_MODEL_LOCK:
        cached = _XPU_MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        import torch  # noqa: PLC0415
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

        processor = AutoProcessor.from_pretrained(model_id)

        if quantize_4bit:
            model = _load_xpu_4bit_model(model_id)
        else:
            model: Any = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16)  # type: ignore[reportAttributeAccessIssue]
            model = model.to("xpu").eval()  # type: ignore[reportCallIssue]

        # Warmup: first generate() triggers XPU kernel JIT compilation
        dummy: Any = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True,
        ).to("xpu")
        with torch.inference_mode():
            _ = model.generate(**dummy, max_new_tokens=10)  # type: ignore[reportCallIssue, reportAttributeAccessIssue]
        torch.xpu.synchronize()

        _XPU_MODEL_CACHE[cache_key] = (model, processor)
        return model, processor


def _get_cuda_model(base_model_id: str, adapter_path: str | None = None) -> tuple[Any, Any]:
    cache_key = f"{base_model_id}::{adapter_path or 'none'}"
    with _CUDA_MODEL_LOCK:
        cached = _CUDA_MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        import torch  # noqa: PLC0415
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig  # noqa: PLC0415

        processor = AutoProcessor.from_pretrained(
            adapter_path if adapter_path else base_model_id,
        )

        use_bf16 = os.environ.get("PARTOGUARD_BF16", "") == "1"

        if use_bf16:
            model: Any = AutoModelForImageTextToText.from_pretrained(
                base_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            model: Any = AutoModelForImageTextToText.from_pretrained(
                base_model_id,
                quantization_config=bnb_config,
                device_map="auto",
                max_memory={0: "7GiB", "cpu": "24GiB"},
            )

        if adapter_path:
            from peft import PeftModel  # noqa: PLC0415
            model = PeftModel.from_pretrained(model, adapter_path)

        model.eval()
        _CUDA_MODEL_CACHE[cache_key] = (model, processor)
        return model, processor


def _load_xpu_4bit_model(model_id: str) -> Any:
    """Load a pre-quantized 4-bit model on XPU with vision tower bf16 replacement.

    bitsandbytes on XPU cannot dequantize the small vision tower linear layers
    (missing quant_state). We work around this by loading the model with bnb 4-bit
    config, then replacing vision tower Linear4bit layers with bf16 Linear layers
    loaded from the corresponding full-precision safetensors.
    """
    import gc  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig  # noqa: PLC0415

    import bitsandbytes as bnb  # noqa: PLC0415
    import bitsandbytes.nn.modules as bnb_mod  # noqa: PLC0415

    # Patch bnb assertion that crashes on XPU for layers without quant_state
    _orig_fix = bnb_mod.fix_4bit_weight_quant_state_from_module

    def _patched_fix(module: Any) -> None:
        if getattr(module.weight, "quant_state", None) is not None:
            return
        if getattr(module, "quant_state", None) is None:
            return
        if module.weight.shape[1] != 1:
            return
        if not isinstance(module.weight, bnb_mod.Params4bit):
            _params4bit: Any = bnb_mod.Params4bit
            module.weight = _params4bit(
                module.weight, quant_storage=module.quant_storage, bnb_quantized=True,
            )
        module.weight.quant_state = module.quant_state

    bnb_mod.fix_4bit_weight_quant_state_from_module = _patched_fix

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",  # type: ignore[reportAttributeAccessIssue]
    )
    model: Any = AutoModelForImageTextToText.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="xpu",
    )

    # Resolve the bf16 base model for vision tower weights
    bf16_model_id = _resolve_bf16_base_model(model_id)
    bf16_cache = _Path(os.environ.get("HF_HOME", "")) / "hub" if os.environ.get("HF_HOME") else _Path.home() / ".cache" / "huggingface" / "hub"
    repo_slug = bf16_model_id.replace("/", "--")
    snapshots = bf16_cache / f"models--{repo_slug}" / "snapshots"

    vision_tensors: dict[str, Any] = {}
    if snapshots.is_dir():
        for snap in snapshots.iterdir():
            for sf in sorted(snap.glob("*.safetensors")):
                with safe_open(str(sf), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if "vision_tower" in key or "multi_modal_projector" in key:
                            vision_tensors[key] = f.get_tensor(key)
            if vision_tensors:
                break

    if vision_tensors:
        for name, module in list(model.named_modules()):
            if ("vision_tower" in name or "multi_modal_projector" in name) and isinstance(module, bnb.nn.Linear4bit):  # type: ignore[reportAttributeAccessIssue]
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                attr = parts[-1]
                weight_key = name + ".weight"
                bias_key = name + ".bias"
                if weight_key in vision_tensors:
                    new_linear = nn.Linear(
                        module.in_features, module.out_features,
                        bias=module.bias is not None, dtype=torch.bfloat16, device="cpu",  # type: ignore[reportAttributeAccessIssue]
                    )
                    new_linear.weight.data = vision_tensors[weight_key].to(torch.bfloat16)  # type: ignore[reportAttributeAccessIssue]
                    if module.bias is not None and bias_key in vision_tensors:
                        new_linear.bias.data = vision_tensors[bias_key].to(torch.bfloat16)  # type: ignore[reportAttributeAccessIssue]
                    setattr(parent, attr, new_linear.to("xpu"))

        del vision_tensors
        gc.collect()

    # Restore original bnb function
    bnb_mod.fix_4bit_weight_quant_state_from_module = _orig_fix

    return model


def _resolve_bf16_base_model(quantized_model_id: str) -> str:
    """Map a quantized model ID to its bf16 base model for vision tower weights."""
    _map: dict[str, str] = {
        "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit": "google/gemma-4-E4B-it",
        "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit": "google/gemma-4-E2B-it",
    }
    return _map.get(quantized_model_id, quantized_model_id)


@dataclass(frozen=True)
class XpuGemmaExtractor:
    model_id: str = "google/gemma-4-E2B-it"
    max_new_tokens: int = 300
    quantize_4bit: bool = False

    def verify(self, extraction: ExtractionResult, chart_crop_path: Path | None = None) -> ExtractionResult:
        return extraction

    def extract_from_image(self, chart_crop_path: Path) -> ExtractionResult:
        baseline = ExtractionResult(
            template_id=TemplateID.MODIFIED_WHO_V1,
            chart_present=True,
            registered=True,
            points=[],
            overall_confidence=0.0,
            warnings=[],
        )

        if not chart_crop_path.exists() or not chart_crop_path.is_file():
            return _manual_review_from(baseline, "gemma_xpu_extract_missing_crop")

        try:
            model, processor = _get_xpu_model(self.model_id, quantize_4bit=self.quantize_4bit)
        except Exception as exc:  # noqa: BLE001
            return _manual_review_from(
                baseline, f"gemma_xpu_unavailable:{exc.__class__.__name__}"
            )

        try:
            import torch  # noqa: PLC0415
            from PIL import Image as PILImage  # noqa: PLC0415

            img = PILImage.open(chart_crop_path).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": _build_litert_extraction_prompt()},
            ]}]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, return_dict=True,
                return_tensors="pt", add_generation_prompt=True,
            ).to("xpu")

            with torch.inference_mode():
                output = model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                )
            torch.xpu.synchronize()

            text = processor.decode(output[0], skip_special_tokens=True)
            # Extract model response after the last "model" turn marker
            parts = text.split("model")
            text = parts[-1].strip() if len(parts) > 1 else text.strip()
        except Exception as exc:  # noqa: BLE001
            return _manual_review_from(
                baseline, f"gemma_xpu_invocation_failed:{exc.__class__.__name__}"
            )

        try:
            payload = _json_payload_from_text(text)
            payload = _normalize_extraction_payload(payload)
            points = _phase1_points_from_payload(payload)
        except (json.JSONDecodeError, TypeError, ValueError):
            return _manual_review_from(baseline, "gemma_xpu_invalid_json")

        if not points:
            return ExtractionResult(
                template_id=TemplateID.MODIFIED_WHO_V1,
                chart_present=True,
                registered=True,
                points=[],
                overall_confidence=0.0,
                warnings=["gemma_extracted_no_marks"],
            )

        confidence = min(point.confidence for point in points)
        return ExtractionResult(
            template_id=TemplateID.MODIFIED_WHO_V1,
            chart_present=True,
            registered=True,
            points=points,
            overall_confidence=confidence,
            warnings=["gemma_xpu_extracted"],
        )


class CudaGemmaExtractor:
    """CUDA inference with optional LoRA adapter and TTA (CLAHE + rotation)."""

    PREPROCESS_REGISTRY: dict[str, Any] = {}

    @staticmethod
    def _init_preprocess_registry() -> dict[str, Any]:
        if CudaGemmaExtractor.PREPROCESS_REGISTRY:
            return CudaGemmaExtractor.PREPROCESS_REGISTRY
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        from PIL import Image as PILImage  # noqa: PLC0415

        def chromatic(img: Any) -> Any:
            arr = np.array(img).astype(np.float32)
            w = arr.shape[1]
            g = np.linspace(0, 1, w).reshape(1, -1)
            arr[:, :, 0] = np.clip(arr[:, :, 0] + g * 40, 0, 255)
            arr[:, :, 2] = np.clip(arr[:, :, 2] + (1 - g) * 40, 0, 255)
            return PILImage.fromarray(arr.astype(np.uint8))

        def fft_notch(img: Any) -> Any:
            arr = np.array(img)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            f = np.fft.fft2(gray.astype(np.float32))
            fshift = np.fft.fftshift(f)
            rows, cols = gray.shape
            crow, ccol = rows // 2, cols // 2
            dc = fshift[crow, ccol].copy()
            fshift[crow - 2:crow + 3, :] = 0
            fshift[:, ccol - 2:ccol + 3] = 0
            fshift[crow, ccol] = dc
            img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift)))
            img_back = np.clip(img_back, 0, 255).astype(np.uint8)
            return PILImage.fromarray(cv2.cvtColor(img_back, cv2.COLOR_GRAY2RGB))

        def clahe_lab(img: Any) -> Any:
            arr = np.array(img)
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return PILImage.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))

        def otsu(img: Any) -> Any:
            """Otsu binarization. Strips degraded paper texture so the
            quantized vision encoder doesn't hallucinate marks from noise.
            See knowledge/partoguard_int4_quant_paths.md (Phase-1 mushroom hack).
            """
            arr = np.array(img)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            # Slight blur first so paper grain doesn't survive thresholding.
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return PILImage.fromarray(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))

        def otsu_dilate(img: Any) -> Any:
            """Otsu + 1-px morphological dilation. Makes thin pen X marks
            thicker so they survive aggressive int4 vision quantization.
            """
            arr = np.array(img)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            # Invert so marks are white on black, dilate (grows white
            # regions), then re-invert back to black-on-white.
            inv = cv2.bitwise_not(binary)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            inv = cv2.dilate(inv, kernel, iterations=1)
            binary = cv2.bitwise_not(inv)
            return PILImage.fromarray(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))

        CudaGemmaExtractor.PREPROCESS_REGISTRY = {
            "chromatic": chromatic,
            "fft_notch": fft_notch,
            "clahe_lab": clahe_lab,
            "otsu": otsu,
            "otsu_dilate": otsu_dilate,
        }
        return CudaGemmaExtractor.PREPROCESS_REGISTRY

    def __init__(
        self,
        base_model_id: str = "google/gemma-4-E2B-it",
        adapter_path: str | None = None,
        max_new_tokens: int = 300,
        tta_passes: int = 1,
        preprocess: str | None = None,
    ):
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.tta_passes = max(1, tta_passes)
        self.preprocess_name = preprocess
        self._preprocess_fn: Any = None
        if preprocess:
            registry = self._init_preprocess_registry()
            if preprocess not in registry:
                raise ValueError(f"Unknown preprocess: {preprocess}. Options: {list(registry.keys())}")
            self._preprocess_fn = registry[preprocess]

    def verify(self, extraction: ExtractionResult, chart_crop_path: Path | None = None) -> ExtractionResult:
        return extraction

    def _infer_once(self, img: Any, model: Any, processor: Any) -> list[DilationPoint]:
        import torch  # noqa: PLC0415

        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": _build_litert_extraction_prompt()},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, return_dict=True,
            return_tensors="pt", add_generation_prompt=True,
        ).to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = processor.decode(output[0], skip_special_tokens=True)
        parts = text.split("model")
        text = parts[-1].strip() if len(parts) > 1 else text.strip()

        payload = _json_payload_from_text(text)
        payload = _normalize_extraction_payload(payload)
        return _phase1_points_from_payload(payload)

    @staticmethod
    def _tta_transforms(img: Any, pass_idx: int) -> tuple[Any, bool]:
        if pass_idx == 0:
            return img, False
        if pass_idx == 1:
            import numpy as np  # noqa: PLC0415

            import cv2  # noqa: PLC0415

            arr = np.array(img)
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            from PIL import Image as PILImage  # noqa: PLC0415

            return PILImage.fromarray(enhanced), False
        return img.rotate(180), True

    @staticmethod
    def _average_point_sets(all_points: list[list[DilationPoint]]) -> list[DilationPoint]:
        if not all_points:
            return []

        from collections import Counter  # noqa: PLC0415

        counts = Counter(len(pts) for pts in all_points)
        majority_count, _ = counts.most_common(1)[0]

        matching = [pts for pts in all_points if len(pts) == majority_count]
        if not matching:
            return all_points[0]

        if majority_count == 0:
            return []

        averaged: list[DilationPoint] = []
        for i in range(majority_count):
            avg_x = sum(m[i].x_hours for m in matching) / len(matching)
            avg_d = sum(m[i].dilation_cm for m in matching) / len(matching)
            avg_c = sum(m[i].confidence for m in matching) / len(matching)
            avg_x = round(avg_x * 2.0) / 2.0
            avg_d = round(avg_d * 2.0) / 2.0
            avg_x = max(0.0, min(12.0, avg_x))
            avg_d = max(0.0, min(10.0, avg_d))
            avg_c = max(0.0, min(1.0, avg_c))
            averaged.append(DilationPoint(
                x_hours=avg_x, dilation_cm=avg_d,
                confidence=avg_c, source="gemma_raw",
            ))
        return sorted(averaged, key=lambda p: (p.x_hours, p.dilation_cm))

    def extract_from_image(self, chart_crop_path: Path) -> ExtractionResult:
        baseline = ExtractionResult(
            template_id=TemplateID.MODIFIED_WHO_V1,
            chart_present=True,
            registered=True,
            points=[],
            overall_confidence=0.0,
            warnings=[],
        )
        if not chart_crop_path.exists() or not chart_crop_path.is_file():
            return _manual_review_from(baseline, "gemma_cuda_extract_missing_crop")
        try:
            model, processor = _get_cuda_model(self.base_model_id, self.adapter_path)
        except Exception as exc:  # noqa: BLE001
            return _manual_review_from(baseline, f"gemma_cuda_unavailable:{exc.__class__.__name__}")
        try:
            from PIL import Image as PILImage  # noqa: PLC0415

            img = PILImage.open(chart_crop_path).convert("RGB")
            if self._preprocess_fn is not None:
                img = self._preprocess_fn(img)

            all_pass_points: list[list[DilationPoint]] = []
            for pass_idx in range(self.tta_passes):
                augmented, is_rotated = self._tta_transforms(img, pass_idx)
                try:
                    pts = self._infer_once(augmented, model, processor)
                    if is_rotated:
                        pts = [
                            DilationPoint(
                                x_hours=round((12.0 - p.x_hours) * 2.0) / 2.0,
                                dilation_cm=round((10.0 - p.dilation_cm) * 2.0) / 2.0,
                                confidence=p.confidence,
                                source=p.source,
                            )
                            for p in pts
                        ]
                        pts.sort(key=lambda p: (p.x_hours, p.dilation_cm))
                    all_pass_points.append(pts)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue

            if not all_pass_points:
                return _manual_review_from(baseline, "gemma_cuda_invalid_json")

            if self.tta_passes == 1:
                points = all_pass_points[0]
            else:
                points = self._average_point_sets(all_pass_points)

        except Exception as exc:  # noqa: BLE001
            return _manual_review_from(baseline, f"gemma_cuda_invocation_failed:{exc.__class__.__name__}")

        if not points:
            return ExtractionResult(
                template_id=TemplateID.MODIFIED_WHO_V1,
                chart_present=True,
                registered=True,
                points=[],
                overall_confidence=0.0,
                warnings=["gemma_extracted_no_marks"],
            )
        confidence = min(point.confidence for point in points)
        tta_warning = "gemma_cuda_extracted" if self.tta_passes == 1 else f"gemma_cuda_tta{self.tta_passes}_extracted"
        return ExtractionResult(
            template_id=TemplateID.MODIFIED_WHO_V1,
            chart_present=True,
            registered=True,
            points=points,
            overall_confidence=confidence,
            warnings=[tta_warning],
        )


def _bounded_points_from_payload(payload: object, candidates: list[DilationPoint]) -> list[DilationPoint]:
    if not isinstance(payload, dict):
        raise ValueError("Gemma payload must be a JSON object")
    accepted = payload.get("accepted_points")
    if not isinstance(accepted, list):
        raise ValueError("Gemma payload must contain accepted_points")

    candidate_map = {_candidate_key(point): point for point in candidates}
    verified: list[DilationPoint] = []
    seen_keys: set[tuple[float, float]] = set()
    for item in accepted:
        if not isinstance(item, dict):
            raise ValueError("Gemma accepted point must be an object")
        try:
            key = (round(float(item["x_hours"]), 1), round(float(item["dilation_cm"]), 1))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Gemma accepted point has invalid coordinates") from exc
        original = candidate_map.get(key)
        if original is None:
            raise ValueError("Gemma returned an invented point")
        if key in seen_keys:
            raise ValueError("Gemma returned duplicate accepted points")
        seen_keys.add(key)
        gemma_confidence = float(item.get("confidence", original.confidence))
        verified.append(original.model_copy(update={"confidence": min(original.confidence, max(0.0, min(1.0, gemma_confidence))), "source": "cv_gemma_verified"}))
    return verified


def _candidate_key(point: DilationPoint) -> tuple[float, float]:
    return (round(point.x_hours, 1), round(point.dilation_cm, 1))


def _build_prompt(extraction: ExtractionResult, chart_crop_path: Path | None) -> str:
    payload = extraction.model_dump(mode="json")
    payload["chart_crop_path"] = str(chart_crop_path) if chart_crop_path else None
    return json.dumps(payload)


def _build_litert_prompt(extraction: ExtractionResult, *, with_image: bool = False) -> str:
    candidates = [
        {
            "x_hours": round(point.x_hours, 1),
            "dilation_cm": round(point.dilation_cm, 1),
            "confidence": round(point.confidence, 3),
            "source": point.source,
        }
        for point in extraction.points
    ]
    image_clause = (
        "You are also shown the registered chart crop as an attached image. "
        "Compare each candidate (x_hours, dilation_cm) to the actual X marks visible on the chart, "
        "and only accept candidates that you can visually confirm sit on a real X mark. "
        if with_image
        else ""
    )
    return (
        "You are PartoGuard's bounded Gemma E2B verifier. "
        "You do not make clinical decisions and you must not invent missing points. "
        + image_clause +
        "Given candidate cervical-dilation X marks already proposed by deterministic vision, "
        "return valid JSON only with an accepted_points array. "
        "Each accepted point must exactly reuse an x_hours and dilation_cm pair from the candidates. "
        "If none are plausible, return {\"accepted_points\": []}.\n\n"
        f"Candidates: {json.dumps(candidates, separators=(',', ':'))}\n\n"
        "Return schema: {\"accepted_points\":[{\"x_hours\":0.0,\"dilation_cm\":0.0,\"confidence\":0.0}]}"
    )


def _litert_env(pythonpath: Path, hf_home: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(pythonpath) if not existing_pythonpath else f"{pythonpath}{os.pathsep}{existing_pythonpath}"
    env["HF_HOME"] = str(hf_home)
    return env


def _build_litert_extraction_prompt() -> str:
    return (
        "You are reading a modified WHO partograph chart. Your task: find ONLY "
        "the handwritten X marks in the cervicograph plot.\n\n"
        "STEP 1 - BLANK CHECK (do this first, every time):\n"
        "Look at the cervicograph plot (the grid where cervical dilation 0-10 cm "
        "is on the y-axis and hours 0-12 is on the x-axis, with two diagonal "
        "Alert and Action lines crossing it).\n"
        "Do you see ANY handwritten X marks drawn by a clinician inside this "
        "plot? An X mark is two short pen strokes crossing at one point.\n"
        '- If you see ZERO X marks, output exactly: {"p":[]}\n'
        "- A printed grid intersection is NOT an X mark.\n"
        "- A diagonal Alert line or Action line is NOT an X mark.\n"
        '- An axis number ("4", "10") is NOT an X mark.\n'
        "- The PLUS sign (+) in a date or text field is NOT an X mark.\n"
        '- An empty chart MUST produce {"p":[]}. Do not invent marks.\n\n'
        "STEP 2 - IDENTIFY THE CERVICOGRAPH (only if Step 1 found marks):\n"
        "The cervicograph is one specific plot on the form. Ignore EVERYTHING "
        "outside it:\n"
        "- Fetal Heart Rate plot uses DOTS or a CONNECTED LINE on a 100-200 bpm "
        "scale. NOT X marks. IGNORE.\n"
        "- Contraction bars are HATCHED or SHADED RECTANGLES at the bottom. "
        "NOT X marks. IGNORE.\n"
        "- Drug, IV, urine, BP, pulse rows are TEXT or ARROWS. NOT X marks. "
        "IGNORE.\n"
        "- Patient header text and dates. NOT X marks. IGNORE.\n\n"
        "STEP 3 - COUNT FIRST, THEN PLOT:\n"
        "Silently count the true X marks inside the cervicograph plot area. "
        "Scan strictly left-to-right, column by column, paying special attention "
        "to early steep vertical jumps where multiple X marks can be stacked "
        "close together on the x-axis. Explicitly verify you have located both "
        "the earliest mark (lowest x_hours) and the latest mark (highest "
        "x_hours) before listing. Then list exactly that many points and no more.\n"
        '- Do NOT add a mark because the trajectory "should" continue past '
        "visible marks.\n"
        "- Do NOT extrapolate. Do NOT interpolate.\n"
        "- If part of the cervicograph is covered by a finger, stain, fold, or "
        "tape, list only the X marks you can clearly see and stop.\n"
        "- If you see fewer than the number you initially counted, output the "
        "smaller number - only marks you can visually verify.\n\n"
        "COORDINATES (for each X mark you list):\n"
        "- x_hours = hours on the x-axis where the X is centred, rounded to the "
        "nearest 0.5, in [0, 12]. Anchor x_hours to the printed vertical grid "
        "lines; double-check the first mark's x-coordinate so you do not shift "
        "the entire curve to the right.\n"
        "- dilation_cm = cervical dilation on the y-axis where the X is centred, "
        "rounded to the nearest 0.5, in [0, 10].\n"
        "- confidence = your confidence the mark is real and correctly placed, "
        "in [0, 1].\n\n"
        "OUTPUT: compact JSON only. No markdown. No prose. No explanation.\n\n"
        'Schema: {"p":[[x_hours, dilation_cm, confidence], ...]}\n\n'
        'Empty cervicograph: {"p":[]}'
    )


def _normalize_extraction_payload(payload: object) -> object:
    """Accept compact `{"p":[[h,d,c],...]}` or verbose `{"points":[{...}]}` schema.

    Returns the verbose form that `_bounded_extracted_points_from_payload` expects.
    """
    if isinstance(payload, dict) and "p" in payload and "points" not in payload:
        raw = payload["p"]
        if not isinstance(raw, list):
            raise ValueError("Gemma extraction 'p' must be a list")
        points: list[dict[str, float]] = []
        for item in raw:
            if not isinstance(item, list) or len(item) < 2:
                raise ValueError("Gemma compact point must be [h,d] or [h,d,c]")
            h, d = item[0], item[1]
            c = item[2] if len(item) >= 3 else 0.5
            points.append({"x_hours": float(h), "dilation_cm": float(d), "confidence": float(c)})
        return {"points": points}
    return payload


_EXTRACT_MAX_POINTS = 20


def _phase1_points_from_payload(payload: object) -> list[DilationPoint]:
    """Phase-1 permissive parser: clamp to chart ranges and round to 0.5. No rejections.

    Intentionally has no plausibility, duplicate, or max-count rejection. The goal of
    phase 1 is to measure raw model behavior. Bounded guardrails will return in phase 2.
    """
    if not isinstance(payload, dict):
        return []
    raw_points = payload.get("points")
    if not isinstance(raw_points, list):
        return []
    out: list[DilationPoint] = []
    for item in raw_points:
        if not isinstance(item, dict):
            continue
        try:
            x = float(item["x_hours"])
            d = float(item["dilation_cm"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            c = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            c = 0.5
        x = max(0.0, min(12.0, x))
        d = max(0.0, min(10.0, d))
        c = max(0.0, min(1.0, c))
        x = round(x * 2.0) / 2.0
        d = round(d * 2.0) / 2.0
        out.append(DilationPoint(x_hours=x, dilation_cm=d, confidence=c, source="gemma_raw"))
    return sorted(out, key=lambda p: (p.x_hours, p.dilation_cm))


def _bounded_extracted_points_from_payload(payload: object) -> list[DilationPoint]:
    """Validate Gemma-extracted points. Raises ValueError on any violation.

    Schema: {"points": [{"x_hours": float in [0,12], "dilation_cm": float in [0,10],
                          "confidence": float in [0,1]}]}.
    Coordinates are rounded to nearest 0.5. Duplicates and implausible trajectories
    are rejected so the pipeline can force manual_review.
    """
    if not isinstance(payload, dict):
        raise ValueError("Gemma extraction payload must be a JSON object")
    raw_points = payload.get("points")
    if not isinstance(raw_points, list):
        raise ValueError("Gemma extraction payload must contain points list")
    if len(raw_points) > _EXTRACT_MAX_POINTS:
        raise ValueError("Gemma returned implausibly many points")

    extracted: list[DilationPoint] = []
    seen_keys: set[tuple[float, float]] = set()
    for item in raw_points:
        if not isinstance(item, dict):
            raise ValueError("Gemma extracted point must be an object")
        try:
            x_hours = float(item["x_hours"])
            dilation_cm = float(item["dilation_cm"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Gemma extracted point has invalid coordinates") from exc
        if not (0.0 <= x_hours <= 12.0):
            raise ValueError("Gemma extracted point x_hours out of range")
        if not (0.0 <= dilation_cm <= 10.0):
            raise ValueError("Gemma extracted point dilation_cm out of range")
        x_hours = round(x_hours * 2.0) / 2.0
        dilation_cm = round(dilation_cm * 2.0) / 2.0
        confidence_raw = item.get("confidence", 0.5)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("Gemma extracted point has invalid confidence") from exc
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("Gemma extracted point confidence out of range")
        key = (x_hours, dilation_cm)
        if key in seen_keys:
            raise ValueError("Gemma extraction contains duplicate point")
        seen_keys.add(key)
        extracted.append(
            DilationPoint(
                x_hours=x_hours,
                dilation_cm=dilation_cm,
                confidence=confidence,
                source="gemma_e2b_extracted",
            )
        )

    if len(extracted) >= 2 and not _extracted_points_are_plausible(extracted):
        raise ValueError("Gemma extracted points form implausible trajectory")
    return sorted(extracted, key=lambda p: (p.x_hours, p.dilation_cm))


def _extracted_points_are_plausible(points: list[DilationPoint]) -> bool:
    ordered = sorted(points, key=lambda p: p.x_hours)
    for prev, cur in zip(ordered, ordered[1:]):
        if cur.dilation_cm + 0.5 < prev.dilation_cm:
            return False
    distinct_hours = {round(p.x_hours, 1) for p in points}
    distinct_dilations = {round(p.dilation_cm, 1) for p in points}
    return len(distinct_hours) >= 2 and len(distinct_dilations) >= 2


def _json_payload_from_text(text: str) -> object:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(stripped[start:end + 1])


def _manual_review_from(extraction: ExtractionResult, warning: str) -> ExtractionResult:
    return ExtractionResult(
        template_id=TemplateID.UNKNOWN if extraction.template_id == TemplateID.UNKNOWN else extraction.template_id,
        chart_present=extraction.chart_present,
        registered=extraction.registered,
        points=[],
        overall_confidence=0.0,
        warnings=[*extraction.warnings, warning, "manual_review"],
    )
