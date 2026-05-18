"""Data contracts for the PartoGuard pipeline.

These schemas define the interfaces between pipeline stages per the implementation plan.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class TemplateID(str, Enum):
    MODIFIED_WHO_V1 = "modified_who_partograph_v1"
    UNKNOWN = "unknown"


class SourceMode(str, Enum):
    CONSOLE = "console"
    MOBILE = "mobile"


class ImageSource(str, Enum):
    SYNTHETIC = "synthetic"
    LOCAL_REFERENCE = "local_reference"
    CAMERA = "camera"


class AnalysisInput(BaseModel):
    image_path: Path
    template_id: TemplateID = TemplateID.MODIFIED_WHO_V1
    mode: SourceMode = SourceMode.CONSOLE
    source: ImageSource = ImageSource.CAMERA


class DilationPoint(BaseModel):
    x_hours: float = Field(ge=0.0)
    dilation_cm: float = Field(ge=0.0, le=10.0)
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = "cv"


class ExtractionResult(BaseModel):
    template_id: TemplateID
    chart_present: bool
    registered: bool
    points: list[DilationPoint] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    warnings: list[str] = Field(default_factory=list)


class ZoneStatus(str, Enum):
    NORMAL = "normal"
    ALERT_ZONE = "alert_zone"
    ACTION_ZONE = "action_zone"
    INDETERMINATE = "indeterminate"
    MANUAL_REVIEW = "manual_review"


class RuleOutput(BaseModel):
    status: ZoneStatus
    framework: str = "modified_who_partograph"
    triggering_point: DilationPoint | None = None
    explanation: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    requires_human_review: bool = True
