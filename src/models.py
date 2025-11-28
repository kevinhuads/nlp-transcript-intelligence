from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TranscriptSegment":
        return TranscriptSegment(
            start=float(data["start"]),
            end=float(data["end"]),
            text=str(data["text"]),
        )


@dataclass
class OCRRecord:
    time: float
    frame: str
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "OCRRecord":
        return OCRRecord(
            time=float(data["time"]),
            frame=str(data["frame"]),
            text=str(data["text"]),
        )


@dataclass
class Segment:
    start: float
    end: float
    mid: float
    speech: str
    slide_text: str
    slide_time: Optional[float]
    slide_frame: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Segment":
        return Segment(
            start=float(data["start"]),
            end=float(data["end"]),
            mid=float(data["mid"]),
            speech=str(data["speech"]),
            slide_text=str(data.get("slide_text", "")),
            slide_time=data.get("slide_time"),
            slide_frame=data.get("slide_frame"),
        )


def segments_to_jsonable(items: List[Segment]) -> List[Dict[str, Any]]:
    return [item.to_dict() for item in items]


def transcript_to_jsonable(items: List[TranscriptSegment]) -> List[Dict[str, Any]]:
    return [item.to_dict() for item in items]


def ocr_to_jsonable(items: List[OCRRecord]) -> List[Dict[str, Any]]:
    return [item.to_dict() for item in items]
