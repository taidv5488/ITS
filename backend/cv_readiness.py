from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util

from backend.config import ENABLE_OPENAI_VISION, ENABLE_PADDLE_FALLBACK, ENABLE_PADDLE_OCR, FAST_MODE, OCR_ACCURACY_MODE

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "external"


@dataclass(frozen=True)
class CvReadinessItem:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class ModelInventoryItem:
    name: str
    path: str
    size_mb: float


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def list_external_models(model_dir: Path = MODEL_DIR) -> list[ModelInventoryItem]:
    if not model_dir.exists():
        return []
    models = []
    for path in sorted(model_dir.glob("*.pt")):
        models.append(ModelInventoryItem(name=path.name, path=str(path), size_mb=round(path.stat().st_size / (1024 * 1024), 2)))
    return models


def build_cv_readiness() -> dict:
    cv2_ok = _module_available("cv2")
    fast_alpr_ok = _module_available("fast_alpr")
    easyocr_ok = _module_available("easyocr")
    paddle_ok = _module_available("paddleocr")
    models = list_external_models()
    checks = [
        CvReadinessItem("opencv", "pass" if cv2_ok else "fail", "OpenCV available" if cv2_ok else "Install opencv-python"),
        CvReadinessItem("fast_alpr", "pass" if fast_alpr_ok else "warn", "fast-alpr available" if fast_alpr_ok else "fast-alpr missing; detector pipeline will fallback"),
        CvReadinessItem("easyocr", "pass" if easyocr_ok else "warn", "EasyOCR available" if easyocr_ok else "EasyOCR missing"),
        CvReadinessItem("paddleocr_installed", "pass" if paddle_ok else "warn", "PaddleOCR installed" if paddle_ok else "PaddleOCR missing"),
        CvReadinessItem("paddleocr_enabled", "pass" if ENABLE_PADDLE_OCR else "warn", "PaddleOCR enabled as primary selectable engine" if ENABLE_PADDLE_OCR else "PaddleOCR primary engine disabled"),
        CvReadinessItem("paddleocr_fallback", "pass" if ENABLE_PADDLE_FALLBACK and paddle_ok else "warn", "PaddleOCR fallback enabled for weak cases" if ENABLE_PADDLE_FALLBACK and paddle_ok else "PaddleOCR fallback disabled or unavailable"),
        CvReadinessItem("ocr_accuracy_mode", "pass" if OCR_ACCURACY_MODE else "info", "Accuracy-first OCR mode enabled" if OCR_ACCURACY_MODE else "Speed-first OCR mode"),
        CvReadinessItem("external_models", "pass" if models else "warn", f"{len(models)} external .pt model(s) found"),
        CvReadinessItem("fast_mode", "warn" if FAST_MODE else "pass", "FAST_MODE enabled: faster but fewer OCR enhancement variants" if FAST_MODE else "Full OCR enhancement mode enabled"),
        CvReadinessItem("openai_vision", "pass" if ENABLE_OPENAI_VISION else "info", "OpenAI Vision fallback enabled" if ENABLE_OPENAI_VISION else "OpenAI Vision fallback disabled"),
    ]
    summary = {
        "pass": sum(1 for item in checks if item.status == "pass"),
        "warn": sum(1 for item in checks if item.status == "warn"),
        "fail": sum(1 for item in checks if item.status == "fail"),
        "info": sum(1 for item in checks if item.status == "info"),
    }
    return {
        "checks": [item.__dict__ for item in checks],
        "models": [item.__dict__ for item in models],
        "summary": summary,
        "pipeline_grade": "field-ready-pending-images" if summary["fail"] == 0 and models else "needs-attention",
    }
