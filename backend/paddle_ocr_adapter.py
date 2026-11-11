from pathlib import Path

from backend.logging_config import get_logger
from backend.ocr_quality import assess_ocr_quality
from backend.plate_format import format_vn_plate

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

logger = get_logger(__name__)


class PaddleOCRAdapter:
    def __init__(self) -> None:
        if PaddleOCR is None:
            raise RuntimeError("paddleocr is not installed")
        self.reader = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)

    def predict_image(self, image_path: str) -> dict:
        result = self.reader.predict(input=image_path)
        texts = []
        scores = []

        try:
            for item in result:
                rec_texts = item.get("rec_texts") or []
                rec_scores = item.get("rec_scores") or []
                texts.extend(rec_texts)
                scores.extend(rec_scores)
        except Exception as exc:
            logger.warning("PaddleOCR parse failed: %s", exc)

        raw_text = " ".join(part.strip() for part in texts if part and str(part).strip())
        confidence = float(sum(scores) / len(scores)) if scores else 0.0
        formatted = format_vn_plate(raw_text)
        assessment = assess_ocr_quality(formatted or raw_text, confidence)
        return {
            "engine": "paddleocr",
            "text": formatted,
            "raw_text": raw_text,
            "confidence": confidence,
            "is_usable": assessment.is_usable,
            "quality_note": assessment.quality_note,
            "normalized_text": assessment.normalized_text,
        }


from functools import lru_cache

@lru_cache(maxsize=1)
def _get_adapter() -> PaddleOCRAdapter:
    return PaddleOCRAdapter()

def run_paddleocr(image_path: str) -> dict:
    return _get_adapter().predict_image(image_path)