import re
from dataclasses import dataclass

from backend.plate_format import classify_plate_type, format_plate_by_type, looks_like_vn_bottom, looks_like_vn_top, normalize_plate_text


MIN_CONFIDENCE_FOR_USABLE_OCR = 0.72
STRONG_CONFIDENCE_FOR_USABLE_OCR = 0.82
MANUAL_REVIEW_CONFIDENCE_FLOOR = 0.78
DOTTED_FIVE_DIGIT_CONFIDENCE_FLOOR = 0.75


@dataclass(frozen=True)
class OcrQualityAssessment:
    is_usable: bool
    quality_note: str
    normalized_text: str
    formatted_text: str
    plate_type: str


FULL_VN_PATTERNS = (
    re.compile(r"\d{2}[A-Z]\d{5}"),
    re.compile(r"\d{2}[A-Z]\d{6}"),
    re.compile(r"\d{2}[A-Z]{2}\d{4}"),
    re.compile(r"\d{2}[A-Z]{2}\d{5}"),
)

FORMATTED_FULL_VN_PATTERNS = (
    re.compile(r"\d{2}[A-Z]\d{5}"),
    re.compile(r"\d{2}[A-Z]\d{6}"),
    re.compile(r"\d{2}[A-Z]{2}\d{4}"),
    re.compile(r"\d{2}[A-Z]{2}\d{5}"),
)


def clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def is_full_vn_plate(normalized_text: str) -> bool:
    return any(pattern.fullmatch(normalized_text) for pattern in FULL_VN_PATTERNS)


def is_full_vn_plate_formatted(formatted_text: str) -> bool:
    compact = "".join(ch for ch in (formatted_text or "").upper() if ch.isalnum())
    return any(pattern.fullmatch(compact) for pattern in FORMATTED_FULL_VN_PATTERNS)


def assess_ocr_quality(text: str, confidence: float) -> OcrQualityAssessment:
    raw_text = (text or "").upper().strip()
    plate_type = classify_plate_type(text)
    formatted = format_plate_by_type(text, plate_type)
    normalized = normalize_plate_text(formatted)
    conf = clamp_confidence(confidence)
    compact_formatted = "".join(ch for ch in formatted.upper() if ch.isalnum())
    raw_alpha_count = sum(ch.isalpha() for ch in raw_text)
    alpha_count = sum(ch.isalpha() for ch in compact_formatted)
    digit_count = sum(ch.isdigit() for ch in normalized)
    is_dotted_five_digit_plate = bool(re.fullmatch(r"\d{2}-[A-Z0-9]{2,3} \d{3}\.\d{2}", formatted or ""))

    if not normalized:
        return OcrQualityAssessment(False, "Không có ký tự hợp lệ", normalized, formatted, plate_type)

    if looks_like_vn_top(normalized) or looks_like_vn_bottom(normalized):
        return OcrQualityAssessment(False, "Mới đọc được một phần biển số", normalized, formatted, plate_type)

    if alpha_count == 0 and len(normalized) >= 7:
        return OcrQualityAssessment(False, "Chuỗi OCR thiếu ký tự chữ, khả năng sai định dạng biển số", normalized, formatted, plate_type)

    if len(normalized) < 7:
        return OcrQualityAssessment(False, "Kết quả OCR quá ngắn để tin cậy", normalized, formatted, plate_type)

    if is_full_vn_plate(normalized) or is_full_vn_plate_formatted(formatted):
        required_confidence = DOTTED_FIVE_DIGIT_CONFIDENCE_FLOOR if is_dotted_five_digit_plate else MIN_CONFIDENCE_FOR_USABLE_OCR
        if conf >= required_confidence and raw_alpha_count >= 1:
            return OcrQualityAssessment(True, "Biển số hợp lệ", normalized, formatted, plate_type)
        if conf >= MANUAL_REVIEW_CONFIDENCE_FLOOR and alpha_count >= 1 and digit_count >= 6:
            return OcrQualityAssessment(True, "Biển số gần đúng, cần xác nhận thủ công", normalized, formatted, plate_type)
        return OcrQualityAssessment(False, "Đúng định dạng nhưng confidence chưa đủ cao", normalized, formatted, plate_type)

    if conf >= STRONG_CONFIDENCE_FOR_USABLE_OCR and alpha_count >= 1 and digit_count >= 5:
        return OcrQualityAssessment(True, "OCR mạnh nhưng chưa khớp hoàn toàn định dạng", normalized, formatted, plate_type)

    return OcrQualityAssessment(False, "Kết quả OCR chưa đủ tin cậy", normalized, formatted, plate_type)
