from pathlib import Path

from backend.ocr_quality import assess_ocr_quality
from backend.plate_format import format_vn_plate

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024


class ValidationError(ValueError):
    pass


def validate_image_extension(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValidationError("Unsupported image format")
    return suffix


def validate_upload_size(size_bytes: int) -> int:
    if size_bytes <= 0:
        raise ValidationError("Empty upload is not allowed")
    if size_bytes > MAX_UPLOAD_SIZE_BYTES:
        raise ValidationError("Uploaded file is too large")
    return size_bytes


def validate_ticket_id(ticket_id: str) -> str:
    clean = (ticket_id or "").strip()
    if not clean:
        raise ValidationError("Ticket ID is required")
    if len(clean) > 64:
        raise ValidationError("Ticket ID is too long")
    return clean


def validate_plate_text(plate_text: str) -> str:
    clean = (plate_text or "").strip()
    if not clean:
        raise ValidationError("Plate text is required")
    if len(clean) > 32:
        raise ValidationError("Plate text is too long")

    formatted = format_vn_plate(clean)
    assessment = assess_ocr_quality(formatted or clean, 1.0)
    if not assessment.is_usable:
        raise ValidationError(f"Invalid plate text: {assessment.quality_note}")
    return assessment.formatted_text or formatted or clean
