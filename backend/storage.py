import base64
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from backend.config import UPLOAD_DIR
from backend.validators import validate_image_extension, validate_upload_size

try:
    import cv2
except Exception:
    cv2 = None

MAX_OCR_INPUT_DIMENSION = 1280


def _timestamped_path(prefix: str, suffix: str) -> Path:
    safe_prefix = re.sub(r"[^A-Za-z0-9_-]+", "_", (prefix or "file").strip())
    safe_prefix = safe_prefix[:20].strip("_") or "file"
    return UPLOAD_DIR / f"{safe_prefix}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:6]}{suffix}"


def _maybe_resize_for_ocr(buffer: bytes, suffix: str) -> bytes:
    if cv2 is None or not buffer:
        return buffer
    try:
        image_array = cv2.imdecode(__import__('numpy').frombuffer(buffer, dtype=__import__('numpy').uint8), cv2.IMREAD_COLOR)
        if image_array is None:
            return buffer
        height, width = image_array.shape[:2]
        longest_edge = max(height, width)
        if longest_edge <= MAX_OCR_INPUT_DIMENSION:
            return buffer
        scale = MAX_OCR_INPUT_DIMENSION / float(longest_edge)
        resized = cv2.resize(image_array, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        ext = ".png" if suffix.lower() == ".png" else ".jpg"
        encode_ok, encoded = cv2.imencode(ext, resized)
        if not encode_ok:
            return buffer
        return encoded.tobytes()
    except Exception:
        return buffer


def save_uploaded_file(uploaded_file, prefix: str):
    # Streamlit UploadedFile uses .name; FastAPI UploadFile uses .filename
    filename = getattr(uploaded_file, "name", None) or getattr(uploaded_file, "filename", "") or ""
    suffix = validate_image_extension(filename)
    # Streamlit UploadedFile has .getbuffer(); FastAPI UploadFile uses .file.read()
    if hasattr(uploaded_file, "getbuffer"):
        buffer = bytes(uploaded_file.getbuffer())
    else:
        file_obj = getattr(uploaded_file, "file", uploaded_file)
        buffer = file_obj.read()
    validate_upload_size(len(buffer))
    output_path = _timestamped_path(prefix, suffix)
    optimized_buffer = _maybe_resize_for_ocr(buffer, suffix)
    output_path.write_bytes(optimized_buffer)
    return str(output_path)


def save_annotated_image(cv2_module, image, prefix: str):
    output_path = _timestamped_path(prefix, ".jpg")
    cv2_module.imwrite(str(output_path), image)
    return str(output_path)


def image_to_base64(path: Path):
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")

