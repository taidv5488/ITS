import hashlib
import inspect
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.parking_service import vehicle_entry, vehicle_exit
from backend.config import ENABLE_HARDCASE_OCR
from backend.validators import validate_plate_text, validate_ticket_id

ROOT_DIR = Path(__file__).resolve().parents[1]
EXTERNAL_MODELS_DIR = ROOT_DIR / "models" / "external"

# In thời gian từng bước ra terminal để biết chỗ nào chậm. Đặt ITS_TIMING=0 để tắt.
TIMING_ENABLED = os.getenv("ITS_TIMING", "1").lower() not in {"0", "false", "no", "off"}

# Cache kết quả nhận diện theo NỘI DUNG ảnh (không theo đường dẫn), vì Streamlit chạy lại
# cả script sau mỗi thao tác và app lưu ảnh với tên file mới mỗi lần.
_RESULT_CACHE_SIZE = 32
_result_cache: "OrderedDict[tuple, tuple[dict[str, Any], str]]" = OrderedDict()
_cache_lock = threading.Lock()


@dataclass(frozen=True)
class ProcessedPlate:
    ocr: dict[str, Any]
    plate_text: str


def _timing(label: str, started: float, extra: str = "") -> None:
    if TIMING_ENABLED:
        print(f"[TIMING] {label}: {time.perf_counter() - started:.2f}s {extra}".rstrip(), flush=True)


def _runner_accepts_engine(runner) -> bool:
    """Kiểm tra một lần xem runner có nhận tham số ocr_engine không, thay vì chạy OCR rồi bắt TypeError."""
    try:
        params = inspect.signature(runner).parameters
    except (TypeError, ValueError):
        return False
    return "ocr_engine" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _file_digest(image_path: str) -> str | None:
    try:
        return hashlib.sha1(Path(image_path).read_bytes()).hexdigest()
    except OSError:
        return None


def _cache_get(key):
    with _cache_lock:
        hit = _result_cache.get(key)
        if hit is None:
            return None
        _result_cache.move_to_end(key)
        return hit


def _cache_put(key, ocr: dict[str, Any], plate_text: str) -> None:
    with _cache_lock:
        _result_cache[key] = (dict(ocr), plate_text)
        _result_cache.move_to_end(key)
        while len(_result_cache) > _RESULT_CACHE_SIZE:
            _result_cache.popitem(last=False)


def process_plate_image(
    image_path: str,
    detector_model: str | None = None,
    ocr_engine: str = "auto",
    use_hardcase: bool | None = None,
    alpr_runner=None,
) -> ProcessedPlate:
    hardcase_enabled = ENABLE_HARDCASE_OCR if use_hardcase is None else use_hardcase

    # Chỉ cache khi dùng runner mặc định (runner tự truyền vào, ví dụ khi test, thì không cache)
    cache_key = None
    if alpr_runner is None:
        digest = _file_digest(image_path)
        if digest:
            cache_key = (digest, detector_model, ocr_engine, bool(hardcase_enabled))
            hit = _cache_get(cache_key)
            if hit is not None:
                cached_ocr, cached_text = hit
                annotated = cached_ocr.get("annotated_path")
                if not annotated or Path(annotated).exists():
                    if TIMING_ENABLED:
                        print("[TIMING] process_plate_image: dùng kết quả đã cache (ảnh trùng nội dung)", flush=True)
                    return ProcessedPlate(ocr=dict(cached_ocr), plate_text=cached_text)

        from backend.alpr_service import run_alpr

        alpr_runner = run_alpr

    t0 = time.perf_counter()
    if _runner_accepts_engine(alpr_runner):
        ocr = alpr_runner(image_path, detector_model=detector_model, ocr_engine=ocr_engine)
    else:
        ocr = alpr_runner(image_path, detector_model=detector_model)
    _timing("alpr_runner", t0, f"| engine={ocr_engine}")
    plate_text = (ocr.get("text") or "").strip()

    if hardcase_enabled and not detector_model and (not ocr.get("is_usable") or len(plate_text) < 7):
        t1 = time.perf_counter()
        from backend.hardcase_ocr import run_hardcase_alpr

        hardcase = run_hardcase_alpr(image_path, EXTERNAL_MODELS_DIR)
        hardcase_text = (hardcase.get("text") or "").strip()
        if hardcase.get("is_usable") or len(hardcase_text) > len(plate_text):
            ocr = hardcase
            plate_text = hardcase_text
        _timing("hardcase", t1, "(chạy thêm vì kết quả đầu chưa đạt)")

    if cache_key is not None and plate_text:
        _cache_put(cache_key, ocr, plate_text)

    return ProcessedPlate(ocr=ocr, plate_text=plate_text)


def confirm_vehicle_entry(plate_text: str, *, image_path: str | None = None, detected_path: str | None = None, plate_path: str | None = None, ocr_confidence: float | None = None, ocr_engine: str | None = None, ocr_text: str | None = None, manual_override: bool = False, actor=None):
    validated_plate = validate_plate_text(plate_text)
    return vehicle_entry(
        validated_plate,
        image_path=image_path,
        detected_path=detected_path,
        plate_path=plate_path,
        ocr_confidence=ocr_confidence,
        ocr_engine=ocr_engine,
        ocr_text=ocr_text,
        manual_override=manual_override,
        actor=actor,
    )


def confirm_vehicle_exit(ticket_id: str, plate_text: str, *, image_path: str | None = None, detected_path: str | None = None, plate_path: str | None = None, ocr_confidence: float | None = None, ocr_engine: str | None = None, ocr_text: str | None = None, manual_override: bool = False, actor=None):
    validated_ticket = validate_ticket_id(ticket_id)
    validated_plate = validate_plate_text(plate_text)
    return vehicle_exit(
        validated_ticket,
        validated_plate,
        image_path=image_path,
        detected_path=detected_path,
        plate_path=plate_path,
        ocr_confidence=ocr_confidence,
        ocr_engine=ocr_engine,
        ocr_text=ocr_text,
        manual_override=manual_override,
        actor=actor,
    )