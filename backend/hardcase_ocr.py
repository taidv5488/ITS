from pathlib import Path

try:
    import cv2
except Exception:
    cv2 = None

from backend.alpr_service import get_detector_choices, run_alpr
from backend.plate_format import format_vn_plate


def _load_image(image_path: str):
    if cv2 is None:
        return None
    return cv2.imread(image_path)


def _save_variant(image, base_path: str, suffix: str) -> str | None:
    if cv2 is None or image is None:
        return None
    src = Path(base_path)
    out = src.with_name(f"{src.stem}_{suffix}{src.suffix}")
    cv2.imwrite(str(out), image)
    return str(out)


def _rotations(image):
    if cv2 is None or image is None:
        return []
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    variants = []
    for angle in (-8, -5, 5, 8):
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        variants.append((f"rot{angle}", rotated))
    return variants


def _external_detector_paths(models_dir: Path):
    return [model_path for _, model_path in get_detector_choices(models_dir) if model_path]


def _is_partial_read_baseline(text: str) -> bool:
    formatted = format_vn_plate(text or "")
    if not formatted:
        return False
    compact = "".join(ch for ch in formatted.upper() if ch.isalnum())
    if compact.isdigit():
        return False
    return len(compact) <= 5 or ("-" in formatted and " " not in formatted)


def _looks_like_garbage_cluster(text: str) -> bool:
    compact = "".join(ch for ch in (text or "").upper() if ch.isalnum())
    if not compact:
        return True
    if compact.isdigit():
        return True
    alpha_count = sum(ch.isalpha() for ch in compact)
    digit_count = sum(ch.isdigit() for ch in compact)
    if alpha_count == 0:
        return True
    if alpha_count >= 2 and digit_count >= 4 and len(compact) >= 7 and "-" not in (text or "") and " " not in (text or ""):
        return True
    return False


def _candidate_is_safe_upgrade(baseline: dict, candidate: dict) -> bool:
    baseline_text = format_vn_plate(baseline.get("text", ""))
    candidate_text = format_vn_plate(candidate.get("text", ""))
    baseline_conf = float(baseline.get("confidence", 0.0) or 0.0)
    candidate_conf = float(candidate.get("confidence", 0.0) or 0.0)
    variant = str(candidate.get("variant", ""))
    is_rotation_variant = variant.startswith("rot")
    is_detector_variant = variant.startswith("detector:")

    if not _is_partial_read_baseline(baseline_text):
        return False
    if _looks_like_garbage_cluster(baseline.get("text", "")):
        return False
    if not (is_rotation_variant or is_detector_variant):
        return False
    if not candidate.get("is_usable"):
        return False
    if len(candidate_text) < 8:
        return False
    if not candidate_text[:2].isdigit():
        return False
    if baseline_text and len(candidate_text) <= len(baseline_text):
        return False

    min_confidence = 0.95 if is_rotation_variant else 0.75
    if candidate_conf < min_confidence:
        return False
    if baseline_conf > 0.0 and candidate_conf + (0.02 if is_detector_variant else 0.0) <= baseline_conf:
        return False
    return True


def run_hardcase_alpr(image_path: str, models_dir: Path) -> dict:
    baseline = run_alpr(image_path)
    text = baseline.get("text", "")
    if baseline.get("is_usable") and text and len(text) >= 7:
        baseline["hardcase_used"] = False
        return baseline

    image = _load_image(image_path)
    candidates = []

    for suffix, variant in _rotations(image):
        variant_path = _save_variant(variant, image_path, suffix)
        if not variant_path:
            continue
        result = run_alpr(variant_path)
        result["variant"] = suffix
        candidates.append(result)

    for detector_model in _external_detector_paths(models_dir):
        result = run_alpr(image_path, detector_model=detector_model)
        result["variant"] = f"detector:{Path(detector_model).name}"
        candidates.append(result)

    safe_candidates = [item for item in candidates if _candidate_is_safe_upgrade(baseline, item)]
    if not safe_candidates:
        baseline["hardcase_used"] = False
        return baseline

    best = sorted(
        safe_candidates,
        key=lambda item: (
            float(item.get("confidence", 0.0) or 0.0),
            len(format_vn_plate(item.get("text", ""))),
        ),
        reverse=True,
    )[0]
    best["hardcase_used"] = True
    return best
