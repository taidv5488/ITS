import os
import re
import sys
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLEOCR_HOME", str(Path.home() / ".paddleocr"))

from backend.config import DEFAULT_MODEL_LABEL, DEEP_OCR_CONFIDENCE_THRESHOLD, ENABLE_DEEP_OCR_RETRY, ENABLE_OPENAI_VISION, ENABLE_PADDLE_FALLBACK, ENABLE_PADDLE_OCR, ENABLE_VISION_IN_AUTO, FAST_MODE, OCR_ACCURACY_MODE
from backend.ocr_quality import assess_ocr_quality
from backend.plate_format import (
    candidate_bottoms,
    classify_plate_type,
    combine_vn_two_line,
    format_plate_by_type,
    format_vn_plate,
    is_top_only_plate,
    looks_like_vn_bottom,
    looks_like_vn_top,
    normalize_plate_text,
    normalize_vn_top,
)
from backend.storage import save_annotated_image

try:
    import cv2
except Exception:
    cv2 = None

try:
    import easyocr
except Exception:
    easyocr = None

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

try:
    from backend.openai_vision_adapter import run_openai_vision
except Exception:
    run_openai_vision = None

try:
    import site

    user_site = site.getusersitepackages()
    if user_site and user_site not in sys.path:
        sys.path.append(user_site)
except Exception:
    pass

try:
    from fast_alpr import ALPR
    from fast_alpr.alpr import BaseOCR, OcrResult
except Exception:
    ALPR = None
    BaseOCR = object
    OcrResult = None


def repair_vn_top(text: str):
    s = normalize_vn_top(text)
    if re.fullmatch(r"\d{2}[A-Z]", s) or re.fullmatch(r"\d{2}[A-Z]{2}", s) or re.fullmatch(r"\d{2}[A-Z]\d", s) or re.fullmatch(r"\d{2}[A-Z]{2}\d", s):
        return s
    if len(s) >= 3 and s[:2].isdigit():
        tail = s[2:]
        if not tail:
            return s
        first = tail[0]
        fix_map = {"8": "B", "6": "G", "0": "D", "2": "Z", "5": "S"}
        fixed = fix_map.get(first, first)
        candidate = s[:2] + fixed + tail[1:]
        if re.fullmatch(r"\d{2}[A-Z]", candidate) or re.fullmatch(r"\d{2}[A-Z]{2}", candidate) or re.fullmatch(r"\d{2}[A-Z]\d", candidate) or re.fullmatch(r"\d{2}[A-Z]{2}\d", candidate):
            return candidate
    return s


def repair_vn_bottom(text: str):
    s = normalize_plate_text(text)
    if re.fullmatch(r"\d{4}", s) or re.fullmatch(r"\d{5}", s):
        return s
    if len(s) == 5:
        sub_map = {"B": "8", "S": "5", "Z": "2", "G": "6", "I": "1", "L": "1", "O": "0", "D": "0"}
        repaired = "".join(sub_map.get(ch, ch) for ch in s)
        if re.fullmatch(r"\d{5}", repaired):
            return repaired
    return s


def score_full_plate(text: str, conf: float):
    norm = normalize_plate_text(text)
    if not norm:
        return -999.0
    score = conf * 100.0
    if re.fullmatch(r"\d{2}[A-Z]\d{5}", norm):
        score += 260
    elif re.fullmatch(r"\d{2}[A-Z]{2}\d{4}", norm):
        score += 255
    elif re.fullmatch(r"\d{2}[A-Z]{2}\d{5}", norm):
        score += 260
    elif re.fullmatch(r"\d{2}[A-Z]\d{4,5}", norm):
        score += 180
    elif looks_like_vn_top(norm):
        score += 60
    elif looks_like_vn_bottom(norm):
        score += 40
    if 6 <= len(norm) <= 9:
        score += 45
    elif len(norm) <= 3:
        score -= 50
    if any(ch.isalpha() for ch in norm[4:]):
        score -= 35
    if len(norm) >= 2 and not norm[:2].isdigit():
        score -= 60
    return score




class EasyOCRVNOCR(BaseOCR):
    _shared_reader = None

    def __init__(self) -> None:
        if easyocr is None:
            raise RuntimeError("easyocr is not installed")
        if EasyOCRVNOCR._shared_reader is None:
            EasyOCRVNOCR._shared_reader = easyocr.Reader(["en"], gpu=False)
        self.reader = EasyOCRVNOCR._shared_reader

    def _enhance(self, image, force_full: bool = False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        blur = cv2.bilateralFilter(gray, 5, 35, 35)
        sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(sharp)
        if FAST_MODE and not force_full:
            return [("clahe", clahe)]
        _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
        return [("clahe", clahe), ("otsu", otsu), ("adaptive", adaptive), ("morph", morph), ("sharp", sharp)]

    def _ocr_candidates(self, image, force_full: bool = False):
        if image is None or image.size == 0:
            return []
        candidates = []
        for _, variant in self._enhance(image, force_full=force_full):
            try:
                rows = self.reader.readtext(
                    variant,
                    detail=1,
                    paragraph=False,
                    batch_size=1,
                    decoder="greedy",
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-",
                    width_ths=0.7,
                    link_threshold=0.2,
                    text_threshold=0.55,
                    low_text=0.3,
                )
            except Exception:
                rows = []
            if rows:
                line_text = " ".join(item[1].strip() for item in rows if len(item) >= 3 and item[1].strip())
                if line_text:
                    line_conf = sum(float(item[2]) for item in rows if len(item) >= 3) / max(1, len(rows))
                    candidates.append((line_text, line_conf))
            for row in rows:
                if len(row) >= 3:
                    _, text, conf = row
                    candidates.append((text.strip(), float(conf)))
        dedup = {}
        for text, conf in candidates:
            norm = normalize_plate_text(text)
            if not norm:
                continue
            score = score_full_plate(text, conf)
            if norm not in dedup or score > dedup[norm][0]:
                dedup[norm] = (score, text, conf)
        sorted_rows = sorted(dedup.values(), key=lambda x: (x[0], x[2]), reverse=True)
        return [(text, conf) for _, text, conf in sorted_rows]

    def _ocr_best(self, image, force_full: bool = False):
        candidates = self._ocr_candidates(image, force_full=force_full)
        if not candidates:
            return "", 0.0
        best_text, best_conf = candidates[0]
        return best_text, best_conf

    def _split_vertical_plate(self, image, top_ratio=0.52, overlap_ratio=0.34, top_trim=0.08):
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return image, image
        top_cut = int(h * top_trim)
        base = image[top_cut:, :] if top_cut < h else image
        hh, _ = base.shape[:2]
        top = base[: max(1, int(hh * top_ratio)), :]
        bottom = base[max(0, int(hh * overlap_ratio)) :, :]
        return top, bottom

    def _combine_from_candidates(self, top_rows, bottom_rows):
        top_candidates = [(text, conf) for text, conf in top_rows[:6] if looks_like_vn_top(text)] or top_rows[:4]
        bottom_candidates = [(text, conf) for text, conf in bottom_rows[:8] if looks_like_vn_bottom(text)] or bottom_rows[:5]
        best = ("", 0.0)
        for t_text, t_conf in top_candidates:
            repaired_top = repair_vn_top(t_text)
            for b_text, b_conf in bottom_candidates:
                bottom_pool = candidate_bottoms(b_text) + candidate_bottoms(repair_vn_bottom(b_text))
                for bottom_candidate in dict.fromkeys(bottom_pool):
                    combined = combine_vn_two_line(repaired_top, bottom_candidate)
                    if not combined:
                        continue
                    combined_norm = normalize_plate_text(combined)
                    score = (t_conf + b_conf) / 2.0
                    if looks_like_vn_top(repaired_top) and looks_like_vn_bottom(bottom_candidate):
                        score += 0.10
                    if len(combined_norm) >= 7:
                        score += 0.05
                    if re.fullmatch(r'\d{2}[A-Z]\d{4}', combined_norm) or re.fullmatch(r'\d{2}[A-Z]{2}\d{4}', combined_norm):
                        score += 0.10
                    if re.fullmatch(r'\d{2}[A-Z]\d{5}', combined_norm) or re.fullmatch(r'\d{2}[A-Z]{2}\d{5}', combined_norm):
                        score += 0.08
                    if score > best[1]:
                        best = (combined, min(0.999, score))
        return best

    def _needs_rescue_ocr(self, text: str, conf: float, ratio: float, has_bottom_rows: bool) -> bool:
        assessment = assess_ocr_quality(text or "", conf)
        normalized_len = len(assessment.normalized_text or "")
        if not text:
            return True
        if conf < 0.86:
            return True
        if not assessment.is_usable:
            return True
        if ratio < 2.2 and (has_bottom_rows or assessment.plate_type in {"vn_partial_top", "vn_partial_bottom", "unknown"}):
            return True
        if normalized_len < 8:
            return True
        if conf < DEEP_OCR_CONFIDENCE_THRESHOLD:
            return True
        return False

    def _predict_core(self, cropped_plate, force_full: bool = False):
        if cropped_plate is None:
            return None

        h, w = cropped_plate.shape[:2]
        ratio = w / max(h, 1)

        if ratio >= 2.2:
            text, conf = self._ocr_best(cropped_plate, force_full=force_full)
            text = format_vn_plate(text)
            return OcrResult(text=text, confidence=conf) if text else None

        split_configs = [
            (0.52, 0.34, 0.08),
        ] if FAST_MODE and not force_full else [
            (0.52, 0.34, 0.08),
            (0.48, 0.30, 0.03),
            (0.56, 0.38, 0.10),
        ]
        best_combo = ("", 0.0)
        last_bottom_rows = []
        for top_ratio, overlap_ratio, top_trim in split_configs:
            top, bottom = self._split_vertical_plate(cropped_plate, top_ratio=top_ratio, overlap_ratio=overlap_ratio, top_trim=top_trim)
            top_rows = self._ocr_candidates(top, force_full=force_full)
            bottom_rows = self._ocr_candidates(bottom, force_full=force_full)
            last_bottom_rows = bottom_rows
            combined_text, combined_conf = self._combine_from_candidates(top_rows, bottom_rows)
            if combined_conf > best_combo[1]:
                best_combo = (combined_text, combined_conf)
        if best_combo[0] and (force_full or not FAST_MODE or not self._needs_rescue_ocr(best_combo[0], best_combo[1], ratio, bool(last_bottom_rows))):
            return OcrResult(text=best_combo[0], confidence=best_combo[1])

        full_text, full_conf = self._ocr_best(cropped_plate, force_full=force_full)
        full_text = format_vn_plate(full_text)
        if full_text and not (is_top_only_plate(full_text) and last_bottom_rows and any(looks_like_vn_bottom(t) for t, _ in last_bottom_rows[:4])):
            if force_full or not FAST_MODE or not self._needs_rescue_ocr(full_text, full_conf, ratio, bool(last_bottom_rows)):
                return OcrResult(text=full_text, confidence=full_conf)

        top, bottom = self._split_vertical_plate(cropped_plate, top_ratio=0.50, overlap_ratio=0.28, top_trim=0.0)
        extra_top_rows = self._ocr_candidates(top, force_full=force_full) + self._ocr_candidates(cv2.resize(top, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC), force_full=force_full)
        extra_bottom_rows = self._ocr_candidates(bottom, force_full=force_full) + self._ocr_candidates(cv2.resize(bottom, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC), force_full=force_full)
        combined_text, combined_conf = self._combine_from_candidates(extra_top_rows, extra_bottom_rows)
        if combined_text:
            return OcrResult(text=combined_text, confidence=combined_conf)

        return OcrResult(text=full_text, confidence=full_conf) if full_text else None

    def _result_payload_score(self, result, ratio: float = 0.0) -> tuple[int, float, int]:
        if result is None:
            return (0, 0.0, 0)
        text = getattr(result, "text", "") or ""
        conf = float(getattr(result, "confidence", 0.0) or 0.0)
        assessment = assess_ocr_quality(text, conf)
        penalty = 1 if self._needs_rescue_ocr(text, conf, ratio, False) else 0
        return (1 if assessment.is_usable else 0, conf - penalty, len(assessment.normalized_text or text))

    def predict(self, cropped_plate):
        if cropped_plate is None:
            return None
        h, w = cropped_plate.shape[:2]
        ratio = w / max(h, 1)
        fast_result = self._predict_core(cropped_plate, force_full=False)
        if not ENABLE_DEEP_OCR_RETRY or not FAST_MODE:
            return fast_result
        fast_text = getattr(fast_result, "text", "") if fast_result is not None else ""
        fast_conf = float(getattr(fast_result, "confidence", 0.0) or 0.0) if fast_result is not None else 0.0
        if not self._needs_rescue_ocr(fast_text, fast_conf, ratio, False):
            return fast_result
        deep_result = self._predict_core(cropped_plate, force_full=True)
        return max([item for item in [fast_result, deep_result] if item is not None], key=lambda item: self._result_payload_score(item, ratio), default=fast_result)


def get_detector_choices(external_model_dir: Path):
    choices = [(DEFAULT_MODEL_LABEL, None)]
    for model_path in sorted(external_model_dir.glob("*.pt")):
        choices.append((f"External: {model_path.name}", str(model_path)))
    return choices


_ALPR_INSTANCE_CACHE = {}
_PADDLE_OCR_READER = None


def _score_ocr_payload(payload: dict) -> tuple[int, float, int]:
    text = (payload.get("text") or "").strip()
    assessment = assess_ocr_quality(text or payload.get("raw_text") or "", float(payload.get("confidence", 0.0) or 0.0))
    return (
        1 if assessment.is_usable else 0,
        float(payload.get("confidence", 0.0) or 0.0),
        len(assessment.normalized_text or text),
    )


def _build_candidate_payload(result) -> dict | None:
    ocr = getattr(result, "ocr", None)
    detection = getattr(result, "detection", None)
    if ocr is None:
        return None
    raw_text = getattr(ocr, "text", "") or ""
    conf = float(getattr(ocr, "confidence", 0.0) or 0.0)
    detected_type = classify_plate_type(raw_text)
    formatted = format_plate_by_type(raw_text, detected_type)
    assessment = assess_ocr_quality(formatted or raw_text, conf)
    bbox = None
    if detection is not None:
        for attr in ("bbox", "box", "xyxy", "xywh"):
            value = getattr(detection, attr, None)
            if value is not None:
                try:
                    bbox = list(value)
                except Exception:
                    bbox = str(value)
                break
    return {
        "text": formatted,
        "raw_text": raw_text,
        "confidence": conf,
        "is_usable": assessment.is_usable,
        "quality_note": assessment.quality_note,
        "normalized_text": assessment.normalized_text,
        "formatted_text": assessment.formatted_text,
        "plate_type": assessment.plate_type,
        "bbox": bbox,
        "detector_confidence": float(getattr(detection, "confidence", 0.0) or 0.0) if detection is not None else 0.0,
    }


def _is_good_enough(payload: dict | None) -> bool:
    if not payload:
        return False
    text = (payload.get("text") or "").strip()
    conf = float(payload.get("confidence", 0.0) or 0.0)
    assessment = assess_ocr_quality(text or payload.get("raw_text") or "", conf)
    return bool(assessment.is_usable and len(assessment.normalized_text or text) >= 8 and conf >= 0.80)


def _get_paddle_reader():
    global _PADDLE_OCR_READER
    if PaddleOCR is None or not (ENABLE_PADDLE_OCR or ENABLE_PADDLE_FALLBACK):
        return None
    if _PADDLE_OCR_READER is None:
        init_variants = [
            {"use_angle_cls": False, "lang": "en", "show_log": False},
            {"use_angle_cls": False, "lang": "en"},
            {"lang": "en"},
            {},
        ]
        for kwargs in init_variants:
            try:
                _PADDLE_OCR_READER = PaddleOCR(**kwargs)
                break
            except TypeError:
                continue
            except ValueError:
                continue
            except Exception:
                continue
    return _PADDLE_OCR_READER


def _run_openai_vision_payload(image_path: str, detector_label: str, annotated_path: str):
    if not ENABLE_OPENAI_VISION or run_openai_vision is None:
        return None
    openai_payload = run_openai_vision(image_path)
    if not openai_payload:
        return None
    payload = dict(openai_payload)
    payload.update({
        "message": f"{detector_label} • OpenAI vision • {payload.get('quality_note', '')}",
        "used_ocr": bool(payload.get("text")),
        "annotated_path": annotated_path,
        "detector_label": detector_label,
        "ocr_engine": "openai-vision",
    })
    return payload


def _payload_identity(payload: dict | None) -> str:
    if not payload:
        return ""
    assessment = assess_ocr_quality(payload.get("text") or payload.get("raw_text") or "", float(payload.get("confidence", 0.0) or 0.0))
    return assessment.normalized_text or normalize_plate_text(payload.get("text") or payload.get("raw_text") or "")


def _candidate_conflict(a: dict | None, b: dict | None) -> bool:
    a_norm = _payload_identity(a)
    b_norm = _payload_identity(b)
    return bool(a_norm and b_norm and a_norm != b_norm)


def _fusion_rank(payload: dict) -> tuple[int, float, int, int]:
    text = payload.get("text") or payload.get("raw_text") or ""
    conf = float(payload.get("confidence", 0.0) or 0.0)
    assessment = assess_ocr_quality(text, conf)
    engine = payload.get("ocr_engine") or payload.get("engine") or ""
    engine_bonus = 2 if engine == "openai-vision" else 1 if engine == "paddleocr" else 0
    return (1 if assessment.is_usable else 0, score_full_plate(text, conf), len(assessment.normalized_text or text), engine_bonus)


def _fuse_ocr_candidates(candidates: list[dict | None], detector_label: str, annotated_path: str) -> dict | None:
    payloads = [dict(item) for item in candidates if item and (item.get("text") or item.get("raw_text"))]
    if not payloads:
        return None
    for item in payloads:
        item.setdefault("annotated_path", annotated_path)
        item.setdefault("detector_label", detector_label)
    selected = max(payloads, key=_fusion_rank)
    identities = {_payload_identity(item) for item in payloads if _payload_identity(item)}
    selected = dict(selected)
    selected["candidates"] = payloads
    selected["fusion_sources"] = [item.get("ocr_engine") or item.get("engine") or "unknown" for item in payloads]
    selected["vision_disagreement"] = len(identities) > 1 and any((item.get("ocr_engine") == "openai-vision") for item in payloads)
    if selected["vision_disagreement"]:
        selected["quality_note"] = "Các engine OCR không thống nhất, cần đối chiếu ảnh"
        selected["message"] = f"{detector_label} • OCR fusion • bất đồng kết quả giữa các engine"
    else:
        selected["message"] = selected.get("message") or f"{detector_label} • OCR fusion"
    selected["ocr_engine"] = "vision-fusion" if any(item.get("ocr_engine") == "openai-vision" for item in payloads) else selected.get("ocr_engine", "fusion")
    selected["used_ocr"] = bool(selected.get("text"))
    selected["annotated_path"] = annotated_path
    selected["detector_label"] = detector_label
    return selected


def _run_paddle_ocr(image_path: str, detector_label: str, annotated_path: str):
    reader = _get_paddle_reader()
    if reader is None:
        return None

    raw_result = None
    try:
        if hasattr(reader, "ocr"):
            raw_result = reader.ocr(image_path, cls=True) or []
        elif hasattr(reader, "predict"):
            raw_result = reader.predict(input=image_path) or []
        else:
            return None
    except TypeError:
        try:
            if hasattr(reader, "ocr"):
                raw_result = reader.ocr(image_path) or []
            else:
                return None
        except Exception:
            return None
    except Exception:
        return None

    candidates = []

    if isinstance(raw_result, list):
        for item in raw_result:
            if isinstance(item, dict):
                rec_texts = item.get("rec_texts") or []
                rec_scores = item.get("rec_scores") or []
                for idx, text in enumerate(rec_texts):
                    raw_text = str(text or "").strip()
                    conf = float(rec_scores[idx] or 0.0) if idx < len(rec_scores) else 0.0
                    formatted = format_vn_plate(raw_text)
                    if raw_text:
                        candidates.append((formatted, raw_text, conf))
                continue

            for row in item or []:
                if not row or len(row) < 2:
                    continue
                text_conf = row[1]
                if not text_conf or len(text_conf) < 2:
                    continue
                raw_text = str(text_conf[0] or "").strip()
                conf = float(text_conf[1] or 0.0)
                formatted = format_vn_plate(raw_text)
                if raw_text:
                    candidates.append((formatted, raw_text, conf))

    if not candidates:
        return None

    best_text, best_raw, best_conf = max(
        candidates,
        key=lambda item: score_full_plate(item[0] or item[1], item[2]),
    )
    assessment = assess_ocr_quality(best_text or best_raw, best_conf)
    return {
        "text": best_text,
        "confidence": best_conf,
        "message": f"{detector_label} • PaddleOCR • {assessment.quality_note}",
        "used_ocr": bool(best_text),
        "annotated_path": annotated_path,
        "detector_label": detector_label,
        "raw_text": best_raw,
        "is_usable": assessment.is_usable,
        "quality_note": assessment.quality_note,
        "normalized_text": assessment.normalized_text,
        "ocr_engine": "paddleocr",
    }


def load_fast_alpr(model_path: str | None = None):
    if ALPR is None or easyocr is None:
        return None
    cache_key = model_path or "__default__"
    if cache_key in _ALPR_INSTANCE_CACHE:
        return _ALPR_INSTANCE_CACHE[cache_key]
    try:
        kwargs = {"ocr": EasyOCRVNOCR()}
        if model_path:
            kwargs["detector_model"] = model_path
        instance = ALPR(**kwargs)
        _ALPR_INSTANCE_CACHE[cache_key] = instance
        return instance
    except TypeError:
        try:
            kwargs = {"ocr": EasyOCRVNOCR()}
            if model_path:
                kwargs["model_path"] = model_path
            instance = ALPR(**kwargs)
            _ALPR_INSTANCE_CACHE[cache_key] = instance
            return instance
        except Exception:
            return None
    except Exception:
        return None


def _load_runtime_frame(image_path: str):
    frame = cv2.imread(image_path)
    if frame is None:
        return None
    h, w = frame.shape[:2]
    longest_side = max(h, w)
    max_side = 1600 if FAST_MODE else 1920
    if longest_side > max_side:
        scale = max_side / float(longest_side)
        frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return frame


def run_alpr(image_path: str, detector_model: str | None = None, ocr_engine: str = "auto"):
    detector_label = DEFAULT_MODEL_LABEL if not detector_model else Path(detector_model).name
    if cv2 is None:
        return {"text": "", "confidence": 0.0, "message": "OpenCV unavailable", "used_ocr": False, "annotated_path": image_path, "detector_label": detector_label}

    frame = _load_runtime_frame(image_path)
    if frame is None:
        return {"text": "", "confidence": 0.0, "message": "Invalid image", "used_ocr": False, "annotated_path": image_path, "detector_label": detector_label}

    alpr = load_fast_alpr(detector_model)
    results = []
    annotated_path = image_path

    if alpr is not None:
        try:
            drawn = alpr.draw_predictions(frame)
            results = drawn.results or []
            annotated_path = save_annotated_image(cv2, drawn.image, "alpr")
        except Exception:
            try:
                results = alpr.predict(image_path) or []
            except Exception:
                results = []
            annotated_path = image_path

    fast_payload = None
    if results:
        candidates = [payload for payload in (_build_candidate_payload(item) for item in results) if payload]
        best = max(candidates, key=_score_ocr_payload) if candidates else None
        if best:
            det_conf = float(best.get("detector_confidence", 0.0) or 0.0)
            fast_payload = {
                "text": best.get("text", ""),
                "confidence": best.get("confidence", 0.0),
                "message": f"{detector_label} • fast-alpr (det {det_conf:.2f}) • {best.get('quality_note', '')}",
                "used_ocr": bool(best.get("text")),
                "annotated_path": annotated_path,
                "detector_label": detector_label,
                "raw_text": best.get("raw_text", ""),
                "is_usable": bool(best.get("is_usable")),
                "quality_note": best.get("quality_note", ""),
                "normalized_text": best.get("normalized_text", ""),
                "ocr_engine": "fast-alpr-easyocr",
                "candidates": candidates,
            }

    requested_engine = (ocr_engine or "auto").strip().lower()
    if requested_engine == "paddleocr":
        paddle_payload = _run_paddle_ocr(image_path, detector_label, annotated_path)
        if paddle_payload:
            return paddle_payload
        if fast_payload:
            return fast_payload
    elif requested_engine == "openai-vision":
        try:
            openai_payload = _run_openai_vision_payload(image_path, detector_label, annotated_path)
            if openai_payload:
                return openai_payload
        except Exception as exc:
            raise RuntimeError(f"OpenAI vision failed: {exc}") from exc
        if fast_payload:
            fast_payload = dict(fast_payload)
            fast_payload["message"] = f"OpenAI vision unavailable; fallback result: {fast_payload.get('message', '')}"
            fast_payload["ocr_engine"] = "openai-vision-unavailable"
            return fast_payload
    elif requested_engine in {"easyocr", "fast-alpr", "fast-alpr-easyocr"}:
        if fast_payload:
            return fast_payload
        paddle_payload = _run_paddle_ocr(image_path, detector_label, annotated_path)
        if paddle_payload:
            return paddle_payload
    else:
        if _is_good_enough(fast_payload) and not OCR_ACCURACY_MODE:
            return fast_payload
        paddle_payload = None
        if ENABLE_PADDLE_FALLBACK or ENABLE_PADDLE_OCR:
            paddle_payload = _run_paddle_ocr(image_path, detector_label, annotated_path)
        vision_payload = None
        if ENABLE_OPENAI_VISION and ENABLE_VISION_IN_AUTO and OCR_ACCURACY_MODE:
            try:
                vision_payload = _run_openai_vision_payload(image_path, detector_label, annotated_path)
            except Exception:
                vision_payload = None
        fused_payload = _fuse_ocr_candidates([fast_payload, paddle_payload, vision_payload], detector_label, annotated_path)
        if fused_payload:
            return fused_payload

    missing = []
    if ALPR is None:
        missing.append("fast-alpr")
    if easyocr is None:
        missing.append("easyocr")
    if PaddleOCR is None:
        missing.append("paddleocr")
    return {
        "text": "",
        "confidence": 0.0,
        "message": (f"Thiếu {', '.join(missing)}" if missing else "OCR failed"),
        "used_ocr": False,
        "annotated_path": annotated_path,
        "detector_label": detector_label,
        "raw_text": "",
        "is_usable": False,
        "quality_note": "Không có kết quả OCR",
        "ocr_engine": "unavailable",
    }

