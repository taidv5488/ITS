import re


LETTER_TO_DIGIT = {
    "O": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
}

DIGIT_TO_LETTER = {
    "0": "D",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
}


def normalize_plate_text(text: str):
    if not text:
        return ""
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return "".join(LETTER_TO_DIGIT.get(ch, ch) for ch in text)


def normalize_generic_plate_text(text: str):
    if not text:
        return ""
    text = text.upper()
    return re.sub(r"[^A-Z0-9]", "", text)


def normalize_vn_top(text: str):
    s = (text or "").upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    if len(s) < 3:
        return s
    chars = list(s)
    for i in range(min(2, len(chars))):
        chars[i] = LETTER_TO_DIGIT.get(chars[i], chars[i])
    if len(chars) >= 3 and chars[2].isdigit():
        chars[2] = DIGIT_TO_LETTER.get(chars[2], chars[2])
    # Position 4 can be either the bike numeric series digit (66B1) or the
    # second letter of car/special series (66FH, 30AB). Preserve letters here;
    # digit normalization for the bottom line happens separately.
    if len(chars) >= 4 and chars[3].isdigit() and len(chars) >= 5 and chars[4].isalpha():
        chars[4] = LETTER_TO_DIGIT.get(chars[4], chars[4])
    return "".join(chars)


def _format_top_bottom(top_candidate: str, bottom_candidate: str):
    top_candidate = normalize_vn_top(top_candidate)
    bottom_candidate = normalize_plate_text(bottom_candidate)
    if not top_candidate:
        return ""
    if bottom_candidate:
        if re.fullmatch(r"\d{4}", bottom_candidate):
            return f"{top_candidate[:2]}-{top_candidate[2:]} {bottom_candidate}"
        if re.fullmatch(r"\d{5}", bottom_candidate):
            return f"{top_candidate[:2]}-{top_candidate[2:]} {bottom_candidate[:3]}.{bottom_candidate[3:]}"
    return f"{top_candidate[:2]}-{top_candidate[2:]}"


def _extract_top_bottom_candidates(raw: str):
    raw = (raw or "").upper()
    raw_alnum = re.sub(r"[^A-Z0-9\s.-]", " ", raw)
    line_parts = [p for p in re.split(r"\s+", raw_alnum) if p]
    top_candidate = ""
    bottom_candidate = ""
    for part in line_parts:
        cleaned = part.replace(".", "")
        normalized_top = normalize_vn_top(cleaned)
        if not top_candidate and looks_like_vn_top(normalized_top):
            top_candidate = normalized_top
            continue
        if not bottom_candidate and re.fullmatch(r"\d{2,3}\.?\d{2}", part):
            bottom_candidate = part.replace(".", "")
            continue
        if not bottom_candidate and re.fullmatch(r"\d{4,5}", cleaned):
            bottom_candidate = cleaned
    compact_raw = normalize_generic_plate_text(raw)
    compact_digits = normalize_plate_text(raw)
    if not top_candidate:
        preferred_sizes = (4, 5, 3) if re.match(r"^\d{2}[A-Z]{2}", compact_raw) else (5, 4, 3)
        for size in preferred_sizes:
            candidate = normalize_vn_top(compact_raw[:size])
            if looks_like_vn_top(candidate):
                top_candidate = candidate
                break
    if top_candidate and not bottom_candidate:
        remain_raw = compact_raw[len(top_candidate):]
        remain_digits = compact_digits[len(top_candidate):]
        if re.fullmatch(r"\d{4,5}", remain_raw):
            bottom_candidate = remain_raw
        elif re.fullmatch(r"\d{4,5}", remain_digits):
            bottom_candidate = remain_digits
    return top_candidate, bottom_candidate


def classify_plate_type(text: str) -> str:
    raw = (text or "").upper().strip()
    if not raw:
        return "unknown"
    normalized = normalize_plate_text(raw)
    compact_raw = normalize_generic_plate_text(raw)
    raw_lines = [re.sub(r"[^A-Z0-9.-]", "", part.upper()) for part in re.split(r"[\r\n]+", raw) if part.strip()]
    is_multiline = len(raw_lines) >= 2
    has_separators = bool(re.search(r"[\s.-]", raw))

    if not is_multiline and not has_separators:
        if re.fullmatch(r"\d{2}[A-Z]{2}\d{5}", compact_raw):
            return "vn_car_5_one_line"
        if re.fullmatch(r"\d{2}[A-Z]\d{4}", compact_raw):
            return "vn_car_4_one_line"
        if re.fullmatch(r"\d{2}[A-Z]\d\d{5}", compact_raw):
            return "vn_bike_5_one_line"
        if re.fullmatch(r"\d{2}[A-Z]\d\d{4}", compact_raw):
            return "vn_bike_4_one_line"

    top_candidate, bottom_candidate = _extract_top_bottom_candidates(raw)

    if top_candidate and re.fullmatch(r"\d{4}", bottom_candidate or ""):
        if re.fullmatch(r"\d{2}[A-Z]\d", top_candidate):
            return "vn_bike_4_two_line" if is_multiline else "vn_bike_4"
        if re.fullmatch(r"\d{2}[A-Z]{2}", top_candidate) or re.fullmatch(r"\d{2}[A-Z]{2}\d", top_candidate):
            return "vn_car_4_two_line" if is_multiline else "vn_car_4"
        return "vn_legacy_4"

    if top_candidate and re.fullmatch(r"\d{5}", bottom_candidate or ""):
        if re.fullmatch(r"\d{2}[A-Z]\d", top_candidate):
            return "vn_bike_5_two_line" if is_multiline else "vn_bike_5"
        if re.fullmatch(r"\d{2}[A-Z]{2}", top_candidate) or re.fullmatch(r"\d{2}[A-Z]{2}\d", top_candidate):
            return "vn_car_5_two_line" if is_multiline else "vn_car_5"
        return "vn_modern_5"

    if looks_like_vn_top(normalized):
        return "vn_partial_top"
    if looks_like_vn_bottom(normalized):
        return "vn_partial_bottom"
    if re.fullmatch(r"[A-Z0-9]{6,12}", normalized):
        return "generic_unknown"
    return "unknown"


def format_plate_by_type(text: str, plate_type: str | None = None) -> str:
    raw = (text or "").upper().strip()
    if not raw:
        return ""
    kind = plate_type or classify_plate_type(raw)
    raw_lines = [re.sub(r"[^A-Z0-9.-]", "", part.upper()) for part in re.split(r"[\r\n]+", raw) if part.strip()]
    if kind.startswith("vn_"):
        if len(raw_lines) >= 2:
            top_line = normalize_vn_top(raw_lines[0].replace("-", ""))
            bottom_line = raw_lines[1].replace(".", "")
            formatted = _format_top_bottom(top_line, bottom_line)
            if formatted and normalize_plate_text(formatted) != normalize_plate_text(top_line):
                return formatted
        top_candidate, bottom_candidate = _extract_top_bottom_candidates(raw)
        if top_candidate:
            return _format_top_bottom(top_candidate, bottom_candidate)
    if kind == "generic_unknown":
        return normalize_generic_plate_text(raw)
    return normalize_plate_text(raw)


def format_vn_plate(text: str):
    return format_plate_by_type(text)


def combine_vn_two_line(top_text: str, bottom_text: str):
    top = normalize_vn_top(top_text)
    bottom = normalize_plate_text(bottom_text)
    top_patterns = (
        r"\d{2}[A-Z]\d",
        r"\d{2}[A-Z]{2}",
        r"\d{2}[A-Z]{2}\d",
    )
    for pattern in top_patterns:
        if re.fullmatch(pattern, top):
            if re.fullmatch(r"\d{4}", bottom):
                return f"{top[:2]}-{top[2:]} {bottom}"
            if re.fullmatch(r"\d{5}", bottom):
                return f"{top[:2]}-{top[2:]} {bottom[:3]}.{bottom[3:]}"
    return ""


def candidate_bottoms(bottom_text: str):
    raw = (bottom_text or "").upper()
    s = normalize_plate_text(raw)
    if not s:
        return []
    candidates = [s]
    if len(s) == 4:
        candidates.append(s[:3] + "0" + s[3:])
        candidates.append(s[:2] + "0" + s[2:])
    if len(s) == 5 and s[0] == "1":
        candidates.append("5" + s[1:])
    if len(s) >= 1:
        swap_map = {"1": ["5", "7"], "5": ["1", "6"], "6": ["5", "8"], "8": ["3", "6"], "3": ["8"]}
        first = s[0]
        for rep in swap_map.get(first, []):
            candidates.append(rep + s[1:])
    for ch, rep in LETTER_TO_DIGIT.items():
        if ch in raw:
            candidates.append(normalize_plate_text(raw.replace(ch, rep)))
    if len(s) == 4:
        candidates.extend(["0" + s, s + "0"])
    return list(dict.fromkeys(candidates))


def looks_like_vn_top(text: str):
    s = normalize_vn_top(text)
    return bool(re.fullmatch(r"\d{2}[A-Z]", s) or re.fullmatch(r"\d{2}[A-Z]{2}", s) or re.fullmatch(r"\d{2}[A-Z]\d", s) or re.fullmatch(r"\d{2}[A-Z]{2}\d", s))


def looks_like_vn_bottom(text: str):
    s = normalize_plate_text(text)
    return bool(re.fullmatch(r"\d{4}", s) or re.fullmatch(r"\d{5}", s) or re.fullmatch(r"\d{3}\d{2}", s))


def is_top_only_plate(text: str):
    s = (text or "").strip().upper()
    normalized = normalize_vn_top(s)
    return bool(
        re.fullmatch(r"\d{2}-[A-Z]\d", s)
        or re.fullmatch(r"\d{2}[A-Z]\d", normalized)
        or re.fullmatch(r"\d{2}[A-Z]{2}", normalized)
        or re.fullmatch(r"\d{2}[A-Z]{2}\d", normalized)
    )
