from dataclasses import dataclass


@dataclass(frozen=True)
class OcrEngineStatus:
    available: bool
    name: str
    detail: str


def get_easyocr_status() -> OcrEngineStatus:
    try:
        import easyocr  # noqa: F401
        return OcrEngineStatus(True, "easyocr", "installed")
    except Exception as exc:
        return OcrEngineStatus(False, "easyocr", str(exc))


def get_paddleocr_status() -> OcrEngineStatus:
    try:
        import paddleocr  # noqa: F401
        return OcrEngineStatus(True, "paddleocr", "installed")
    except Exception as exc:
        return OcrEngineStatus(False, "paddleocr", str(exc))


def get_openai_vision_status() -> OcrEngineStatus:
    try:
        import openai  # noqa: F401
        return OcrEngineStatus(True, "openai-vision", "sdk-installed")
    except Exception as exc:
        return OcrEngineStatus(False, "openai-vision", str(exc))


def list_ocr_engine_statuses() -> list[OcrEngineStatus]:
    return [get_easyocr_status(), get_paddleocr_status(), get_openai_vision_status()]
