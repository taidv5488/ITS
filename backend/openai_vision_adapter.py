import base64
from pathlib import Path

from backend.config import ENABLE_OPENAI_VISION, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_VISION_MODEL
from backend.logging_config import get_logger
from backend.ocr_quality import assess_ocr_quality
from backend.plate_format import format_vn_plate

logger = get_logger(__name__)

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


class OpenAIVisionAdapter:
    def __init__(self) -> None:
        if not ENABLE_OPENAI_VISION:
            raise RuntimeError("OpenAI vision is disabled")
        if OpenAI is None:
            raise RuntimeError("openai sdk is not installed")
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        client_kwargs = {"api_key": OPENAI_API_KEY}
        if OPENAI_BASE_URL:
            client_kwargs["base_url"] = OPENAI_BASE_URL
        self.client = OpenAI(**client_kwargs)
        self.model = OPENAI_VISION_MODEL

    def predict_image(self, image_path: str) -> dict:
        image_bytes = Path(image_path).read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Read the full Vietnamese license plate from this image, including both the top line and bottom line if it is a two-line motorcycle plate. Return only one plate string in Vietnamese format, for example: 69-F1 023.41 or 66-B1 069.99. Do not explain. Do not return only the top line.",
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    ],
                }
            ],
        )
        raw_text = (getattr(response, "output_text", "") or "").strip()
        formatted = format_vn_plate(raw_text)
        confidence = 0.80 if formatted else 0.0
        assessment = assess_ocr_quality(formatted or raw_text, confidence)
        return {
            "engine": "openai-vision",
            "text": formatted,
            "raw_text": raw_text,
            "confidence": confidence,
            "is_usable": assessment.is_usable,
            "quality_note": assessment.quality_note,
            "normalized_text": assessment.normalized_text,
            "formatted_text": assessment.formatted_text,
        }


def run_openai_vision(image_path: str) -> dict:
    adapter = OpenAIVisionAdapter()
    return adapter.predict_image(image_path)
