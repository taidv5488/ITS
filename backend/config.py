import os
from pathlib import Path

APP_TITLE = os.getenv("ITS_APP_TITLE", "Hệ thống nhận diện biển số xe máy")
BASE_DIR = Path(os.getenv("ITS_BASE_DIR", str(Path.home() / "plate_demo_data")))
BASE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = BASE_DIR / os.getenv("ITS_DB_NAME", "parking_system.db")
UPLOAD_DIR = BASE_DIR / os.getenv("ITS_UPLOAD_DIR", "uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_MODEL_LABEL = os.getenv("ITS_DEFAULT_MODEL_LABEL", "Baseline detector")
FAST_MODE = os.getenv("ITS_FAST_MODE", "true").lower() in {"1", "true", "yes", "on"}
API_TOKEN = os.getenv("ITS_API_TOKEN", "dev-token")
LOG_LEVEL = os.getenv("ITS_LOG_LEVEL", "INFO")
SCHEMA_VERSION = 1
# Hiện khung "Tài khoản dùng thử" ở trang đăng nhập (đặt ITS_SHOW_DEMO_HINT=false khi triển khai thật)
SHOW_DEMO_HINT = os.getenv("ITS_SHOW_DEMO_HINT", "true").lower() in {"1", "true", "yes", "on"}
ENABLE_HARDCASE_OCR = True  # hoặc False tùy theo nhu cầu của bạn
# Các cấu hình OCR và ALPR bổ sung
DEEP_OCR_CONFIDENCE_THRESHOLD = 0.5
ENABLE_DEEP_OCR_RETRY = True
ENABLE_OPENAI_VISION = False
ENABLE_PADDLE_FALLBACK = True
ENABLE_PADDLE_OCR = True
ENABLE_VISION_IN_AUTO = False
FAST_MODE = False
OCR_ACCURACY_MODE = "balanced"
DEFAULT_MODEL_LABEL = "default"