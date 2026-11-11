# Backend Module

## Responsibility

The `backend/` module contains core application logic, business rules, OCR/CV orchestration, security, validation, persistence, and production readiness checks.

## Layering

```text
app_service.py
  -> parking_service.py
  -> db.py
  -> alpr_service.py / OCR adapters
  -> validators.py
```

## Important Files

| File | Purpose |
|---|---|
| `app_service.py` | Application-level use cases for processing images and confirming entry/exit |
| `parking_service.py` | Ticket lifecycle and business rules |
| `db.py` | SQLite schema and persistence helpers |
| `auth.py` | Users, passwords, session tokens, RBAC |
| `alpr_service.py` | Detection/OCR orchestration and fusion |
| `openai_vision_adapter.py` | 9Router/OpenAI-compatible Vision adapter |
| `plate_format.py` | Vietnamese plate normalization/format support |
| `ocr_quality.py` | OCR quality assessment and uncertainty labels |
| `validators.py` | Input validation |
| `production_readiness.py` | Production readiness checks |

## Business Rule Boundary

All ticket state changes should go through `parking_service.py` or `app_service.py`.

Do not update ticket status directly from UI/API code.

## OCR Policy

Primary OCR:

```text
9Router -> cx/gpt-5.5 Vision
```

Fallbacks:

- fast-alpr/EasyOCR.
- PaddleOCR.
- local format normalization.

OCR result policy:

- `is_usable` means format usable, not guaranteed true.
- disagreement must be surfaced, not hidden.
- manual override should be recorded.

## Security Policy

- PBKDF2 password hashing.
- Strong JWT secret in production.
- Static bearer tokens disabled in production.
- Role checks must be centralized via auth helpers.
