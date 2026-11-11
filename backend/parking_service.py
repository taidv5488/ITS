import json
import sqlite3
from datetime import datetime
from typing import Any
from uuid import uuid4

from backend.config import DB_PATH
from backend.db import append_audit_log, append_parking_event
from backend.plate_format import format_vn_plate, normalize_plate_text


ACTIVE_STATUS = "IN_PARKING"
COMPLETED_STATUS = "COMPLETED"
NEEDS_REVIEW_STATUS = "NEEDS_REVIEW"
CANCELLED_STATUS = "CANCELLED"
PENDING_MATCH = "PENDING"
MATCH = "MATCH"
MISMATCH = "MISMATCH"
REVIEW_RESOLVED_MATCH = "RESOLVED_MATCH"
REVIEW_RESOLVED_APPROVED = "RESOLVED_APPROVED"
REVIEW_RESOLVED_REJECTED = "RESOLVED_REJECTED"
BLOCKING_ACTIVE_STATUSES = {ACTIVE_STATUS, NEEDS_REVIEW_STATUS}
FINAL_STATUSES = {COMPLETED_STATUS, CANCELLED_STATUS}
LOW_CONFIDENCE_THRESHOLD = 0.55


def create_ticket_id() -> str:
    return f"T{datetime.now().strftime('%Y%m%d%H%M%S')}{str(uuid4())[:4].upper()}"


def _utc_like_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _sanitize_plate_text(plate_text: str | None) -> str:
    return format_vn_plate(plate_text or "")


def _normalized_plate_text(plate_text: str | None) -> str:
    return normalize_plate_text(plate_text or "")


def _safe_confidence(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric < 0:
        return 0.0
    if numeric > 1:
        return 1.0
    return numeric


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def _actor_name(actor: Any | None) -> str | None:
    if actor is None:
        return None
    if isinstance(actor, str):
        return actor.strip() or None
    return getattr(actor, "username", None)


def _actor_role(actor: Any | None) -> str | None:
    if actor is None or isinstance(actor, str):
        return None
    return getattr(actor, "role", None)


def _safe_event(ticket_id: str | None, event_type: str, **kwargs):
    try:
        append_parking_event(ticket_id, event_type, **kwargs)
    except Exception:
        pass


def _safe_audit(actor_username: str | None, actor_role: str | None, action: str, **kwargs):
    try:
        append_audit_log(actor_username, actor_role, action, **kwargs)
    except Exception:
        pass


def _json_details(**details) -> str:
    return json.dumps({k: v for k, v in details.items() if v is not None}, ensure_ascii=False)


def _review_reason(confidence: float | None, manual_override: bool, mismatch: bool = False) -> str | None:
    reasons: list[str] = []
    if mismatch:
        reasons.append("PLATE_MISMATCH")
    if confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD and not manual_override:
        reasons.append("LOW_OCR_CONFIDENCE")
    return ";".join(reasons) or None


def vehicle_entry(plate_text: str, image_path: str | None = None, detected_path: str | None = None, plate_path: str | None = None, ocr_confidence: float | None = None, ocr_engine: str | None = None, ocr_text: str | None = None, manual_override: bool = False, actor: Any | None = None) -> dict[str, Any]:
    clean_plate = _sanitize_plate_text(plate_text)
    normalized_plate = _normalized_plate_text(clean_plate)
    if not normalized_plate:
        return {"success": False, "message": "Invalid plate text", "plate_entry": clean_plate}

    confidence = _safe_confidence(ocr_confidence)
    review_reason = _review_reason(confidence, manual_override)
    review_required = bool(review_reason)

    with _connect() as conn:
        cursor = conn.cursor()
        placeholders = ", ".join(["?"] * len(BLOCKING_ACTIVE_STATUSES))
        cursor.execute(
            f"SELECT ticket_id, plate_entry, entry_time, status FROM parking_logs WHERE status IN ({placeholders}) ORDER BY entry_time DESC",
            tuple(BLOCKING_ACTIVE_STATUSES),
        )
        existing = None
        for active_ticket_id, active_plate_entry, active_entry_time, status in cursor.fetchall():
            if _normalized_plate_text(active_plate_entry) == normalized_plate:
                existing = (active_ticket_id, active_entry_time, status)
                break
        if existing is not None:
            return {
                "success": False,
                "message": "Vehicle already has an active ticket",
                "ticket_id": existing[0],
                "plate_entry": clean_plate,
                "entry_time": existing[1],
                "status": existing[2],
            }

        ticket_id = create_ticket_id()
        entry_time = _utc_like_timestamp()
        existing_columns = {row[1] for row in cursor.execute("PRAGMA table_info(parking_logs)").fetchall()}
        required_columns = {
            "review_required": "INTEGER NOT NULL DEFAULT 0",
            "review_reason": "TEXT",
            "resolved_by": "TEXT",
            "resolved_at": "TEXT",
            "resolution_note": "TEXT",
            "cancel_reason": "TEXT",
            "entry_actor": "TEXT",
            "exit_actor": "TEXT",
        }
        for column_name, column_type in required_columns.items():
            if column_name not in existing_columns:
                cursor.execute(f"ALTER TABLE parking_logs ADD COLUMN {column_name} {column_type}")
        existing_columns = {row[1] for row in cursor.execute("PRAGMA table_info(parking_logs)").fetchall()}
        insert_columns = [
            "ticket_id", "plate_entry", "plate_exit", "entry_time", "exit_time", "status", "match_result",
            "entry_image_path", "entry_detected_path", "entry_plate_path", "exit_image_path", "exit_detected_path", "exit_plate_path",
            "entry_ocr_confidence", "exit_ocr_confidence", "entry_ocr_engine", "exit_ocr_engine", "entry_ocr_text", "exit_ocr_text",
            "entry_manual_override", "exit_manual_override", "review_required", "review_reason", "entry_actor",
        ]
        insert_values = [
            ticket_id, clean_plate, None, entry_time, None, ACTIVE_STATUS, PENDING_MATCH,
            image_path, detected_path, plate_path, None, None, None,
            confidence, None, ocr_engine, None, (ocr_text or "").strip() or None, None,
            1 if manual_override else 0, 0, 1 if review_required else 0, review_reason, _actor_name(actor),
        ]
        if "plate_number" in existing_columns and "plate_number" not in insert_columns:
            insert_columns.insert(1, "plate_number")
            insert_values.insert(1, clean_plate)
        if "time_in" in existing_columns and "time_in" not in insert_columns:
            time_index = insert_columns.index("entry_time")
            insert_columns.insert(time_index + 1, "time_in")
            insert_values.insert(time_index + 1, entry_time)
        placeholders = ", ".join(["?"] * len(insert_columns))
        columns_sql = ", ".join(insert_columns)
        cursor.execute(f"INSERT INTO parking_logs ({columns_sql}) VALUES ({placeholders})", tuple(insert_values))

    _safe_event(ticket_id, "VEHICLE_ENTRY", actor_username=_actor_name(actor), actor_role=_actor_role(actor), plate_text=clean_plate, confidence=confidence, source=ocr_engine or "manual", details={"manual_override": manual_override, "review_reason": review_reason}, db_path=DB_PATH)
    _safe_audit(_actor_name(actor), _actor_role(actor), "vehicle.entry", target_type="ticket", target_id=ticket_id, details_json=_json_details(plate=clean_plate, review_reason=review_reason), db_path=DB_PATH)
    return {"success": True, "ticket_id": ticket_id, "plate_entry": clean_plate, "entry_time": entry_time, "status": ACTIVE_STATUS, "review_required": review_required, "review_reason": review_reason}


def vehicle_exit(ticket_id: str, plate_text: str, image_path: str | None = None, detected_path: str | None = None, plate_path: str | None = None, ocr_confidence: float | None = None, ocr_engine: str | None = None, ocr_text: str | None = None, manual_override: bool = False, actor: Any | None = None) -> dict[str, Any]:
    clean_ticket_id = (ticket_id or "").strip()
    if not clean_ticket_id:
        return {"success": False, "message": "Ticket ID is required"}

    clean_plate = _sanitize_plate_text(plate_text)
    if not _normalized_plate_text(clean_plate):
        return {"success": False, "message": "Invalid plate text", "plate_exit": clean_plate}

    confidence = _safe_confidence(ocr_confidence)
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT ticket_id, plate_entry, status FROM parking_logs WHERE ticket_id = ?", (clean_ticket_id,))
        row = cursor.fetchone()
        if row is None:
            return {"success": False, "message": "Ticket ID not found"}

        _, plate_entry, status = row
        if status == COMPLETED_STATUS:
            return {"success": False, "message": "Vehicle already checked out"}
        if status == CANCELLED_STATUS:
            return {"success": False, "message": "Ticket is cancelled"}
        if status == NEEDS_REVIEW_STATUS:
            return {"success": False, "message": "Ticket requires review before checkout can continue"}

        exit_time = _utc_like_timestamp()
        is_match = _normalized_plate_text(plate_entry) == _normalized_plate_text(clean_plate)
        match_result = MATCH if is_match else MISMATCH
        review_reason = _review_reason(confidence, manual_override, mismatch=not is_match)
        review_required = bool(review_reason)
        new_status = NEEDS_REVIEW_STATUS if review_required else COMPLETED_STATUS
        cursor.execute(
            """
            UPDATE parking_logs
            SET plate_exit = ?, exit_time = ?, status = ?, match_result = ?,
                exit_image_path = ?, exit_detected_path = ?, exit_plate_path = ?, exit_ocr_confidence = ?, exit_ocr_engine = ?,
                exit_ocr_text = ?, exit_manual_override = ?, review_required = ?, review_reason = ?, exit_actor = ?
            WHERE ticket_id = ? AND status = ?
            """,
            (clean_plate, exit_time, new_status, match_result, image_path, detected_path, plate_path, confidence, ocr_engine, (ocr_text or "").strip() or None, 1 if manual_override else 0, 1 if review_required else 0, review_reason, _actor_name(actor), clean_ticket_id, ACTIVE_STATUS),
        )
        if cursor.rowcount == 0:
            return {"success": False, "message": "Vehicle already checked out"}

    _safe_event(clean_ticket_id, "VEHICLE_EXIT", actor_username=_actor_name(actor), actor_role=_actor_role(actor), plate_text=clean_plate, confidence=confidence, source=ocr_engine or "manual", details={"manual_override": manual_override, "match_result": match_result, "review_reason": review_reason}, db_path=DB_PATH)
    _safe_audit(_actor_name(actor), _actor_role(actor), "vehicle.exit", target_type="ticket", target_id=clean_ticket_id, details_json=_json_details(plate_entry=plate_entry, plate_exit=clean_plate, match_result=match_result, status=new_status), db_path=DB_PATH)
    return {"success": True, "ticket_id": clean_ticket_id, "plate_entry": plate_entry, "plate_exit": clean_plate, "exit_time": exit_time, "match_result": match_result, "status": new_status, "review_required": review_required, "review_reason": review_reason}


def resolve_review_ticket(ticket_id: str, resolution: str, actor: Any | None = None, note: str | None = None) -> dict[str, Any]:
    clean_ticket_id = (ticket_id or "").strip()
    clean_resolution = (resolution or "").strip().upper()
    if clean_resolution not in {"APPROVE_EXIT", "REJECT_EXIT", "MARK_MATCH"}:
        return {"success": False, "message": "Invalid resolution"}

    new_status = COMPLETED_STATUS if clean_resolution in {"APPROVE_EXIT", "MARK_MATCH"} else NEEDS_REVIEW_STATUS
    match_result = REVIEW_RESOLVED_MATCH if clean_resolution == "MARK_MATCH" else (REVIEW_RESOLVED_APPROVED if clean_resolution == "APPROVE_EXIT" else REVIEW_RESOLVED_REJECTED)
    resolved_at = _utc_like_timestamp()
    with _connect() as conn:
        cursor = conn.cursor()
        row = cursor.execute("SELECT status FROM parking_logs WHERE ticket_id = ?", (clean_ticket_id,)).fetchone()
        if row is None:
            return {"success": False, "message": "Ticket ID not found"}
        if row[0] != NEEDS_REVIEW_STATUS:
            return {"success": False, "message": "Ticket does not require review"}
        cursor.execute(
            """
            UPDATE parking_logs
            SET status = ?, match_result = ?, review_required = ?, resolved_by = ?, resolved_at = ?, resolution_note = ?
            WHERE ticket_id = ? AND status = ?
            """,
            (new_status, match_result, 0 if new_status == COMPLETED_STATUS else 1, _actor_name(actor), resolved_at, (note or "").strip() or None, clean_ticket_id, NEEDS_REVIEW_STATUS),
        )
    _safe_event(clean_ticket_id, "REVIEW_RESOLVED", actor_username=_actor_name(actor), actor_role=_actor_role(actor), details={"resolution": clean_resolution, "note": note}, db_path=DB_PATH)
    _safe_audit(_actor_name(actor), _actor_role(actor), "ticket.resolve", target_type="ticket", target_id=clean_ticket_id, details_json=_json_details(resolution=clean_resolution, note=note), db_path=DB_PATH)
    return {"success": True, "ticket_id": clean_ticket_id, "status": new_status, "match_result": match_result, "resolved_at": resolved_at}


def cancel_ticket(ticket_id: str, actor: Any | None = None, reason: str | None = None) -> dict[str, Any]:
    clean_ticket_id = (ticket_id or "").strip()
    clean_reason = (reason or "").strip()
    if not clean_reason:
        return {"success": False, "message": "Cancel reason is required"}
    with _connect() as conn:
        cursor = conn.cursor()
        row = cursor.execute("SELECT status FROM parking_logs WHERE ticket_id = ?", (clean_ticket_id,)).fetchone()
        if row is None:
            return {"success": False, "message": "Ticket ID not found"}
        if row[0] in FINAL_STATUSES:
            return {"success": False, "message": "Finalized ticket cannot be cancelled"}
        cursor.execute(
            "UPDATE parking_logs SET status = ?, cancel_reason = ?, review_required = 0 WHERE ticket_id = ?",
            (CANCELLED_STATUS, clean_reason, clean_ticket_id),
        )
    _safe_event(clean_ticket_id, "TICKET_CANCELLED", actor_username=_actor_name(actor), actor_role=_actor_role(actor), details={"reason": clean_reason}, db_path=DB_PATH)
    _safe_audit(_actor_name(actor), _actor_role(actor), "ticket.cancel", target_type="ticket", target_id=clean_ticket_id, details_json=_json_details(reason=clean_reason), db_path=DB_PATH)
    return {"success": True, "ticket_id": clean_ticket_id, "status": CANCELLED_STATUS}


def get_ticket_detail(ticket_id: str) -> dict | None:
    from backend.db import fetch_ticket_detail

    return fetch_ticket_detail(ticket_id, db_path=DB_PATH)
