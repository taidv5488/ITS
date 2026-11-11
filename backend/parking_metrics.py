from dataclasses import dataclass

import pandas as pd

from backend import parking_service


@dataclass(frozen=True)
class ParkingKpis:
    total_tickets: int
    active_tickets: int
    completed_tickets: int
    review_tickets: int
    cancelled_tickets: int
    mismatch_tickets: int
    manual_override_tickets: int
    avg_confidence: float
    review_rate: float
    mismatch_rate: float


def build_parking_kpis(df_logs: pd.DataFrame) -> ParkingKpis:
    if df_logs is None or df_logs.empty:
        return ParkingKpis(0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0)

    total = len(df_logs)
    status = df_logs["status"] if "status" in df_logs.columns else pd.Series([], dtype=str)
    match = df_logs["match_result"] if "match_result" in df_logs.columns else pd.Series([], dtype=str)
    active = int((status == parking_service.ACTIVE_STATUS).sum())
    completed = int((status == parking_service.COMPLETED_STATUS).sum())
    review = int((status == parking_service.NEEDS_REVIEW_STATUS).sum())
    cancelled = int((status == parking_service.CANCELLED_STATUS).sum())
    mismatch = int((match == parking_service.MISMATCH).sum())

    manual_cols = [col for col in ["entry_manual_override", "exit_manual_override"] if col in df_logs.columns]
    manual_override = 0
    if manual_cols:
        manual_override = int((df_logs[manual_cols].fillna(0).astype(int).sum(axis=1) > 0).sum())

    conf_cols = [col for col in ["entry_ocr_confidence", "exit_ocr_confidence", "ocr_confidence"] if col in df_logs.columns]
    avg_conf = 0.0
    if conf_cols:
        values = pd.concat([pd.to_numeric(df_logs[col], errors="coerce") for col in conf_cols]).dropna()
        if not values.empty:
            avg_conf = float(values.mean())

    return ParkingKpis(
        total_tickets=total,
        active_tickets=active,
        completed_tickets=completed,
        review_tickets=review,
        cancelled_tickets=cancelled,
        mismatch_tickets=mismatch,
        manual_override_tickets=manual_override,
        avg_confidence=avg_conf,
        review_rate=review / total if total else 0.0,
        mismatch_rate=mismatch / total if total else 0.0,
    )


def classify_operational_health(kpis: ParkingKpis) -> tuple[str, str]:
    if kpis.review_tickets > 0 or kpis.mismatch_rate >= 0.15:
        return "ATTENTION_REQUIRED", "warning"
    if kpis.avg_confidence and kpis.avg_confidence < 0.65:
        return "OCR_DEGRADED", "warning"
    return "NORMAL", "success"
