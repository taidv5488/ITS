from pathlib import Path

import pandas as pd
import streamlit as st


SPACING_4 = "4px"
SPACING_8 = "8px"
SPACING_12 = "12px"
SPACING_16 = "16px"
SPACING_24 = "24px"


BASE_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@600;700&display=swap');

    :root {
        --font-sans: 'Inter', 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        --font-mono: 'JetBrains Mono', Consolas, 'Courier New', monospace;
        --bg: #070b14;
        --sidebar: #0f172a;
        --card: #111827;
        --card-soft: #1f2937;
        --border: #334155;
        --text: #f1f5f9;
        --text-soft: #cbd5e1;
        --muted: #a8b3c7;
        --primary: #3b82f6;
        --primary-soft: rgba(59, 130, 246, 0.12);
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        --radius-card: 16px;
    }

    /* Phông chữ thống nhất toàn app (không đụng tới font icon của Streamlit) */
    html { font-size: 16px; }
    .stApp, .stApp p, .stApp li, .stApp a, .stApp label, .stApp input, .stApp textarea,
    .stApp button, .stApp select, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp [data-testid="stMarkdownContainer"] *, .stApp [data-baseweb="select"] *,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        font-family: var(--font-sans) !important;
    }
    .stApp { color: var(--text); line-height: 1.6; -webkit-font-smoothing: antialiased; }
    .stApp h1, .stApp h2, .stApp h3 { font-weight: 700; letter-spacing: -0.01em; color: #fff; }
    .stApp p { font-size: 1rem; }

    [data-testid="stWidgetLabel"] p { font-size: 0.95rem !important; font-weight: 600 !important; color: var(--text-soft) !important; }
    .stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div {
        font-size: 1rem !important; border-radius: 12px !important;
    }
    .stTextInput input { padding: 0.7rem 0.9rem !important; }
    [data-testid="stCaptionContainer"], .stApp small { font-size: 0.88rem !important; color: var(--muted) !important; }
    [data-testid="stSidebar"] [role="radiogroup"] label p { font-size: 1rem !important; font-weight: 500 !important; }
    [data-testid="stSidebar"] h3 { font-size: 1.05rem; font-weight: 700; }
    .stButton > button, .stFormSubmitButton > button {
        font-size: 1rem; font-weight: 600; letter-spacing: 0.01em;
        background: linear-gradient(135deg, var(--primary), #1e40af); color: #fff; border: none;
        border-radius: 12px; padding: 0.7rem 1rem;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover { filter: brightness(1.12); color: #fff; }
    .stDataFrame { border: 1px solid var(--border); border-radius: 12px; }
</style>
"""

EXTRA_CSS = """
<style>
    [data-testid="stSidebarHeader"] button,
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    #MainMenu,
    header button[kind="header"],
    .stAppDeployButton { display: none !important; }
    [data-testid="stSidebarHeader"] > div { display: none !important; }
    [data-testid="stSidebarHeader"]::before {
        content: "🅿️"; display: block; width: 40px; height: 40px; line-height: 40px; text-align: center;
        font-size: 1.2rem; background: #2563eb; color: white; border-radius: 10px; margin: .45rem auto .8rem auto;
    }

    html, body, [data-testid="stAppViewContainer"] { min-height: 100vh; overflow: auto; background-color: var(--bg) !important; }
    [data-testid="stApp"] { background: var(--bg) !important; color: var(--text); }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stAppViewContainer"], [data-testid="stMain"], .stMainBlockContainer { background: transparent !important; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, var(--sidebar), #020617) !important; border-right: 1px solid var(--border); }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    .block-container { max-width: 1400px; padding-top: 1rem; padding-bottom: 2rem; }

    .topbar {
        display: flex; justify-content: space-between; align-items: center;
        background: linear-gradient(90deg, #1e3a8a, #172554); border-radius: var(--radius-card);
        padding: 20px 30px; margin-bottom: 24px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .topbar h1 { margin: 0; padding: 0; color: white; font-size: 1.6rem; font-weight: 700; line-height: 1.3; }
    .topbar .sub { margin-top: 4px; color: #bfdbfe; font-size: 0.95rem; }
    .status-badge {
        background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 6px 14px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.04em;
        animation: pulse 2s infinite; white-space: nowrap;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    .stat-card {
        background: var(--card); border: 1px solid var(--border); border-radius: var(--radius-card);
        padding: 20px; box-shadow: var(--shadow); transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .stat-card:hover { transform: translateY(-3px); border-color: var(--primary); }
    .stat-label { color: var(--muted); font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }
    .stat-value { color: white; font-size: 2rem; font-weight: 700; margin-top: 6px; font-variant-numeric: tabular-nums; }

    .panel-card {
        background: var(--card); border: 1px solid var(--border); border-radius: var(--radius-card);
        padding: 24px; box-shadow: var(--shadow); position: relative; overflow: hidden;
    }
    .panel-card::after {
        content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, transparent, var(--primary), transparent); opacity: 0.5;
    }
    .panel-header { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding-bottom: 16px; margin-bottom: 16px; border-bottom: 1px solid var(--border); }
    .panel-title { color: white; font-size: 1.15rem; font-weight: 700; line-height: 1.4; }
    .panel-subtitle { color: var(--muted); font-size: 0.92rem; margin-top: 2px; line-height: 1.5; }
    .panel-tag { background: var(--primary-soft); color: #93c5fd; padding: 4px 12px; border-radius: 8px; font-size: 0.8rem; font-weight: 600; }

    .preview-label { color: var(--text-soft); font-size: 0.9rem; font-weight: 600; margin-bottom: 6px; }
    .muted { color: var(--muted); font-size: 0.92rem; margin-top: 6px; }
    .ocr-note { color: var(--text-soft); font-size: 0.9rem; margin-top: 4px; }

    .plate-result-wrap { background: rgba(59, 130, 246, 0.06); border: 1px solid var(--primary); border-radius: 12px; padding: 16px; text-align: center; margin-bottom: 14px; }
    .plate-result { color: white; font-family: var(--font-mono) !important; font-size: 2rem; font-weight: 700; letter-spacing: 0.06em; }

    .meta-box { background: var(--card-soft); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; margin: 12px 0; }
    .meta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 14px; }
    .meta-label { color: var(--muted); font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .meta-value { color: #fff; font-size: 1.02rem; font-weight: 600; margin-top: 2px; word-break: break-word; }

    .ticket-box { background: var(--card-soft); border: 1px dashed var(--primary); border-radius: 12px; padding: 16px; text-align: center; margin-top: 12px; }
    .ticket-title { color: var(--muted); font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .ticket-code { color: #fff; font-family: var(--font-mono) !important; font-size: 1.25rem; font-weight: 700; margin-top: 6px; word-break: break-all; }

    .user-chip { background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(96, 165, 250, 0.35); border-radius: 12px; padding: 10px 14px; margin-bottom: 10px; }
    .user-chip .name { font-weight: 700; font-size: 1rem; color: #fff !important; }
    .user-chip .role { font-size: 0.85rem; color: #93c5fd !important; }

    .status-in { color: var(--warning); font-weight: 600; }
    .status-out { color: var(--success); font-weight: 600; }
    .match-ok { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; padding: 2px 8px; border-radius: 4px; }
    .match-err { background: rgba(239, 68, 68, 0.2); color: #fca5a5; padding: 2px 8px; border-radius: 4px; }
</style>
"""


def render_base_css():
    st.markdown(BASE_CSS, unsafe_allow_html=True)


def render_custom_css(app_title: str, background_path: Path, logo_path: Path):
    render_base_css()
    st.markdown(EXTRA_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <h1>{app_title}</h1>
                <div class="sub">Hệ thống giám sát giao thông thông minh - Trung tâm điều khiển</div>
            </div>
            <div class="status-badge">● TRỰC TUYẾN</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_qr_code(data: str):
    import io
    import qrcode
    from PIL import Image

    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    st.image(buf.getvalue(), caption=f"Mã vé: {data}", width=200)


def render_dashboard_shell_start():
    st.markdown("<div class='dashboard-shell'>", unsafe_allow_html=True)


def render_dashboard_shell_end():
    st.markdown("</div>", unsafe_allow_html=True)


def render_section_wrapper_start(class_name: str = "panel-card"):
    st.markdown(f"<div class='{class_name}'>", unsafe_allow_html=True)


def render_section_wrapper_end():
    st.markdown("</div>", unsafe_allow_html=True)


def render_panel_header(title: str, subtitle: str, pill: str | None = None):
    pill_html = f"<div class='panel-tag'>{pill}</div>" if pill else ""
    st.markdown(
        f"""
        <div class='panel-header'>
            <div>
                <div class='panel-title'>{title}</div>
                <div class='panel-subtitle'>{subtitle}</div>
            </div>
            {pill_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_result_images(original_path: str | None, detected_path: str | None):
    cols = st.columns(2, gap="medium")
    for col, (label, img_path) in zip(cols, [("Ảnh gốc", original_path), ("Ảnh nhận diện", detected_path)]):
        with col:
            st.markdown(f"<div class='preview-label'>{label}</div>", unsafe_allow_html=True)
            if img_path and Path(img_path).exists():
                st.image(img_path, use_container_width=True)
            else:
                st.info("Chưa có ảnh hiển thị")


def render_dashboard_metrics(df: pd.DataFrame):
    total_logs = len(df)
    active_count = int((df["status"] == "IN_PARKING").sum()) if not df.empty else 0
    completed_count = int((df["status"] == "COMPLETED").sum()) if not df.empty else 0
    match_count = int((df["match_result"] == "MATCH").sum()) if not df.empty else 0
    cols = st.columns(4, gap="medium")
    for col, (label, value) in zip(cols, [("Tổng lượt", total_logs), ("Trong bãi", active_count), ("Đã ra", completed_count), ("Khớp biển", match_count)]):
        with col:
            st.markdown(f"<div class='stat-card'><div class='stat-label'>{label}</div><div class='stat-value'>{value}</div></div>", unsafe_allow_html=True)


def render_logs_table(df: pd.DataFrame):
    if df.empty:
        st.info("Chưa có dữ liệu")
        return
    display_df = df[["ticket_id", "plate_entry", "plate_exit", "entry_time", "exit_time", "status", "match_result"]].copy()
    display_df.columns = ["Mã vé", "Biển vào", "Biển ra", "Giờ vào", "Giờ ra", "Trạng thái", "Đối chiếu"]
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=440)


def render_ocr_result(ocr: dict):
    text = ocr.get("text") or "Không đọc được"
    confidence = ocr.get("confidence", 0.0) or 0.0
    note = ocr.get("quality_note") or ocr.get("message") or "Chưa có đánh giá chất lượng"
    st.markdown(
        f"""
        <div class='plate-result-wrap'>
            <div class='plate-result'>{text}</div>
            <div class='muted'>Độ tin cậy: {confidence:.2f}</div>
            <div class='ocr-note'>{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_capture_details(*, plate_text: str, captured_at: str, ticket_id: str | None = None):
    ticket_block = ""
    if ticket_id:
        ticket_block = f"<div><div class='meta-label'>Mã vé</div><div class='meta-value'>{ticket_id}</div></div>"
    st.markdown(
        f"""
        <div class='meta-box'>
            <div class='meta-grid'>
                <div><div class='meta-label'>Biển số</div><div class='meta-value'>{plate_text or 'Chưa đọc được'}</div></div>
                <div><div class='meta-label'>Thời gian</div><div class='meta-value'>{captured_at}</div></div>
                {ticket_block}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ticket(ticket_id: str):
    st.markdown(
        f"""
        <div class='ticket-box'>
            <div class='ticket-title'>Mã vé xe</div>
            <div class='ticket-code'>{ticket_id}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
