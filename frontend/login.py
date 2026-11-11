import base64
import time
from pathlib import Path

import streamlit as st

from backend.auth import verify_user
from backend.config import APP_TITLE, SHOW_DEMO_HINT

MAX_FAILED_ATTEMPTS = 5
LOCK_SECONDS = 30

LOGIN_CSS = """
<style>
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    .login-brand { text-align: center; margin: 2.5rem 0 1.5rem 0; }
    .login-brand img { width: 88px; height: 88px; object-fit: contain; border-radius: 16px; margin-bottom: 14px; }
    .login-title { font-size: 1.6rem; font-weight: 700; color: #ffffff; line-height: 1.3; }
    .login-sub { margin-top: 6px; color: var(--muted); font-size: 0.95rem; }
    [data-testid="stForm"] {
        background: var(--card); border: 1px solid var(--border);
        border-radius: 16px; padding: 28px 28px 20px 28px; box-shadow: var(--shadow);
    }
    .demo-box {
        margin-top: 18px; padding: 14px 18px; border-radius: 12px;
        background: var(--primary-soft); border: 1px dashed rgba(96, 165, 250, 0.5);
        color: #dbeafe; font-size: 0.92rem; line-height: 1.7;
    }
    .demo-box b { color: #ffffff; }
    .demo-box code {
        font-family: var(--font-mono); background: rgba(15, 23, 42, 0.7);
        padding: 2px 8px; border-radius: 6px; color: #93c5fd; font-size: 0.88rem;
    }
</style>
"""


def _logo_html(logo_path: Path) -> str:
    if logo_path and logo_path.exists():
        data = base64.b64encode(logo_path.read_bytes()).decode()
        return f"<img src='data:image/png;base64,{data}' alt='logo'>"
    return "<div style='font-size:2.5rem'>🅿️</div>"


def render_login_page(logo_path: Path) -> None:
    """Hiển thị form đăng nhập. Khi đúng tài khoản sẽ lưu st.session_state['auth_user'] và rerun."""
    st.markdown(LOGIN_CSS, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 1.15, 1])
    with center:
        st.markdown(
            f"""
            <div class='login-brand'>
                {_logo_html(logo_path)}
                <div class='login-title'>{APP_TITLE}</div>
                <div class='login-sub'>Đăng nhập để vào hệ thống quản lý bãi xe</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        locked_for = int(st.session_state.get("login_locked_until", 0) - time.time())
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Tên đăng nhập", placeholder="Ví dụ: admin")
            password = st.text_input("Mật khẩu", type="password", placeholder="Nhập mật khẩu")
            submitted = st.form_submit_button("Đăng nhập", use_container_width=True)

        if submitted:
            if locked_for > 0:
                st.error(f"Bạn nhập sai quá nhiều lần. Vui lòng thử lại sau {locked_for} giây.")
            elif not username.strip() or not password:
                st.warning("Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu.")
            else:
                user = verify_user(username, password)
                if user:
                    st.session_state["auth_user"] = user
                    st.session_state["login_failures"] = 0
                    st.session_state.pop("login_locked_until", None)
                    st.rerun()
                else:
                    fails = st.session_state.get("login_failures", 0) + 1
                    st.session_state["login_failures"] = fails
                    if fails >= MAX_FAILED_ATTEMPTS:
                        st.session_state["login_locked_until"] = time.time() + LOCK_SECONDS
                        st.session_state["login_failures"] = 0
                        st.error(f"Sai quá {MAX_FAILED_ATTEMPTS} lần. Tạm khóa đăng nhập {LOCK_SECONDS} giây.")
                    else:
                        st.error(f"Sai tên đăng nhập hoặc mật khẩu (lần {fails}/{MAX_FAILED_ATTEMPTS}).")

        if SHOW_DEMO_HINT:
            st.markdown(
                """
                <div class='demo-box'>
                    <b>Tài khoản dùng thử</b><br>
                    Quản trị viên: <code>admin</code> / <code>Admin@123</code><br>
                    Nhân viên bảo vệ: <code>baove</code> / <code>Baove@123</code>
                </div>
                """,
                unsafe_allow_html=True,
            )
