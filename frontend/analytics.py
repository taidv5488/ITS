import streamlit as st

def render_analytics_dashboard(df_logs=None):
    st.title("📊 Thống kê & Phân tích")
    st.write("Giao diện tổng hợp dữ liệu xe ra vào hệ thống.")
    
    if df_logs is not None and not df_logs.empty:
        st.metric(label="Tổng số lượt ghi nhận", value=len(df_logs))
        st.dataframe(df_logs, use_container_width=True)
    else:
        st.info("Chưa có dữ liệu thống kê nào được ghi nhận.")