import streamlit as st
import pandas as pd
import plotly.express as px

# === Cấu hình ban đầu ===
st.set_page_config(page_title="Phân tích cảm xúc bình luận Facebook", layout="wide")

# === Tải dữ liệu ===
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/facebook_comments_cleaned_with_sentiment.csv")
    return df

df = load_data()

# === Tiêu đề ===
st.title("Phân tích cảm xúc bình luận Facebook")
st.markdown("Ứng dụng này cho phép bạn phân tích và xem các bình luận theo cảm xúc: positive, neutral, negative.")

# === Thống kê tổng quan ===
st.subheader("Tổng quan cảm xúc")

col1, col2 = st.columns(2)

with col1:
    sentiment_counts = df["sentiment"].value_counts()
    fig = px.pie(values=sentiment_counts.values,
                 names=sentiment_counts.index,
                 title="Tỷ lệ cảm xúc",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.write("### Bảng số liệu")
    sentiment_percent = (sentiment_counts / sentiment_counts.sum() * 100).round(2)
    st.dataframe(pd.DataFrame({
        "Số lượng": sentiment_counts,
        "Tỷ lệ (%)": sentiment_percent
    }))

# === Bộ lọc ===
st.subheader("Lọc bình luận theo cảm xúc")

selected_sentiment = st.selectbox("Chọn loại cảm xúc", options=["Tất cả"] + df["sentiment"].unique().tolist())

if selected_sentiment != "Tất cả":
    filtered_df = df[df["sentiment"] == selected_sentiment]
else:
    filtered_df = df

st.write(f"Hiển thị {len(filtered_df)} bình luận:")
st.dataframe(filtered_df[["comment_text_remove_emojis", "sentiment"]])

