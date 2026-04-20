import streamlit as st
import plotly.express as px
from segmentation import load_and_clean, build_rfm, apply_kmeans

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🛍️ E-Commerce Customer Segmentation")
st.markdown("Using **RFM Analysis + K-Means Clustering** to identify customer groups")

# Load data
df  = load_and_clean(r"data\OnlineRetail.csv")
rfm = build_rfm(df)
rfm = apply_kmeans(rfm)

# --- KPI Cards ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", len(rfm))
col2.metric("Segments Found", rfm['Segment'].nunique())
col3.metric("Avg Monetary Value", f"${rfm['Monetary'].mean():.0f}")

# --- Segment Distribution ---
st.subheader("Segment Distribution")
fig1 = px.pie(rfm, names='Segment', title='Customer Segments')
st.plotly_chart(fig1, use_container_width=True)

# --- 3D Scatter Plot ---
st.subheader("RFM Cluster Visualization")
fig2 = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary',
                     color='Segment', hover_data=['CustomerID'])
st.plotly_chart(fig2, use_container_width=True)

# --- Segment Summary Table ---
st.subheader("Segment Profiles")
summary = rfm.groupby('Segment')[['Recency','Frequency','Monetary']].mean().round(1)
st.dataframe(summary, use_container_width=True)