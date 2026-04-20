# 🛍️ Customer Segmentation using RFM Analysis & K-Means Clustering

## Overview
An end-to-end machine learning project that segments e-commerce customers 
into distinct groups using unsupervised learning.

## Tech Stack
- Python, Pandas, Scikit-learn
- Streamlit (dashboard)
- Plotly (visualizations)

## ML Approach
- **RFM Analysis** to engineer features (Recency, Frequency, Monetary)
- **K-Means Clustering** to group customers
- **Elbow Method** to find optimal number of clusters

## Segments Identified
| Segment | Description |
|---------|-------------|
| 💎 Champions | High value, frequent, recent buyers |
| ⚠️ At Risk | Used to buy often but haven't recently |
| 🌱 New Customers | Bought recently but not often |
| 💤 Hibernating | Low engagement, low spending |

## Dataset
Online Retail Dataset from UCI ML Repository via Kaggle
500K+ transactions from a UK e-commerce store (2010–2011)

## How to Run
pip install -r requirements.txt
streamlit run app.py

## Live Demo
https://customer-segmentation-lprxgdgvkjkfofdpxrvvtb.streamlit.app
