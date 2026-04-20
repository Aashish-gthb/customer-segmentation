# E-Commerce Customer Segmentation

An end-to-end machine learning project that segments e-commerce customers 
into distinct groups using unsupervised learning techniques.

## Live Demo
https://customer-segmentation-lprxgdgvkjkfofdpxrvvtb.streamlit.app

## Overview
Applied RFM Analysis and K-Means Clustering on a real UK e-commerce dataset 
containing 541,910 transactions from 4,338 unique customers to identify 
distinct customer groups that businesses can use for targeted marketing.

## Segments Identified
| Segment | Description |
|---------|-------------|
| High Value Customers | Recent, frequent buyers with high spending |
| At Risk Customers | Previously active but showing declining engagement |
| New Customers | Recent first-time buyers with low frequency |
| Low Engagement Customers | Infrequent buyers with low monetary value |

## Tech Stack
- Python
- Pandas & NumPy
- Scikit-learn (K-Means Clustering)
- Plotly (Visualizations)
- Streamlit (Dashboard)

## ML Approach
- RFM Feature Engineering (Recency, Frequency, Monetary)
- Standard Scaling for normalization
- K-Means Clustering with Elbow Method for optimal K
- Interactive 3D cluster visualization

## Dataset
Online Retail Dataset from UCI Machine Learning Repository
- 541,910 transactions
- 4,338 unique customers
- UK based e-commerce store (2010-2011)

## How to Run
pip install -r requirements.txt
streamlit run app.py

## Project Structure
customer-segmentation/
├── data/
│   └── OnlineRetail.csv
├── app.py
├── segmentation.py
├── requirements.txt
└── README.md
