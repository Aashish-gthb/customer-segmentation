import pandas as pd

def load_and_clean(filepath="data/OnlineRetail.csv"):
    df = pd.read_csv(filepath, encoding='latin-1')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])  
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df

    return df


def build_rfm(df):
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  
        'InvoiceNo':   'nunique',                                  
        'TotalPrice':  'sum'                                       
    }).reset_index()

    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    return rfm


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def apply_kmeans(rfm, n_clusters=4):
    # Scale the features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    # Train K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Label the segments meaningfully
    segment_map = {
        0: "High Value Customers",
        1: "At Risk Customers",
        2: "New Customers",
        3: "Low Engagement Customers"
    }
    rfm['Segment'] = rfm['Cluster'].map(segment_map)

    return rfm


import matplotlib.pyplot as plt

def plot_elbow(rfm_scaled):
    inertia = []
    K = range(2, 11)

    for k in K:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(rfm_scaled)
        inertia.append(km.inertia_)

    plt.plot(K, inertia, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method to Find Optimal K')
    plt.savefig('elbow_plot.png')
