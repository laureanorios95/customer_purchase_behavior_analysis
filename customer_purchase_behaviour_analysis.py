# Customer Purchase Behavior Analysis
# Goal: Analyze e-commerce data to identify patterns and provide business insights

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

### Data Loading and Initial Exploration ###
# Load datasets
customers_df = pd.read_csv('customers.csv')
orders_df = pd.read_csv('orders.csv')
products_df = pd.read_csv('products.csv')

# Quick data overview
print("Dataset Shapes:")
print(f"Customers: {customers_df.shape}")
print(f"Orders: {orders_df.shape}")
print(f"Products: {products_df.shape}")

# Check for missing values
def check_missing_values(df, df_name):
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_table = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(f"\nMissing values in {df_name}:")
    print(missing_table[missing_table['Missing Count'] > 0])
    
for df, name in [(customers_df, 'Customers'), (orders_df, 'Orders'), (products_df, 'Products')]:
    check_missing_values(df, name)

### Data Cleaning and Preprocessing ###
# Convert date columns
orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
customers_df['registration_date'] = pd.to_datetime(customers_df['registration_date'])

# Create derived features
orders_df['order_month'] = orders_df['order_date'].dt.to_period('M')
orders_df['order_day_of_week'] = orders_df['order_date'].dt.day_name()
orders_df['order_hour'] = orders_df['order_date'].dt.hour

# Handle missing values
orders_df['discount_applied'].fillna(0, inplace=True)
customers_df['age'].fillna(customers_df['age'].median(), inplace=True)

### Exploratory Data Analysis ###
# Customer Demographics Analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Age distribution
axes[0, 0].hist(customers_df['age'], bins=20, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Customer Age Distribution')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Count')

# Customer segments
segment_counts = customers_df['segment'].value_counts()
axes[0, 1].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
axes[0, 1].set_title('Customer Segments')

# Revenue by month
monthly_revenue = orders_df.groupby('order_month')['total_amount'].sum()
axes[1, 0].plot(monthly_revenue.index.astype(str), monthly_revenue.values, marker='o')
axes[1, 0].set_title('Monthly Revenue Trend')
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Revenue ($)')
axes[1, 0].tick_params(axis='x', rotation=45)

# Order distribution by day of week
day_orders = orders_df['order_day_of_week'].value_counts().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])
axes[1, 1].bar(day_orders.index, day_orders.values)
axes[1, 1].set_title('Orders by Day of Week')
axes[1, 1].set_xlabel('Day')
axes[1, 1].set_ylabel('Number of Orders')

plt.tight_layout()
plt.show()

### Customer Segmentation Analysis ###
# RFM Analysis (Recency, Frequency, Monetary)
current_date = orders_df['order_date'].max()

rfm = orders_df.groupby('customer_id').agg({
    'order_date': lambda x: (current_date - x.max()).days,  # Recency
    'order_id': 'count',  # Frequency
    'total_amount': 'sum'  # Monetary
}).rename(columns={
    'order_date': 'recency',
    'order_id': 'frequency',
    'total_amount': 'monetary'
})

# Create RFM segments
def rfm_segmentation(row):
    if row['recency'] < 30 and row['frequency'] > 10 and row['monetary'] > 1000:
        return 'Champions'
    elif row['recency'] < 90 and row['frequency'] > 5 and row['monetary'] > 500:
        return 'Loyal Customers'
    elif row['recency'] < 90 and row['frequency'] <= 5:
        return 'Potential Loyalists'
    elif row['recency'] >= 90 and row['frequency'] > 5:
        return 'At Risk'
    elif row['recency'] >= 180:
        return 'Lost'
    else:
        return 'Others'

rfm['segment'] = rfm.apply(rfm_segmentation, axis=1)

# Visualize RFM segments
plt.figure(figsize=(10, 6))
segment_summary = rfm['segment'].value_counts()
plt.bar(segment_summary.index, segment_summary.values)
plt.title('Customer Segments based on RFM Analysis')
plt.xlabel('Segment')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

### Product Performance Analysis ###
# Merge orders with products
order_products = orders_df.merge(products_df, on='product_id')

# Top performing products
top_products = order_products.groupby('product_name').agg({
    'quantity': 'sum',
    'total_amount': 'sum',
    'order_id': 'count'
}).rename(columns={'order_id': 'order_count'}).sort_values('total_amount', ascending=False).head(10)

# Visualize top products
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

top_products['total_amount'].plot(kind='barh', ax=ax1)
ax1.set_title('Top 10 Products by Revenue')
ax1.set_xlabel('Revenue ($)')

# Product category performance
category_performance = order_products.groupby('category')['total_amount'].sum().sort_values(ascending=False)
ax2.pie(category_performance.values, labels=category_performance.index, autopct='%1.1f%%')
ax2.set_title('Revenue by Product Category')

plt.tight_layout()
plt.show()

### Predictive Analysis ###
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Prepare data for clustering
customer_features = rfm[['recency', 'frequency', 'monetary']].copy()
scaler = StandardScaler()
customer_features_scaled = scaler.fit_transform(customer_features)

# Find optimal number of clusters
silhouette_scores = []
K = range(2, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(customer_features_scaled)
    score = silhouette_score(customer_features_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Optimal Number of Clusters')
plt.show()

# Apply final clustering
optimal_k = K[np.argmax(silhouette_scores)]
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
rfm['cluster'] = kmeans.fit_predict(customer_features_scaled)