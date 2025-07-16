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