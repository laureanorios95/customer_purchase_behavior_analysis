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

###Data Loading and Initial Exploration###
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