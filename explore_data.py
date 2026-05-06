import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('netflix_titles.csv')
print("Total rows:", len(df))
print(df.isnull().sum())

# Drop nulls in date_added
df = df.dropna(subset=['date_added'])
df['date_added'] = df['date_added'].str.strip()
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df = df.dropna(subset=['date_added'])
print("Rows after parsing dates:", len(df))

# Extract year and month
df['added_year'] = df['date_added'].dt.year
df['added_month'] = df['date_added'].dt.month

# Monthly counts
monthly_counts = df.groupby(['added_year', 'added_month']).size().reset_index(name='uploads')
monthly_counts = monthly_counts.sort_values(['added_year', 'added_month'])

print("\nUploads by year:")
yearly_totals = df.groupby('added_year').size().reset_index(name='total_uploads')
print(yearly_totals)

print("\nLast 20 rows of monthly uploads:")
print(monthly_counts.tail(20))
