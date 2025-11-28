#!/usr/bin/env python3
"""
Quick test set analysis - What makes test different?
"""

import pandas as pd
import numpy as np

print("=" * 100)
print("QUICK TEST SET ANALYSIS")
print("=" * 100)

# Load data
train_df = pd.read_parquet('Task_A/train.parquet')
test_df = pd.read_parquet('Task_A/test.parquet')

print(f"\nTrain: {len(train_df):,} | Test: {len(test_df):,}")

# Quick stats
print("\n" + "=" * 100)
print("BASIC STATISTICS")
print("=" * 100)

train_sample = train_df.sample(n=5000, random_state=42)

train_stats = {
    'avg_length': train_sample['code'].str.len().mean(),
    'avg_lines': train_sample['code'].str.count('\n').mean(),
    'human_pct': (train_sample['label'] == 0).sum() / len(train_sample) * 100
}

test_stats = {
    'avg_length': test_df['code'].str.len().mean(),
    'avg_lines': test_df['code'].str.count('\n').mean(),
}

print("\nTrain (sample):")
for k, v in train_stats.items():
    print(f"  {k}: {v:.2f}")

print("\nTest:")
for k, v in test_stats.items():
    print(f"  {k}: {v:.2f}")

print(f"\nLength ratio (test/train): {test_stats['avg_length']/train_stats['avg_length']:.2f}x")
print(f"Lines ratio (test/train): {test_stats['avg_lines']/train_stats['avg_lines']:.2f}x")

# Check test set columns
print("\n" + "=" * 100)
print("TEST SET STRUCTURE")
print("=" * 100)
print(f"Columns: {test_df.columns.tolist()}")

# Show examples
print("\n" + "=" * 100)
print("TEST SET EXAMPLES (first 3)")
print("=" * 100)

for i in range(min(3, len(test_df))):
    code = test_df.iloc[i]['code']
    print(f"\n--- ID: {test_df.iloc[i]['ID']} ---")
    print(f"Length: {len(code)} chars, {code.count(chr(10))} lines")
    print(f"First 400 chars:")
    print(code[:400])
    print("...")

# Check if there are languages
if 'language' in train_df.columns:
    print("\n" + "=" * 100)
    print("LANGUAGE DISTRIBUTION")
    print("=" * 100)
    print("\nTrain:")
    print(train_sample['language'].value_counts())

# Check generators
if 'generator' in train_df.columns:
    print("\n" + "=" * 100)
    print("GENERATOR DISTRIBUTION (Machine code only)")
    print("=" * 100)
    machine_sample = train_sample[train_sample['label'] == 1]
    if len(machine_sample) > 0:
        print("\nTrain (machine-generated):")
        print(machine_sample['generator'].value_counts())

print("\n" + "=" * 100)
print("LABEL BALANCE IN TRAIN")
print("=" * 100)
print(train_sample['label'].value_counts())
print(f"\nHuman: {(train_sample['label']==0).sum()/len(train_sample)*100:.1f}%")
print(f"Machine: {(train_sample['label']==1).sum()/len(train_sample)*100:.1f}%")
