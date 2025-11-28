#!/usr/bin/env python3
"""
Analyze test set to understand distribution shift
"""

import pandas as pd
import numpy as np
import re

def extract_code_features(code):
    """Extract interpretable features from code"""
    features = {}
    
    # Basic statistics
    features['length'] = len(code)
    features['num_lines'] = len(code.split('\n'))
    features['avg_line_length'] = len(code) / max(len(code.split('\n')), 1)
    
    # Indentation patterns
    lines = code.split('\n')
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    features['avg_indent'] = np.mean(indents) if indents else 0
    features['indent_variance'] = np.var(indents) if indents else 0
    features['max_indent'] = max(indents) if indents else 0
    
    # Naming conventions
    identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
    if identifiers:
        features['avg_identifier_length'] = np.mean([len(i) for i in identifiers])
        features['num_unique_identifiers'] = len(set(identifiers))
    else:
        features['avg_identifier_length'] = 0
        features['num_unique_identifiers'] = 0
    
    # Line length variance
    line_lengths = [len(line) for line in lines]
    features['line_length_variance'] = np.var(line_lengths) if line_lengths else 0
    features['max_line_length'] = max(line_lengths) if line_lengths else 0
    
    # Complexity
    features['max_nesting'] = max(line.count('    ') for line in lines) if lines else 0
    features['num_hash_comments'] = code.count('#')
    
    return features

print("=" * 100)
print("TEST SET DISTRIBUTION ANALYSIS")
print("=" * 100)

# Load data
train_df = pd.read_parquet('Task_A/train.parquet')
val_df = pd.read_parquet('Task_A/validation.parquet')
test_df = pd.read_parquet('Task_A/test.parquet')

print(f"\nDataset sizes:")
print(f"  Train: {len(train_df):,}")
print(f"  Validation: {len(val_df):,}")
print(f"  Test: {len(test_df):,}")

# Sample from each
train_sample = train_df.sample(n=min(5000, len(train_df)), random_state=42)
val_sample = val_df.sample(n=min(5000, len(val_df)), random_state=42)

print("\n" + "=" * 100)
print("EXTRACTING FEATURES")
print("=" * 100)

# Extract features from each set
print("\nTrain...")
train_features = [extract_code_features(code) for code in train_sample['code']]
train_feat_df = pd.DataFrame(train_features)

print("Validation...")
val_features = [extract_code_features(code) for code in val_sample['code']]
val_feat_df = pd.DataFrame(val_features)

print("Test...")
test_features = [extract_code_features(code) for code in test_df['code']]
test_feat_df = pd.DataFrame(test_features)

print("\n" + "=" * 100)
print("DISTRIBUTION COMPARISON")
print("=" * 100)

# Compare distributions
comparison = pd.DataFrame({
    'Feature': train_feat_df.columns,
    'Train_Mean': train_feat_df.mean(),
    'Val_Mean': val_feat_df.mean(),
    'Test_Mean': test_feat_df.mean()
})

comparison['Train_Test_Diff'] = abs(comparison['Train_Mean'] - comparison['Test_Mean'])
comparison['Train_Test_Ratio'] = comparison['Train_Mean'] / (comparison['Test_Mean'] + 0.001)

comparison = comparison.sort_values('Train_Test_Diff', ascending=False)

print("\nFeatures with BIGGEST distribution shift (Train vs Test):\n")
print(comparison.head(15).to_string(index=False))

print("\n" + "=" * 100)
print("LABEL DISTRIBUTION")
print("=" * 100)

print("\nTrain:")
print(train_sample['label'].value_counts())
print(f"  Human: {sum(train_sample['label']==0)/len(train_sample)*100:.1f}%")
print(f"  Machine: {sum(train_sample['label']==1)/len(train_sample)*100:.1f}%")

print("\nValidation:")
print(val_sample['label'].value_counts())
print(f"  Human: {sum(val_sample['label']==0)/len(val_sample)*100:.1f}%")
print(f"  Machine: {sum(val_sample['label']==1)/len(val_sample)*100:.1f}%")

# Save comparison
comparison.to_csv('task_a_solution/results/distribution_shift_analysis.csv', index=False)
print(f"\n✓ Saved: task_a_solution/results/distribution_shift_analysis.csv")

# Check if test set has any metadata
print("\n" + "=" * 100)
print("TEST SET COLUMNS")
print("=" * 100)
print(test_df.columns.tolist())
print(f"\nTest set sample:")
print(test_df.head(3))

print("\n" + "=" * 100)
print("CODE EXAMPLES FROM TEST SET")
print("=" * 100)
for i in range(min(3, len(test_df))):
    print(f"\n--- Example {i+1} (ID: {test_df.iloc[i]['ID']}) ---")
    print(test_df.iloc[i]['code'][:300])
    print("...")
