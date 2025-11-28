#!/usr/bin/env python3
"""
Advanced Feature Analysis for Human vs Machine Code Detection
Focus on interpretable features rather than black-box models
"""

import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import re
import ast
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def extract_code_features(code):
    """Extract interpretable features from code"""
    features = {}
    
    # Basic statistics
    features['length'] = len(code)
    features['num_lines'] = len(code.split('\n'))
    features['avg_line_length'] = len(code) / max(len(code.split('\n')), 1)
    
    # Character-level features
    features['alpha_ratio'] = sum(c.isalpha() for c in code) / max(len(code), 1)
    features['digit_ratio'] = sum(c.isdigit() for c in code) / max(len(code), 1)
    features['space_ratio'] = sum(c.isspace() for c in code) / max(len(code), 1)
    features['special_char_ratio'] = sum(not c.isalnum() and not c.isspace() for c in code) / max(len(code), 1)
    
    # Indentation patterns (machines often have perfect indentation)
    lines = code.split('\n')
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    features['avg_indent'] = np.mean(indents) if indents else 0
    features['indent_variance'] = np.var(indents) if indents else 0
    features['max_indent'] = max(indents) if indents else 0
    
    # Naming conventions (humans use more descriptive names)
    # Look for variable names (simple heuristic)
    identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
    if identifiers:
        features['avg_identifier_length'] = np.mean([len(i) for i in identifiers])
        features['max_identifier_length'] = max(len(i) for i in identifiers)
        features['num_unique_identifiers'] = len(set(identifiers))
        features['identifier_diversity'] = len(set(identifiers)) / max(len(identifiers), 1)
        
        # Check for common single-letter variables (i, j, k, x, y, etc.)
        single_letter = sum(1 for i in identifiers if len(i) == 1)
        features['single_letter_var_ratio'] = single_letter / max(len(identifiers), 1)
    else:
        features['avg_identifier_length'] = 0
        features['max_identifier_length'] = 0
        features['num_unique_identifiers'] = 0
        features['identifier_diversity'] = 0
        features['single_letter_var_ratio'] = 0
    
    # Comments (humans tend to comment more/differently)
    features['has_comments'] = 1 if '#' in code or '//' in code or '/*' in code else 0
    features['num_hash_comments'] = code.count('#')
    features['num_slashslash_comments'] = code.count('//')
    
    # Common Python keywords
    keywords = ['def', 'class', 'import', 'from', 'if', 'else', 'elif', 'for', 'while', 
                'try', 'except', 'return', 'lambda', 'with']
    for kw in keywords:
        features[f'has_{kw}'] = 1 if f' {kw} ' in f' {code} ' or f'\n{kw} ' in code else 0
    
    # Complexity indicators
    features['num_brackets'] = code.count('(') + code.count('[') + code.count('{')
    features['num_operators'] = code.count('+') + code.count('-') + code.count('*') + code.count('/')
    features['num_comparisons'] = code.count('==') + code.count('!=') + code.count('<') + code.count('>')
    
    # Line length variance (machines might have more uniform line lengths)
    line_lengths = [len(line) for line in lines]
    features['line_length_variance'] = np.var(line_lengths) if line_lengths else 0
    features['max_line_length'] = max(line_lengths) if line_lengths else 0
    
    # Blank lines (formatting style)
    features['num_blank_lines'] = sum(1 for line in lines if not line.strip())
    features['blank_line_ratio'] = features['num_blank_lines'] / max(len(lines), 1)
    
    # Complexity - nested structures
    features['max_nesting'] = max(line.count('    ') for line in lines) if lines else 0
    
    # String usage
    features['num_single_quotes'] = code.count("'")
    features['num_double_quotes'] = code.count('"')
    features['string_quote_ratio'] = features['num_single_quotes'] / max(features['num_double_quotes'], 1)
    
    return features

print("=" * 100)
print("ADVANCED FEATURE ANALYSIS")
print("=" * 100)

# Load training data
print("\nLoading training data...")
train_df = pd.read_parquet('Task_A/train.parquet')
print(f"Training samples: {len(train_df):,}")

# Sample for faster analysis
sample_size = 50000
train_sample = train_df.sample(n=sample_size, random_state=42)
print(f"Using {len(train_sample):,} samples for feature analysis")

# Extract features
print("\nExtracting features...")
train_features = []
for idx, row in train_sample.iterrows():
    if idx % 5000 == 0:
        print(f"  Processed {idx}/{len(train_sample)}...")
    features = extract_code_features(row['code'])
    features['label'] = row['label']
    features['language'] = row.get('language', 'unknown')
    features['generator'] = row.get('generator', 'unknown')
    train_features.append(features)

features_df = pd.DataFrame(train_features)
print(f"\n✓ Extracted {len(features_df.columns)-3} features")

# Analyze feature importance
print("\n" + "=" * 100)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 100)

X = features_df.drop(['label', 'language', 'generator'], axis=1)
y = features_df['label']

# Train Random Forest to get feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
print("\nTraining Random Forest...")
rf.fit(X, y)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importance.head(20).to_string(index=False))

# Analyze features by label
print("\n" + "=" * 100)
print("FEATURE STATISTICS BY LABEL")
print("=" * 100)

human_features = features_df[features_df['label'] == 0].drop(['label', 'language', 'generator'], axis=1)
machine_features = features_df[features_df['label'] == 1].drop(['label', 'language', 'generator'], axis=1)

print("\nTop features that differ between Human and Machine code:\n")
differences = []
for col in X.columns:
    human_mean = human_features[col].mean()
    machine_mean = machine_features[col].mean()
    diff = abs(human_mean - machine_mean)
    ratio = (human_mean / max(machine_mean, 0.0001))
    differences.append({
        'feature': col,
        'human_mean': human_mean,
        'machine_mean': machine_mean,
        'abs_diff': diff,
        'ratio': ratio
    })

diff_df = pd.DataFrame(differences).sort_values('abs_diff', ascending=False)
print(diff_df.head(20).to_string(index=False))

# Save analysis
print("\n" + "=" * 100)
print("SAVING ANALYSIS")
print("=" * 100)

# Save feature importance
feature_importance.to_csv('task_a_solution/results/feature_importance.csv', index=False)
print("✓ Saved: task_a_solution/results/feature_importance.csv")

# Save feature differences
diff_df.to_csv('task_a_solution/results/feature_differences.csv', index=False)
print("✓ Saved: task_a_solution/results/feature_differences.csv")

# Save sample features
features_df.head(1000).to_csv('task_a_solution/results/sample_features.csv', index=False)
print("✓ Saved: task_a_solution/results/sample_features.csv")

# Create interpretable insights
print("\n" + "=" * 100)
print("KEY INSIGHTS")
print("=" * 100)

insights = []
top_features = feature_importance.head(10)['feature'].tolist()

for feat in top_features:
    row = diff_df[diff_df['feature'] == feat].iloc[0]
    human_val = row['human_mean']
    machine_val = row['machine_mean']
    
    if human_val > machine_val:
        direction = "HIGHER in HUMAN code"
    else:
        direction = "HIGHER in MACHINE code"
    
    insight = f"{feat}: {direction} (H={human_val:.3f}, M={machine_val:.3f})"
    insights.append(insight)
    print(f"  • {insight}")

# Save insights
with open('task_a_solution/results/insights.txt', 'w') as f:
    f.write("KEY INSIGHTS FOR HUMAN VS MACHINE CODE\n")
    f.write("=" * 80 + "\n\n")
    for insight in insights:
        f.write(f"• {insight}\n")

print("\n✓ Saved: task_a_solution/results/insights.txt")

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
print("\nNext steps:")
print("1. Review feature_importance.csv to see what matters most")
print("2. Check feature_differences.csv for human vs machine patterns")
print("3. Use insights to build better classifier")
