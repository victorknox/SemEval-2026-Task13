#!/usr/bin/env python3
"""
Build feature-based classifier using interpretable features
"""

import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import re
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
    identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
    if identifiers:
        features['avg_identifier_length'] = np.mean([len(i) for i in identifiers])
        features['max_identifier_length'] = max(len(i) for i in identifiers)
        features['num_unique_identifiers'] = len(set(identifiers))
        features['identifier_diversity'] = len(set(identifiers)) / max(len(identifiers), 1)
        features['single_letter_var_ratio'] = sum(1 for i in identifiers if len(i) == 1) / max(len(identifiers), 1)
    else:
        features['avg_identifier_length'] = 0
        features['max_identifier_length'] = 0
        features['num_unique_identifiers'] = 0
        features['identifier_diversity'] = 0
        features['single_letter_var_ratio'] = 0
    
    # Comments
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
    
    # Line length variance
    line_lengths = [len(line) for line in lines]
    features['line_length_variance'] = np.var(line_lengths) if line_lengths else 0
    features['max_line_length'] = max(line_lengths) if line_lengths else 0
    
    # Blank lines
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
print("FEATURE-BASED CLASSIFIER")
print("=" * 100)

# Load data
print("\nLoading data...")
train_df = pd.read_parquet('Task_A/train.parquet')
val_df = pd.read_parquet('Task_A/validation.parquet')
test_df = pd.read_parquet('Task_A/test.parquet')

print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# Use more training data
train_sample = train_df.sample(n=100000, random_state=42)
print(f"Using {len(train_sample):,} training samples")

# Extract features
print("\nExtracting training features...")
train_features = []
for idx, row in train_sample.iterrows():
    if idx % 10000 == 0:
        print(f"  {idx}/{len(train_sample)}...")
    features = extract_code_features(row['code'])
    features['label'] = row['label']
    train_features.append(features)

train_df_feat = pd.DataFrame(train_features)
X_train = train_df_feat.drop('label', axis=1)
y_train = train_df_feat['label']

print(f"\n✓ Training features: {X_train.shape}")

# Extract validation features
print("\nExtracting validation features...")
val_features = []
for idx, row in val_df.iterrows():
    if idx % 10000 == 0:
        print(f"  {idx}/{len(val_df)}...")
    features = extract_code_features(row['code'])
    features['label'] = row['label']
    val_features.append(features)

val_df_feat = pd.DataFrame(val_features)
X_val = val_df_feat.drop('label', axis=1)
y_val = val_df_feat['label']

print(f"✓ Validation features: {X_val.shape}")

# Extract test features
print("\nExtracting test features...")
test_features = []
for idx, row in test_df.iterrows():
    features = extract_code_features(row['code'])
    test_features.append(features)

test_df_feat = pd.DataFrame(test_features)
X_test = test_df_feat

print(f"✓ Test features: {X_test.shape}")

# Train models
print("\n" + "=" * 100)
print("TRAINING MODELS")
print("=" * 100)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=7, random_state=42)
}

best_model = None
best_f1 = 0

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    val_pred = model.predict(X_val)
    val_f1 = f1_score(y_val, val_pred)
    
    print(f"  Validation F1: {val_f1:.4f}")
    print(classification_report(y_val, val_pred, target_names=['Human', 'Machine']))
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_model = model
        best_name = name

print(f"\n✓ Best model: {best_name} (F1: {best_f1:.4f})")

# Generate predictions
print("\n" + "=" * 100)
print("GENERATING PREDICTIONS")
print("=" * 100)

predictions = best_model.predict(X_test)
print(f"\nPredictions: {len(predictions)}")
print(f"  Human (0): {sum(predictions == 0)} ({sum(predictions == 0)/len(predictions)*100:.1f}%)")
print(f"  Machine (1): {sum(predictions == 1)} ({sum(predictions == 1)/len(predictions)*100:.1f}%)")

# Create submission
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'label': predictions
})

submission.to_csv('task_a_solution/results/feature_based_submission.csv', index=False)
print(f"\n✓ Saved: task_a_solution/results/feature_based_submission.csv")

print("\n" + "=" * 100)
print("COMPLETE!")
print("=" * 100)
