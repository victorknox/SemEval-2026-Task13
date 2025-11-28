#!/usr/bin/env python3
"""
Hybrid approach: Ensemble of feature-based + small transformer
Focus on generalizable patterns given distribution shift
"""

import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import re
import warnings
warnings.filterwarnings('ignore')

def extract_robust_features(code):
    """Extract features that should generalize across distribution shifts"""
    features = {}
    
    lines = code.split('\n')
    non_empty_lines = [l for l in lines if l.strip()]
    
    # ROBUST FEATURES (less likely to shift with code length)
    
    # 1. Ratios (normalized, length-independent)
    total_chars = max(len(code), 1)
    features['comment_ratio'] = code.count('#') / max(len(non_empty_lines), 1)  # comments per line
    features['blank_line_ratio'] = sum(1 for l in lines if not l.strip()) / max(len(lines), 1)
    
    # 2. Indentation consistency (machine code tends to be more consistent)
    indents = [len(line) - len(line.lstrip()) for line in non_empty_lines]
    if indents:
        features['indent_std'] = np.std(indents)
        features['indent_consistency'] = 1 / (np.std(indents) + 1)  # higher = more consistent
    else:
        features['indent_std'] = 0
        features['indent_consistency'] = 1
    
    # 3. Variable naming patterns (humans use longer, more descriptive names)
    identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
    if identifiers:
        id_lengths = [len(i) for i in identifiers]
        features['median_identifier_length'] = np.median(id_lengths)
        features['pct_short_identifiers'] = sum(1 for i in identifiers if len(i) <= 2) / len(identifiers)
        features['identifier_diversity'] = len(set(identifiers)) / len(identifiers)
    else:
        features['median_identifier_length'] = 0
        features['pct_short_identifiers'] = 0
        features['identifier_diversity'] = 0
    
    # 4. Code structure patterns
    features['has_function'] = 1 if 'def ' in code else 0
    features['has_class'] = 1 if 'class ' in code else 0
    features['has_imports'] = 1 if 'import' in code else 0
    
    # 5. Line length consistency (machines might have more uniform formatting)
    line_lengths = [len(l) for l in lines]
    if line_lengths:
        features['line_length_cv'] = np.std(line_lengths) / (np.mean(line_lengths) + 1)  # coefficient of variation
    else:
        features['line_length_cv'] = 0
    
    # 6. Complexity per line (normalized)
    features['operators_per_line'] = (code.count('+') + code.count('-') + code.count('*') + code.count('/')) / max(len(non_empty_lines), 1)
    features['brackets_per_line'] = (code.count('(') + code.count('[') + code.count('{')) / max(len(non_empty_lines), 1)
    
    return features

print("=" * 100)
print("HYBRID APPROACH: ROBUST CLASSIFIER")
print("=" * 100)

# Load data
print("\nLoading data...")
train_df = pd.read_parquet('Task_A/train.parquet')
val_df = pd.read_parquet('Task_A/validation.parquet')
test_df = pd.read_parquet('Task_A/test.parquet')

print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# Sample strategically - balance and diversity
print("\nCreating balanced training sample...")
human_codes = train_df[train_df['label'] == 0].sample(n=50000, random_state=42)
machine_codes = train_df[train_df['label'] == 1].sample(n=50000, random_state=42)
train_sample = pd.concat([human_codes, machine_codes]).sample(frac=1, random_state=42)  # shuffle

print(f"Training on {len(train_sample):,} balanced samples")

# Extract ROBUST features
print("\nExtracting robust features...")
print("  Training set...")
train_features = []
for idx, row in train_sample.iterrows():
    if idx % 10000 == 0:
        print(f"    {len(train_features)}/{len(train_sample)}...")
    features = extract_robust_features(row['code'])
    features['label'] = row['label']
    train_features.append(features)

train_feat_df = pd.DataFrame(train_features)
X_train = train_feat_df.drop('label', axis=1)
y_train = train_feat_df['label']

print(f"  ✓ Training features: {X_train.shape}")

print("  Validation set...")
val_features = []
for idx, row in val_df.iterrows():
    if idx % 20000 == 0:
        print(f"    {len(val_features)}/{len(val_df)}...")
    features = extract_robust_features(row['code'])
    features['label'] = row['label']
    val_features.append(features)

val_feat_df = pd.DataFrame(val_features)
X_val = val_feat_df.drop('label', axis=1)
y_val = val_feat_df['label']

print(f"  ✓ Validation features: {X_val.shape}")

print("  Test set...")
test_features = []
for idx, row in test_df.iterrows():
    if idx % 200 == 0:
        print(f"    {len(test_features)}/{len(test_df)}...")
    features = extract_robust_features(row['code'])
    test_features.append(features)

test_feat_df = pd.DataFrame(test_features)
X_test = test_feat_df

print(f"  ✓ Test features: {X_test.shape}")

# Train ensemble of classifiers
print("\n" + "=" * 100)
print("TRAINING ENSEMBLE")
print("=" * 100)

# Use multiple models with different biases
models = [
    ('lr', LogisticRegression(max_iter=1000, C=0.1, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42))
]

ensemble = VotingClassifier(estimators=models, voting='soft')

print("Training ensemble...")
ensemble.fit(X_train, y_train)

# Evaluate
print("\n" + "=" * 100)
print("EVALUATION")
print("=" * 100)

val_pred = ensemble.predict(X_val)
val_proba = ensemble.predict_proba(X_val)

val_f1 = f1_score(y_val, val_pred)
print(f"\nValidation F1: {val_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, val_pred, target_names=['Human', 'Machine']))

# Analyze predictions by confidence
print("\n" + "=" * 100)
print("CONFIDENCE ANALYSIS")
print("=" * 100)

max_probas = val_proba.max(axis=1)
high_conf = max_probas > 0.7
print(f"\nHigh confidence (>0.7): {high_conf.sum()} / {len(high_conf)} ({high_conf.sum()/len(high_conf)*100:.1f}%)")
if high_conf.sum() > 0:
    high_conf_f1 = f1_score(y_val[high_conf], val_pred[high_conf])
    print(f"F1 on high confidence: {high_conf_f1:.4f}")

# Generate test predictions
print("\n" + "=" * 100)
print("TEST PREDICTIONS")
print("=" * 100)

test_pred = ensemble.predict(X_test)
test_proba = ensemble.predict_proba(X_test)

print(f"\nPredictions: {len(test_pred)}")
print(f"  Human (0): {sum(test_pred == 0)} ({sum(test_pred == 0)/len(test_pred)*100:.1f}%)")
print(f"  Machine (1): {sum(test_pred == 1)} ({sum(test_pred == 1)/len(test_pred)*100:.1f}%)")

# Analyze test confidence
test_max_probas = test_proba.max(axis=1)
print(f"\nMean confidence: {test_max_probas.mean():.3f}")
print(f"High confidence (>0.7): {(test_max_probas > 0.7).sum()} ({(test_max_probas > 0.7).sum()/len(test_max_probas)*100:.1f}%)")

# Save submission
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'label': test_pred
})

submission.to_csv('task_a_solution/results/robust_ensemble_submission.csv', index=False)
print(f"\n✓ Saved: task_a_solution/results/robust_ensemble_submission.csv")

# Also save with confidence scores
submission_with_conf = pd.DataFrame({
    'ID': test_df['ID'],
    'label': test_pred,
    'confidence': test_max_probas,
    'prob_human': test_proba[:, 0],
    'prob_machine': test_proba[:, 1]
})

submission_with_conf.to_csv('task_a_solution/results/predictions_with_confidence.csv', index=False)
print(f"✓ Saved: task_a_solution/results/predictions_with_confidence.csv")

print("\n" + "=" * 100)
print("COMPLETE!")
print("=" * 100)
print("\nKey insights:")
print("1. Focused on ROBUST features that normalize for code length")
print("2. Used balanced training to avoid bias")
print("3. Ensemble voting for robustness")
print("4. Test set is 64% longer - features should handle this")
