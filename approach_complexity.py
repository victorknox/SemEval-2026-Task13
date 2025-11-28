#!/usr/bin/env python3
"""
Novel Approach 2: Code Complexity Analysis
Maybe simpler code = human, complex = machine?
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import re

def complexity_features(code):
    """Extract complexity-based features"""
    features = {}
    
    lines = code.split('\n')
    code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
    
    # Cyclomatic complexity indicators
    features['num_if'] = code.count(' if ') + code.count('\nif ')
    features['num_for'] = code.count(' for ') + code.count('\nfor ')
    features['num_while'] = code.count(' while ') + code.count('\nwhile ')
    features['num_else'] = code.count(' else') + code.count('\nelse')
    features['num_elif'] = code.count(' elif ') + code.count('\nelif ')
    
    # Nesting depth (rough estimate)
    max_indent = 0
    for line in code_lines:
        indent = len(line) - len(line.lstrip())
        max_indent = max(max_indent, indent)
    features['max_nesting'] = max_indent // 4  # Assume 4 spaces per indent
    
    # Function complexity
    features['num_functions'] = code.count('def ') + code.count('function ')
    features['num_classes'] = code.count('class ')
    features['num_returns'] = code.count('return ')
    
    # Operator density
    operators = ['+', '-', '*', '/', '==', '!=', '<=', '>=', '&&', '||', 'and', 'or']
    features['operator_count'] = sum(code.count(op) for op in operators)
    features['operator_density'] = features['operator_count'] / max(len(code_lines), 1)
    
    # Variable/identifier count
    identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
    features['unique_identifiers'] = len(set(identifiers))
    features['total_identifiers'] = len(identifiers)
    features['identifier_reuse'] = features['total_identifiers'] / max(features['unique_identifiers'], 1)
    
    # Code structure
    features['lines_of_code'] = len(code_lines)
    features['avg_line_complexity'] = features['operator_count'] / max(len(code_lines), 1)
    
    # String/comment ratio
    features['string_count'] = code.count('"') + code.count("'")
    features['comment_count'] = code.count('#') + code.count('//')
    
    # Bracket/parenthesis nesting
    features['bracket_depth'] = code.count('(') + code.count('[') + code.count('{')
    
    return features

print("=" * 100)
print("APPROACH 2: COMPLEXITY-BASED CLASSIFICATION")
print("=" * 100)

# Load data
train_df = pd.read_parquet('Task_A/train.parquet')
val_df = pd.read_parquet('Task_A/validation.parquet')
test_df = pd.read_parquet('Task_A/test.parquet')

# Sample training data
print("\nSampling training data...")
human = train_df[train_df['label'] == 0].sample(n=30000, random_state=42)
machine = train_df[train_df['label'] == 1].sample(n=30000, random_state=42)
train_sample = pd.concat([human, machine]).sample(frac=1, random_state=42)

print(f"Training samples: {len(train_sample)}")

# Extract complexity features
print("\nExtracting complexity features...")
train_feats = []
for idx, code in enumerate(train_sample['code']):
    if idx % 5000 == 0:
        print(f"  {idx}/{len(train_sample)}...")
    train_feats.append(complexity_features(code))

X_train = pd.DataFrame(train_feats)
y_train = train_sample['label'].values

print(f"\nFeature matrix: {X_train.shape}")

# Train model
print("\nTraining Random Forest...")
model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Validate
print("\nValidating...")
val_sample = val_df.sample(n=10000, random_state=42)
val_feats = [complexity_features(code) for code in val_sample['code']]
X_val = pd.DataFrame(val_feats)
y_val = val_sample['label'].values

val_preds = model.predict(X_val)
val_f1 = f1_score(y_val, val_preds)

print(f"Validation F1: {val_f1:.4f}")

# Show feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 complexity features:")
print(feature_importance.head(10).to_string(index=False))

# Predict on test
print("\nPredicting on test...")
test_feats = [complexity_features(code) for code in test_df['code']]
X_test = pd.DataFrame(test_feats)
test_preds = model.predict(X_test)

submission = pd.DataFrame({
    'ID': test_df['ID'],
    'label': test_preds
})

submission.to_csv('task_a_solution/results/complexity_based_submission.csv', index=False)

human_pct = (submission['label'] == 0).sum() / len(submission) * 100
print(f"\n✓ Saved: complexity_based_submission.csv")
print(f"  Distribution: {human_pct:.1f}% Human, {100-human_pct:.1f}% Machine")
