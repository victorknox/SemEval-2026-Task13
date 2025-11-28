#!/usr/bin/env python3
"""
Novel Approach 3: Statistical Anomaly Detection
Treat machine code as "anomalies" vs normal human code
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import re

def statistical_features(code):
    """Extract statistical features for anomaly detection"""
    features = {}
    
    lines = code.split('\n')
    
    # Length statistics
    line_lengths = [len(l) for l in lines]
    features['mean_line_len'] = np.mean(line_lengths)
    features['std_line_len'] = np.std(line_lengths)
    features['max_line_len'] = np.max(line_lengths)
    features['min_line_len'] = np.min([l for l in line_lengths if l > 0]) if any(l > 0 for l in line_lengths) else 0
    
    # Character distribution
    features['alpha_pct'] = sum(c.isalpha() for c in code) / max(len(code), 1)
    features['digit_pct'] = sum(c.isdigit() for c in code) / max(len(code), 1)
    features['space_pct'] = sum(c.isspace() for c in code) / max(len(code), 1)
    features['special_pct'] = sum(not c.isalnum() and not c.isspace() for c in code) / max(len(code), 1)
    
    # Indentation patterns
    indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    if indents:
        features['indent_mean'] = np.mean(indents)
        features['indent_std'] = np.std(indents)
        features['indent_max'] = np.max(indents)
        features['indent_range'] = np.max(indents) - np.min(indents)
    else:
        features['indent_mean'] = 0
        features['indent_std'] = 0
        features['indent_max'] = 0
        features['indent_range'] = 0
    
    # Naming entropy (identifier diversity)
    identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
    if identifiers:
        unique_ratio = len(set(identifiers)) / len(identifiers)
        features['identifier_entropy'] = unique_ratio
        features['avg_identifier_len'] = np.mean([len(i) for i in identifiers])
        features['std_identifier_len'] = np.std([len(i) for i in identifiers])
    else:
        features['identifier_entropy'] = 0
        features['avg_identifier_len'] = 0
        features['std_identifier_len'] = 0
    
    # Regularity metrics (machine code might be more regular)
    features['line_len_cv'] = features['std_line_len'] / (features['mean_line_len'] + 1)  # coefficient of variation
    features['indent_cv'] = features['indent_std'] / (features['indent_mean'] + 1)
    
    return features

print("=" * 100)
print("APPROACH 3: ANOMALY DETECTION (Machine = Anomaly)")
print("=" * 100)

# Load data
train_df = pd.read_parquet('Task_A/train.parquet')
test_df = pd.read_parquet('Task_A/test.parquet')

# Strategy: Train on HUMAN code only, treat MACHINE as anomalies
print("\nTraining on HUMAN code only...")
human_train = train_df[train_df['label'] == 0].sample(n=50000, random_state=42)

print(f"Human training samples: {len(human_train)}")

# Extract features
print("\nExtracting features from human code...")
human_feats = []
for idx, code in enumerate(human_train['code']):
    if idx % 5000 == 0:
        print(f"  {idx}/{len(human_train)}...")
    human_feats.append(statistical_features(code))

X_human = pd.DataFrame(human_feats)

# Standardize features
scaler = StandardScaler()
X_human_scaled = scaler.fit_transform(X_human)

# Train Isolation Forest (anomaly detector)
print("\nTraining Isolation Forest...")
iso_forest = IsolationForest(
    contamination=0.1,  # Expect ~10% anomalies
    random_state=42,
    n_estimators=200,
    max_samples='auto'
)

iso_forest.fit(X_human_scaled)

# Test on validation to see if it works
print("\nTesting on validation...")
val_df = pd.read_parquet('Task_A/validation.parquet')
val_sample = val_df.sample(n=5000, random_state=42)

val_feats = [statistical_features(code) for code in val_sample['code']]
X_val = pd.DataFrame(val_feats)
X_val_scaled = scaler.transform(X_val)

# Predict (-1 = anomaly/machine, 1 = normal/human)
val_preds_raw = iso_forest.predict(X_val_scaled)
val_preds = (val_preds_raw == -1).astype(int)  # Convert to 0/1 (0=human, 1=machine)

from sklearn.metrics import classification_report, f1_score
print("\nValidation results:")
print(classification_report(val_sample['label'], val_preds, target_names=['Human', 'Machine']))
print(f"F1-Score: {f1_score(val_sample['label'], val_preds):.4f}")

# Predict on test
print("\nPredicting on test...")
test_feats = [statistical_features(code) for code in test_df['code']]
X_test = pd.DataFrame(test_feats)
X_test_scaled = scaler.transform(X_test)

test_preds_raw = iso_forest.predict(X_test_scaled)
test_preds = (test_preds_raw == -1).astype(int)

submission = pd.DataFrame({
    'ID': test_df['ID'],
    'label': test_preds
})

submission.to_csv('task_a_solution/results/anomaly_detection_submission.csv', index=False)

human_pct = (submission['label'] == 0).sum() / len(submission) * 100
print(f"\n✓ Saved: anomaly_detection_submission.csv")
print(f"  Distribution: {human_pct:.1f}% Human, {100-human_pct:.1f}% Machine")
