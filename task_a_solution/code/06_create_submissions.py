"""
Create submission files in SemEval format for all models
Format: id, label (label ID, not string)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("="*80)
print("CREATING SUBMISSION FILES FOR SEMEVAL-2026 TASK 13")
print("="*80)

# Load original data to get IDs
print("\n[1] Loading original data with IDs...")
df = pd.read_parquet('task_A/task_a_trial.parquet')
print(f"✓ Loaded {len(df):,} samples")
print(f"✓ Index range: {df.index.min()} to {df.index.max()}")

# Get the same test split (must use same random state)
print("\n[2] Splitting data (same as training)...")
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    df['code'].values, 
    df['label'].values,
    df.index.values,
    test_size=0.2, 
    random_state=42, 
    stratify=df['label']
)
print(f"✓ Test samples: {len(X_test):,}")
print(f"✓ Test IDs range: {idx_test.min()} to {idx_test.max()}")

# Create submission files for each model

# 1. BASELINE
print("\n[3] Creating Baseline submission file...")
baseline_preds = pd.read_csv('task_a_solution/results/baseline_predictions.csv')
submission_baseline = pd.DataFrame({
    'id': idx_test,
    'label': baseline_preds['predicted_label'].values
})
submission_baseline.to_csv('task_a_solution/results/submission_baseline.csv', index=False)
print(f"✓ Saved: task_a_solution/results/submission_baseline.csv")
print(f"  Shape: {submission_baseline.shape}")
print(f"  Sample:\n{submission_baseline.head()}")

# 2. DISTILBERT
print("\n[4] Creating DistilBERT submission file...")
distilbert_preds = pd.read_csv('task_a_solution/results/distilbert_predictions.csv')
submission_distilbert = pd.DataFrame({
    'id': idx_test,
    'label': distilbert_preds['predicted_label'].values
})
submission_distilbert.to_csv('task_a_solution/results/submission_distilbert.csv', index=False)
print(f"✓ Saved: task_a_solution/results/submission_distilbert.csv")
print(f"  Shape: {submission_distilbert.shape}")
print(f"  Sample:\n{submission_distilbert.head()}")

# 3. CODEBERT (BEST MODEL)
print("\n[5] Creating CodeBERT submission file (BEST MODEL) ⭐...")
codebert_preds = pd.read_csv('task_a_solution/results/codebert_predictions.csv')
submission_codebert = pd.DataFrame({
    'id': idx_test,
    'label': codebert_preds['predicted_label'].values
})
submission_codebert.to_csv('task_a_solution/results/submission_codebert.csv', index=False)
print(f"✓ Saved: task_a_solution/results/submission_codebert.csv")
print(f"  Shape: {submission_codebert.shape}")
print(f"  Sample:\n{submission_codebert.head()}")

# Verify format
print("\n" + "="*80)
print("SUBMISSION FILE VERIFICATION")
print("="*80)

for model_name, filename in [
    ("Baseline", "submission_baseline.csv"),
    ("DistilBERT", "submission_distilbert.csv"),
    ("CodeBERT ⭐", "submission_codebert.csv")
]:
    print(f"\n{model_name}:")
    df_sub = pd.read_csv(f'task_a_solution/results/{filename}')
    print(f"  Columns: {df_sub.columns.tolist()}")
    print(f"  Shape: {df_sub.shape}")
    print(f"  Label values: {sorted(df_sub['label'].unique())}")
    print(f"  Label counts: {df_sub['label'].value_counts().to_dict()}")
    
    # Check format compliance
    assert list(df_sub.columns) == ['id', 'label'], "Wrong column names!"
    assert df_sub['label'].isin([0, 1]).all(), "Labels must be 0 or 1!"
    assert len(df_sub) == 2000, "Must have 2000 predictions!"
    print(f"  ✅ Format verified - ready for submission!")

print("\n" + "="*80)
print("SUBMISSION FILES CREATED SUCCESSFULLY!")
print("="*80)
print("\n📤 RECOMMENDED FOR SUBMISSION:")
print("   task_a_solution/results/submission_codebert.csv")
print("   (Best model: 95.95% F1-Score)")
print("\n✅ All files follow SemEval format: id, label")
