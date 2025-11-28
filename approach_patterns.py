#!/usr/bin/env python3
"""
Novel Approach 4: N-gram/Pattern Analysis
Look at actual code patterns/sequences that differ
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import re

def preprocess_code(code):
    """Convert code to sequence of tokens"""
    # Remove strings/comments to focus on structure
    code = re.sub(r'"[^"]*"', 'STRING', code)
    code = re.sub(r"'[^']*'", 'STRING', code)
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    
    # Tokenize
    tokens = re.findall(r'\b\w+\b|[^\w\s]', code)
    return ' '.join(tokens)

print("=" * 100)
print("APPROACH 4: CODE PATTERN ANALYSIS (TF-IDF on tokens)")
print("=" * 100)

# Load data
train_df = pd.read_parquet('Task_A/train.parquet')
val_df = pd.read_parquet('Task_A/validation.parquet')
test_df = pd.read_parquet('Task_A/test.parquet')

# Sample training
print("\nSampling and preprocessing...")
train_sample = train_df.sample(n=100000, random_state=42)

print("Tokenizing training code...")
train_processed = [preprocess_code(code) for code in train_sample['code']]

# Create TF-IDF features on code patterns
print("\nBuilding TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
    min_df=5,
    max_df=0.8
)

X_train = vectorizer.fit_transform(train_processed)
y_train = train_sample['label'].values

print(f"Feature matrix: {X_train.shape}")

# Train logistic regression
print("\nTraining Logistic Regression...")
model = LogisticRegression(C=1.0, max_iter=1000, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Validate
print("\nValidating...")
val_sample = val_df.sample(n=10000, random_state=42)
val_processed = [preprocess_code(code) for code in val_sample['code']]
X_val = vectorizer.transform(val_processed)
y_val = val_sample['label'].values

val_preds = model.predict(X_val)
val_f1 = f1_score(y_val, val_preds)

print(f"Validation F1: {val_f1:.4f}")

# Show discriminative patterns
print("\nTop patterns for HUMAN code:")
feature_names = vectorizer.get_feature_names_out()
coef = model.coef_[0]
top_human_idx = np.argsort(coef)[:20]
for idx in top_human_idx:
    print(f"  {feature_names[idx]:30s} (coef: {coef[idx]:.4f})")

print("\nTop patterns for MACHINE code:")
top_machine_idx = np.argsort(coef)[-20:][::-1]
for idx in top_machine_idx:
    print(f"  {feature_names[idx]:30s} (coef: {coef[idx]:.4f})")

# Predict on test
print("\nPredicting on test...")
test_processed = [preprocess_code(code) for code in test_df['code']]
X_test = vectorizer.transform(test_processed)
test_preds = model.predict(X_test)

submission = pd.DataFrame({
    'ID': test_df['ID'],
    'label': test_preds
})

submission.to_csv('task_a_solution/results/pattern_analysis_submission.csv', index=False)

human_pct = (submission['label'] == 0).sum() / len(submission) * 100
print(f"\n✓ Saved: pattern_analysis_submission.csv")
print(f"  Distribution: {human_pct:.1f}% Human, {100-human_pct:.1f}% Machine")
