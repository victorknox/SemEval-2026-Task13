#!/usr/bin/env python3
"""
Novel Approach 1: Per-Language Models
Maybe different languages have different human/machine patterns
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
import re
import warnings
warnings.filterwarnings('ignore')

def extract_features(code):
    """Quick feature extraction"""
    features = {}
    lines = code.split('\n')
    non_empty = [l for l in lines if l.strip()]
    
    features['length'] = len(code)
    features['num_lines'] = len(lines)
    features['avg_line_len'] = len(code) / max(len(lines), 1)
    
    indents = [len(l) - len(l.lstrip()) for l in non_empty]
    features['indent_std'] = np.std(indents) if indents else 0
    features['indent_mean'] = np.mean(indents) if indents else 0
    
    features['comment_ratio'] = code.count('#') / max(len(non_empty), 1)
    features['blank_ratio'] = sum(1 for l in lines if not l.strip()) / max(len(lines), 1)
    
    identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
    if identifiers:
        features['avg_id_len'] = np.mean([len(i) for i in identifiers])
        features['short_id_pct'] = sum(1 for i in identifiers if len(i) <= 2) / len(identifiers)
    else:
        features['avg_id_len'] = 0
        features['short_id_pct'] = 0
    
    features['has_def'] = 1 if 'def ' in code else 0
    features['has_class'] = 1 if 'class ' in code else 0
    features['has_import'] = 1 if 'import' in code else 0
    
    return features

def detect_language(code):
    """Detect programming language"""
    python_score = sum([
        'def ' in code, 'import ' in code, 'print(' in code,
        '.append(' in code, 'range(' in code, 'lambda' in code
    ])
    
    cpp_score = sum([
        '#include' in code, 'int main' in code, 'std::' in code,
        'cout' in code, 'cin' in code, 'vector<' in code
    ])
    
    java_score = sum([
        'public class' in code, 'public static void main' in code,
        'System.out' in code, '.println' in code, 'String[]' in code
    ])
    
    if python_score > cpp_score and python_score > java_score:
        return 'Python'
    elif cpp_score > python_score and cpp_score > java_score:
        return 'C++'
    elif java_score > 0:
        return 'Java'
    else:
        return 'Unknown'

print("=" * 100)
print("APPROACH 1: PER-LANGUAGE MODELS")
print("=" * 100)

# Load data
train_df = pd.read_parquet('Task_A/train.parquet')
val_df = pd.read_parquet('Task_A/validation.parquet')
test_df = pd.read_parquet('Task_A/test.parquet')

# Detect languages
print("\nDetecting languages...")
train_df['detected_lang'] = train_df['code'].apply(detect_language)
val_df['detected_lang'] = val_df['code'].apply(detect_language)
test_df['detected_lang'] = test_df['code'].apply(detect_language)

print("\nLanguage distribution:")
print("\nTraining:")
print(train_df['detected_lang'].value_counts())
print("\nValidation:")
print(val_df['detected_lang'].value_counts())
print("\nTest:")
print(test_df['detected_lang'].value_counts())

# Train separate model for each language
language_models = {}

for lang in ['Python', 'C++', 'Java', 'Unknown']:
    print(f"\n{'='*80}")
    print(f"Training model for {lang}")
    print(f"{'='*80}")
    
    # Get samples for this language
    train_lang = train_df[train_df['detected_lang'] == lang]
    
    if len(train_lang) < 100:
        print(f"  Skipping {lang} (too few samples: {len(train_lang)})")
        continue
    
    # Sample balanced
    n_samples = min(20000, len(train_lang) // 2)
    human = train_lang[train_lang['label'] == 0].sample(n=min(n_samples, len(train_lang[train_lang['label'] == 0])), random_state=42)
    machine = train_lang[train_lang['label'] == 1].sample(n=min(n_samples, len(train_lang[train_lang['label'] == 1])), random_state=42)
    train_sample = pd.concat([human, machine])
    
    print(f"  Training samples: {len(train_sample)} ({len(human)} human, {len(machine)} machine)")
    
    # Extract features
    train_feats = [extract_features(code) for code in train_sample['code']]
    X_train = pd.DataFrame(train_feats)
    y_train = train_sample['label'].values
    
    # Train model
    model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Validate
    val_lang = val_df[val_df['detected_lang'] == lang]
    if len(val_lang) > 0:
        val_feats = [extract_features(code) for code in val_lang['code']]
        X_val = pd.DataFrame(val_feats)
        y_val = val_lang['label'].values
        
        preds = model.predict(X_val)
        f1 = f1_score(y_val, preds)
        print(f"  Validation F1: {f1:.4f}")
    
    language_models[lang] = model

# Predict on test
print("\n" + "=" * 100)
print("PREDICTING ON TEST SET")
print("=" * 100)

predictions = []
for idx, row in test_df.iterrows():
    lang = row['detected_lang']
    
    if lang in language_models:
        feats = extract_features(row['code'])
        X = pd.DataFrame([feats])
        pred = language_models[lang].predict(X)[0]
    else:
        # Default to most common prediction from Python model
        pred = 1  # Default to Machine
    
    predictions.append(pred)

test_df['prediction'] = predictions

# Analyze
print(f"\nPredictions by language:")
for lang in test_df['detected_lang'].unique():
    lang_preds = test_df[test_df['detected_lang'] == lang]['prediction']
    human_pct = (lang_preds == 0).sum() / len(lang_preds) * 100
    print(f"  {lang:10s}: {len(lang_preds):4d} samples | {human_pct:5.1f}% Human")

# Save
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'label': predictions
})

submission.to_csv('task_a_solution/results/per_language_submission.csv', index=False)

overall_human = (submission['label'] == 0).sum() / len(submission) * 100
print(f"\n✓ Saved: per_language_submission.csv")
print(f"  Overall: {overall_human:.1f}% Human, {100-overall_human:.1f}% Machine")
