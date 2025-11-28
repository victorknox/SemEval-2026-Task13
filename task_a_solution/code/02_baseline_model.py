"""
Baseline Model: TF-IDF + Logistic Regression
SemEval-2026 Task 13 - Task A: Binary Machine-Generated Code Detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import json
import pickle
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("BASELINE MODEL: TF-IDF + LOGISTIC REGRESSION")
print("="*80)

# Load data
print("\n[1] Loading data...")
df = pd.read_parquet('task_A/task_a_trial.parquet')
print(f"✓ Loaded {len(df):,} samples")

# Load label mappings
with open('task_A/id_to_label.json', 'r') as f:
    id_to_label = json.load(f)

# Prepare data
print("\n[2] Preparing data...")
X = df['code'].values
y = df['label'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Train samples: {len(X_train):,}")
print(f"✓ Test samples: {len(X_test):,}")
print(f"✓ Train label distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"✓ Test label distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

# Create TF-IDF features
print("\n[3] Creating TF-IDF features...")
print("   Using character n-grams (1-4) to capture code patterns...")
start_time = time.time()

vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(1, 4),
    max_features=10000,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

vectorization_time = time.time() - start_time
print(f"✓ Feature matrix shape: {X_train_tfidf.shape}")
print(f"✓ Vectorization time: {vectorization_time:.2f} seconds")

# Train Logistic Regression
print("\n[4] Training Logistic Regression...")
start_time = time.time()

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',
    solver='saga',
    n_jobs=-1,
    verbose=1
)

model.fit(X_train_tfidf, y_train)
training_time = time.time() - start_time
print(f"✓ Training time: {training_time:.2f} seconds")

# Make predictions
print("\n[5] Making predictions...")
y_pred = model.predict(X_test_tfidf)
y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]

# Calculate metrics
print("\n[6] Evaluating model...")
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average='macro'
)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*80)
print("BASELINE MODEL RESULTS")
print("="*80)
print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\n" + "-"*80)
print("Classification Report:")
print("-"*80)
target_names = [id_to_label[str(i)] for i in sorted(np.unique(y_test))]
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Save results
results = {
    'model_name': 'TF-IDF + Logistic Regression',
    'timestamp': datetime.now().isoformat(),
    'metrics': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc)
    },
    'training_time': float(training_time),
    'vectorization_time': float(vectorization_time),
    'train_samples': int(len(X_train)),
    'test_samples': int(len(X_test)),
    'confusion_matrix': cm.tolist()
}

with open('task_a_solution/results/baseline_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n✓ Saved: task_a_solution/results/baseline_results.json")

# Save model
with open('task_a_solution/models/baseline_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'vectorizer': vectorizer}, f)
print("✓ Saved: task_a_solution/models/baseline_model.pkl")

# Create visualizations
print("\n[7] Creating visualizations...")

# 1. Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=target_names, yticklabels=target_names,
            cbar_kws={'label': 'Count'})
ax.set_title('Baseline Model - Confusion Matrix', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('task_a_solution/plots/baseline_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: task_a_solution/plots/baseline_confusion_matrix.png")
plt.close()

# 2. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Baseline Model - ROC Curve', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('task_a_solution/plots/baseline_roc_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: task_a_solution/plots/baseline_roc_curve.png")
plt.close()

# 3. Performance metrics bar plot
fig, ax = plt.subplots(figsize=(10, 6))
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
metrics_values = [accuracy, precision, recall, f1, roc_auc]
colors = sns.color_palette("husl", len(metrics_names))
bars = ax.bar(metrics_names, metrics_values, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylim(0, 1.0)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Baseline Model - Performance Metrics', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, metrics_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('task_a_solution/plots/baseline_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: task_a_solution/plots/baseline_metrics.png")
plt.close()

# 4. Prediction distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Probability distribution for each class
for label in [0, 1]:
    label_name = id_to_label[str(label)]
    probs = y_pred_proba[y_test == label]
    axes[0].hist(probs, bins=30, alpha=0.6, label=f'True: {label_name}', edgecolor='black')
axes[0].set_xlabel('Predicted Probability (Machine)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Prediction counts
pred_counts = pd.Series(y_pred).value_counts().sort_index()
axes[1].bar([id_to_label[str(i)] for i in pred_counts.index], 
            pred_counts.values, color=['#3498db', '#e74c3c'], edgecolor='black')
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Prediction Distribution', fontsize=14, fontweight='bold')
for i, v in enumerate(pred_counts.values):
    axes[1].text(i, v + 10, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('task_a_solution/plots/baseline_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: task_a_solution/plots/baseline_predictions.png")
plt.close()

print("\n" + "="*80)
print("BASELINE MODEL TRAINING COMPLETE!")
print("="*80)
print(f"\nMacro F1-Score: {f1:.4f}")
print(f"Total time: {training_time + vectorization_time:.2f} seconds")
