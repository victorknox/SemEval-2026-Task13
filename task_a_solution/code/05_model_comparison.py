"""
Model Comparison and Comprehensive Analysis
SemEval-2026 Task 13 - Task A: Binary Machine-Generated Code Detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPREHENSIVE MODEL COMPARISON AND ANALYSIS")
print("="*80)

# Load all results
print("\n[1] Loading results from all models...")
with open('task_a_solution/results/baseline_results.json', 'r') as f:
    baseline_results = json.load(f)

with open('task_a_solution/results/distilbert_results.json', 'r') as f:
    distilbert_results = json.load(f)

with open('task_a_solution/results/codebert_results.json', 'r') as f:
    codebert_results = json.load(f)

models = ['Baseline\n(TF-IDF + LR)', 'DistilBERT', 'CodeBERT']
results = [baseline_results, distilbert_results, codebert_results]

print("✓ Loaded results for 3 models")

# Extract metrics
print("\n[2] Extracting metrics...")
metrics_data = {
    'Model': models,
    'Accuracy': [r['metrics']['accuracy'] for r in results],
    'Precision': [r['metrics']['precision'] for r in results],
    'Recall': [r['metrics']['recall'] for r in results],
    'F1-Score': [r['metrics']['f1_score'] for r in results],
    'ROC-AUC': [r['metrics']['roc_auc'] for r in results]
}

metrics_df = pd.DataFrame(metrics_data)
print("\n" + "="*80)
print("MODEL COMPARISON TABLE")
print("="*80)
print(metrics_df.to_string(index=False))

# Save comparison table
metrics_df.to_csv('task_a_solution/results/model_comparison.csv', index=False)
print("\n✓ Saved: task_a_solution/results/model_comparison.csv")

# Create comprehensive visualizations
print("\n[3] Creating comprehensive visualizations...")

# 1. Performance Metrics Comparison (Bar Plot)
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(models))
width = 0.15
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
colors = sns.color_palette("husl", len(metrics_to_plot))

for i, metric in enumerate(metrics_to_plot):
    values = metrics_df[metric].values
    bars = ax.bar(x + i*width, values, width, label=metric, color=colors[i])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(models)
ax.legend(loc='lower right')
ax.set_ylim(0.8, 1.0)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('task_a_solution/plots/model_comparison_all_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: task_a_solution/plots/model_comparison_all_metrics.png")
plt.close()

# 2. F1-Score Comparison (Focus)
fig, ax = plt.subplots(figsize=(10, 6))
f1_scores = metrics_df['F1-Score'].values
colors_f1 = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax.bar(models, f1_scores, color=colors_f1, edgecolor='black', linewidth=2)

for i, (bar, score) in enumerate(zip(bars, f1_scores)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{score:.4f}\n({score*100:.2f}%)', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Show improvement over baseline
    if i > 0:
        improvement = score - f1_scores[0]
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                f'+{improvement:.4f}', 
                ha='center', va='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_ylabel('F1-Score', fontsize=12)
ax.set_title('Macro F1-Score Comparison (Primary Metric)', fontsize=14, fontweight='bold')
ax.set_ylim(0.85, 1.0)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('task_a_solution/plots/model_comparison_f1.png', dpi=300, bbox_inches='tight')
print("✓ Saved: task_a_solution/plots/model_comparison_f1.png")
plt.close()

# 3. Confusion Matrices Comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (result, model_name) in enumerate(zip(results, models)):
    cm = np.array(result['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Human', 'Machine'], yticklabels=['Human', 'Machine'],
                cbar_kws={'label': 'Count'})
    axes[i].set_title(f'{model_name}\nF1: {result["metrics"]["f1_score"]:.4f}', 
                     fontsize=12, fontweight='bold')
    axes[i].set_ylabel('True Label', fontsize=11)
    axes[i].set_xlabel('Predicted Label', fontsize=11)

plt.tight_layout()
plt.savefig('task_a_solution/plots/model_comparison_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved: task_a_solution/plots/model_comparison_confusion_matrices.png")
plt.close()

# 4. Training Time Comparison
training_times = [
    baseline_results['training_time'] + baseline_results['vectorization_time'],
    distilbert_results['training_time'],
    codebert_results['training_time']
]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(models, training_times, color=['#3498db', '#e74c3c', '#2ecc71'], 
              edgecolor='black', linewidth=2)

for bar, time_val in zip(bars, training_times):
    height = bar.get_height()
    minutes = time_val / 60
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{time_val:.1f}s\n({minutes:.2f}m)', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Training Time (seconds)', fontsize=12)
ax.set_title('Model Training Time Comparison', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('task_a_solution/plots/model_comparison_training_time.png', dpi=300, bbox_inches='tight')
print("✓ Saved: task_a_solution/plots/model_comparison_training_time.png")
plt.close()

# 5. Load predictions for ROC curves
baseline_preds = pd.read_csv('task_a_solution/results/baseline_predictions.csv') if 'baseline_predictions.csv' in str(baseline_results) else None
distilbert_preds = pd.read_csv('task_a_solution/results/distilbert_predictions.csv')
codebert_preds = pd.read_csv('task_a_solution/results/codebert_predictions.csv')

# ROC Curves Comparison
fig, ax = plt.subplots(figsize=(10, 8))

colors_roc = ['#3498db', '#e74c3c', '#2ecc71']
for i, (preds, model_name, color) in enumerate(zip(
    [distilbert_preds, codebert_preds], 
    ['DistilBERT', 'CodeBERT'],
    colors_roc[1:]
)):
    fpr, tpr, _ = roc_curve(preds['true_label'], preds['predicted_proba'])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, linewidth=2.5, label=f'{model_name} (AUC = {roc_auc:.4f})', color=color)

ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('task_a_solution/plots/model_comparison_roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: task_a_solution/plots/model_comparison_roc_curves.png")
plt.close()

# 6. Performance vs Complexity Trade-off
param_counts = [10000, 66955010, 124647170]  # Features/parameters for each model
fig, ax = plt.subplots(figsize=(10, 8))

scatter = ax.scatter(param_counts, f1_scores, 
                    s=[500, 800, 1000], 
                    c=colors_f1, 
                    alpha=0.6, 
                    edgecolors='black', 
                    linewidth=2)

for i, (x, y, model) in enumerate(zip(param_counts, f1_scores, models)):
    ax.annotate(f'{model}\nF1: {y:.4f}\nParams: {x:,}', 
                xy=(x, y), 
                xytext=(15, 15 if i == 0 else -15),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=colors_f1[i], alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax.set_xlabel('Model Complexity (Parameters/Features)', fontsize=12)
ax.set_ylabel('F1-Score', fontsize=12)
ax.set_title('Performance vs Complexity Trade-off', fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('task_a_solution/plots/model_comparison_complexity.png', dpi=300, bbox_inches='tight')
print("✓ Saved: task_a_solution/plots/model_comparison_complexity.png")
plt.close()

# 7. Error Analysis - where do models disagree?
print("\n[4] Error Analysis...")
distilbert_preds['model'] = 'distilbert'
codebert_preds['model'] = 'codebert'

# Find samples where models disagree
disagree = distilbert_preds[
    distilbert_preds['predicted_label'] != codebert_preds['predicted_label']
]
print(f"\nModels disagree on {len(disagree)} samples ({len(disagree)/len(distilbert_preds)*100:.2f}%)")

# Analyze errors
distilbert_errors = distilbert_preds[distilbert_preds['predicted_label'] != distilbert_preds['true_label']]
codebert_errors = codebert_preds[codebert_preds['predicted_label'] != codebert_preds['true_label']]

print(f"DistilBERT errors: {len(distilbert_errors)} ({len(distilbert_errors)/len(distilbert_preds)*100:.2f}%)")
print(f"CodeBERT errors: {len(codebert_errors)} ({len(codebert_errors)/len(codebert_preds)*100:.2f}%)")

# Error type analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# DistilBERT errors
distilbert_fp = len(distilbert_errors[(distilbert_errors['true_label'] == 0) & (distilbert_errors['predicted_label'] == 1)])
distilbert_fn = len(distilbert_errors[(distilbert_errors['true_label'] == 1) & (distilbert_errors['predicted_label'] == 0)])
axes[0].bar(['False Positives\n(Human→Machine)', 'False Negatives\n(Machine→Human)'], 
           [distilbert_fp, distilbert_fn], color=['#e74c3c', '#f39c12'], edgecolor='black', linewidth=2)
axes[0].set_title('DistilBERT Error Types', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=11)
for i, v in enumerate([distilbert_fp, distilbert_fn]):
    axes[0].text(i, v + 1, str(v), ha='center', fontweight='bold')

# CodeBERT errors
codebert_fp = len(codebert_errors[(codebert_errors['true_label'] == 0) & (codebert_errors['predicted_label'] == 1)])
codebert_fn = len(codebert_errors[(codebert_errors['true_label'] == 1) & (codebert_errors['predicted_label'] == 0)])
axes[1].bar(['False Positives\n(Human→Machine)', 'False Negatives\n(Machine→Human)'], 
           [codebert_fp, codebert_fn], color=['#2ecc71', '#27ae60'], edgecolor='black', linewidth=2)
axes[1].set_title('CodeBERT Error Types', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=11)
for i, v in enumerate([codebert_fp, codebert_fn]):
    axes[1].text(i, v + 1, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('task_a_solution/plots/model_error_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: task_a_solution/plots/model_error_analysis.png")
plt.close()

# Save comprehensive summary
summary = {
    'models_compared': 3,
    'best_model': 'CodeBERT',
    'best_f1_score': float(codebert_results['metrics']['f1_score']),
    'baseline_f1_score': float(baseline_results['metrics']['f1_score']),
    'improvement': float(codebert_results['metrics']['f1_score'] - baseline_results['metrics']['f1_score']),
    'all_metrics': metrics_df.to_dict('records'),
    'error_analysis': {
        'distilbert_total_errors': len(distilbert_errors),
        'distilbert_error_rate': float(len(distilbert_errors) / len(distilbert_preds)),
        'codebert_total_errors': len(codebert_errors),
        'codebert_error_rate': float(len(codebert_errors) / len(codebert_preds)),
        'models_disagree_count': len(disagree),
        'models_disagree_rate': float(len(disagree) / len(distilbert_preds))
    }
}

with open('task_a_solution/results/comprehensive_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\n✓ Saved: task_a_solution/results/comprehensive_summary.json")

print("\n" + "="*80)
print("COMPREHENSIVE ANALYSIS COMPLETE!")
print("="*80)
print(f"\nGenerated 7 comparison plots")
print(f"Best Model: {summary['best_model']}")
print(f"Best F1-Score: {summary['best_f1_score']:.4f}")
print(f"Improvement over baseline: +{summary['improvement']:.4f} ({summary['improvement']*100:.2f}%)")
