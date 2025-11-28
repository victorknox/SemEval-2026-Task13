#!/usr/bin/env python3
"""
Create visual summary of our approach
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

# Create comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Model Performance Comparison
models = ['CodeBERT\n(trial)', 'CodeBERT\n(full)', 'Robust\nEnsemble']
val_f1 = [0.9595, 0.9937, 0.9615]
test_f1 = [0.374, 0.27, None]  # Unknown for ensemble

ax = axes[0, 0]
x = range(len(models))
bars1 = ax.bar([i-0.2 for i in x], val_f1, 0.4, label='Validation F1', color='#2ecc71')
bars2 = ax.bar([i+0.2 for i in x[:2]], test_f1[:2], 0.4, label='Test F1', color='#e74c3c')
ax.bar([2+0.2], [0], 0.4, color='#95a5a6', label='Test F1 (TBD)')

ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim([0, 1.1])
ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Target: 90%')

for i, v in enumerate(val_f1):
    ax.text(i-0.2, v+0.02, f'{v:.1%}', ha='center', fontsize=9)
for i, v in enumerate(test_f1[:2]):
    if v:
        ax.text(i+0.2, v+0.02, f'{v:.1%}', ha='center', fontsize=9, color='red')

# 2. Distribution Shift
ax = axes[0, 1]
categories = ['Code Length\n(chars)', 'Num Lines']
train_vals = [835, 35.3]
test_vals = [1370, 40.7]

x = range(len(categories))
width = 0.35
ax.bar([i-width/2 for i in x], train_vals, width, label='Training', color='#3498db')
ax.bar([i+width/2 for i in x], test_vals, width, label='Test Set', color='#e67e22')

ax.set_ylabel('Average Value', fontsize=12)
ax.set_title('Distribution Shift: Train vs Test', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Add ratio annotations
for i, (t, ts) in enumerate(zip(train_vals, test_vals)):
    ratio = ts/t
    ax.text(i, max(t, ts)+50, f'{ratio:.2f}x', ha='center', fontsize=10, fontweight='bold', color='red')

# 3. Feature Importance (Top 10)
ax = axes[1, 0]
features = ['max_nesting', 'indent_variance', 'max_indent', 'avg_indent', 
            'has_return', 'blank_line_ratio', 'num_hash_comments', 'num_blank_lines',
            'has_if', 'special_char_ratio']
importance = [0.164, 0.161, 0.105, 0.084, 0.059, 0.054, 0.049, 0.045, 0.037, 0.033]

ax.barh(features[::-1], importance[::-1], color='#9b59b6')
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
ax.set_xlim([0, max(importance)*1.1])

# 4. Prediction Distribution
ax = axes[1, 1]
submissions = ['v1\n(trial)', 'v2\n(full)', 'v3\n(ensemble)', 'v3\n(majority)', 'v3\n(conf adj)']
human_pct = [63.1, 13.3, 13.3, 7.1, 11.5]
machine_pct = [36.9, 86.7, 86.7, 92.9, 88.5]

x = range(len(submissions))
ax.bar(x, human_pct, label='Human', color='#3498db')
ax.bar(x, machine_pct, bottom=human_pct, label='Machine', color='#e74c3c')

ax.set_ylabel('Percentage', fontsize=12)
ax.set_title('Prediction Distribution Across Submissions', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(submissions, fontsize=9)
ax.legend()
ax.set_ylim([0, 100])

# Add percentage labels
for i, (h, m) in enumerate(zip(human_pct, machine_pct)):
    ax.text(i, h/2, f'{h:.1f}%', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    ax.text(i, h + m/2, f'{m:.1f}%', ha='center', va='center', fontsize=8, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('task_a_solution/results/approach_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: task_a_solution/results/approach_summary.png")
plt.close()

# Create feature comparison plot
fig, ax = plt.subplots(figsize=(12, 8))

features = ['line_length_variance', 'indent_variance', 'max_nesting', 'num_hash_comments',
            'avg_indent', 'num_unique_identifiers', 'max_identifier_length']
human_means = [17106, 3.10, 0.46, 0.68, 1.84, 27.13, 9.19]
machine_means = [866, 15.82, 2.96, 3.89, 4.27, 50.66, 13.91]

x = range(len(features))
width = 0.35

# Normalize for visualization (some values very different scales)
# Use log scale
import numpy as np

ax.bar([i-width/2 for i in x], human_means, width, label='Human Code', color='#3498db', alpha=0.7)
ax.bar([i+width/2 for i in x], machine_means, width, label='Machine Code', color='#e74c3c', alpha=0.7)

ax.set_ylabel('Mean Value (note: mixed scales)', fontsize=12)
ax.set_title('Feature Comparison: Human vs Machine Code', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(features, rotation=45, ha='right')
ax.legend()
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('task_a_solution/results/feature_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: task_a_solution/results/feature_comparison.png")

print("\n" + "="*80)
print("VISUALIZATIONS CREATED")
print("="*80)
print("\n1. approach_summary.png - 4-panel overview")
print("2. feature_comparison.png - Human vs Machine features")
print("\nThese visualizations show:")
print("  • Why CodeBERT failed (distribution shift)")
print("  • How feature-based approach differs")
print("  • Key patterns that distinguish human/machine code")
