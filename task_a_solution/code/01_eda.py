"""
Exploratory Data Analysis for SemEval-2026 Task 13 - Task A
Binary Machine-Generated Code Detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directories
import os
os.makedirs('task_a_solution/plots', exist_ok=True)
os.makedirs('task_a_solution/results', exist_ok=True)

print("="*80)
print("EXPLORATORY DATA ANALYSIS - TASK A")
print("="*80)

# Load data
print("\n[1] Loading data...")
df = pd.read_parquet('task_A/task_a_trial.parquet')
print(f"✓ Loaded {len(df):,} samples")
print(f"✓ Columns: {df.columns.tolist()}")

# Load label mappings
with open('task_A/id_to_label.json', 'r') as f:
    id_to_label = json.load(f)
with open('task_A/label_to_id.json', 'r') as f:
    label_to_id = json.load(f)

print(f"\n✓ Label mapping: {id_to_label}")

# Basic statistics
print("\n" + "="*80)
print("[2] BASIC STATISTICS")
print("="*80)
print(f"\nDataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\nMissing values:")
print(df.isnull().sum())

# Label distribution
print("\n" + "="*80)
print("[3] LABEL DISTRIBUTION")
print("="*80)
label_counts = df['label'].value_counts().sort_index()
print("\nLabel counts:")
for label_id, count in label_counts.items():
    label_name = id_to_label[str(label_id)]
    percentage = (count / len(df)) * 100
    print(f"  {label_name} (label {label_id}): {count:,} ({percentage:.2f}%)")

# Visualize label distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
label_names = [id_to_label[str(i)] for i in label_counts.index]
axes[0].bar(label_names, label_counts.values, color=['#3498db', '#e74c3c'])
axes[0].set_title('Label Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_xlabel('Label', fontsize=12)
for i, v in enumerate(label_counts.values):
    axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')

# Pie chart
axes[1].pie(label_counts.values, labels=label_names, autopct='%1.1f%%',
            colors=['#3498db', '#e74c3c'], startangle=90)
axes[1].set_title('Label Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('task_a_solution/plots/01_label_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: task_a_solution/plots/01_label_distribution.png")
plt.close()

# Language distribution
print("\n" + "="*80)
print("[4] PROGRAMMING LANGUAGE DISTRIBUTION")
print("="*80)
lang_counts = df['language'].value_counts()
print(f"\nLanguages found: {lang_counts.index.tolist()}")
print("\nLanguage distribution:")
for lang, count in lang_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {lang}: {count:,} ({percentage:.2f}%)")

# Visualize language distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overall language distribution
axes[0].bar(lang_counts.index, lang_counts.values, color=sns.color_palette("Set2", len(lang_counts)))
axes[0].set_title('Programming Language Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_xlabel('Programming Language', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(lang_counts.values):
    axes[0].text(i, v + 30, str(v), ha='center', fontweight='bold')

# Language distribution by label
lang_label_df = df.groupby(['language', 'label']).size().unstack(fill_value=0)
lang_label_df.plot(kind='bar', stacked=False, ax=axes[1], 
                   color=['#3498db', '#e74c3c'], width=0.7)
axes[1].set_title('Language Distribution by Label', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_xlabel('Programming Language', fontsize=12)
axes[1].legend(['Human', 'Machine'], title='Label')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('task_a_solution/plots/02_language_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: task_a_solution/plots/02_language_distribution.png")
plt.close()

# Generator distribution
print("\n" + "="*80)
print("[5] GENERATOR DISTRIBUTION")
print("="*80)
gen_counts = df['generator'].value_counts()
print(f"\nUnique generators: {len(gen_counts)}")
print(f"\nTop 10 generators:")
for i, (gen, count) in enumerate(gen_counts.head(10).items(), 1):
    percentage = (count / len(df)) * 100
    print(f"  {i}. {gen}: {count:,} ({percentage:.2f}%)")

# Visualize generator distribution
fig, ax = plt.subplots(figsize=(12, 6))
top_gens = gen_counts.head(15)
ax.barh(range(len(top_gens)), top_gens.values, color=sns.color_palette("viridis", len(top_gens)))
ax.set_yticks(range(len(top_gens)))
ax.set_yticklabels(top_gens.index)
ax.set_xlabel('Count', fontsize=12)
ax.set_title('Top 15 Generators', fontsize=14, fontweight='bold')
ax.invert_yaxis()
for i, v in enumerate(top_gens.values):
    ax.text(v + 20, i, str(v), va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('task_a_solution/plots/03_generator_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: task_a_solution/plots/03_generator_distribution.png")
plt.close()

# Code length analysis
print("\n" + "="*80)
print("[6] CODE LENGTH ANALYSIS")
print("="*80)
df['code_length'] = df['code'].apply(len)
df['code_lines'] = df['code'].apply(lambda x: len(x.split('\n')))

print("\nCode length statistics (characters):")
print(df.groupby('label')['code_length'].describe())

print("\nCode lines statistics:")
print(df.groupby('label')['code_lines'].describe())

# Visualize code length distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Character count distribution
for label in [0, 1]:
    label_name = id_to_label[str(label)]
    data = df[df['label'] == label]['code_length']
    axes[0, 0].hist(data, bins=50, alpha=0.6, label=label_name, edgecolor='black')
axes[0, 0].set_xlabel('Code Length (characters)', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Code Length Distribution', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].set_xlim(0, df['code_length'].quantile(0.95))

# Line count distribution
for label in [0, 1]:
    label_name = id_to_label[str(label)]
    data = df[df['label'] == label]['code_lines']
    axes[0, 1].hist(data, bins=50, alpha=0.6, label=label_name, edgecolor='black')
axes[0, 1].set_xlabel('Code Lines', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title('Code Lines Distribution', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].set_xlim(0, df['code_lines'].quantile(0.95))

# Box plots
df_plot = df.copy()
df_plot['label_name'] = df_plot['label'].map(lambda x: id_to_label[str(x)])
sns.boxplot(data=df_plot, x='label_name', y='code_length', ax=axes[1, 0])
axes[1, 0].set_ylabel('Code Length (characters)', fontsize=12)
axes[1, 0].set_xlabel('Label', fontsize=12)
axes[1, 0].set_title('Code Length by Label (Boxplot)', fontsize=14, fontweight='bold')
axes[1, 0].set_ylim(0, df['code_length'].quantile(0.95))

sns.boxplot(data=df_plot, x='label_name', y='code_lines', ax=axes[1, 1])
axes[1, 1].set_ylabel('Code Lines', fontsize=12)
axes[1, 1].set_xlabel('Label', fontsize=12)
axes[1, 1].set_title('Code Lines by Label (Boxplot)', fontsize=14, fontweight='bold')
axes[1, 1].set_ylim(0, df['code_lines'].quantile(0.95))

plt.tight_layout()
plt.savefig('task_a_solution/plots/04_code_length_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: task_a_solution/plots/04_code_length_analysis.png")
plt.close()

# Language vs Label heatmap
print("\n" + "="*80)
print("[7] LANGUAGE VS LABEL CROSS-TABULATION")
print("="*80)
lang_label_ct = pd.crosstab(df['language'], df['label'])
lang_label_ct.columns = [id_to_label[str(c)] for c in lang_label_ct.columns]
print("\n", lang_label_ct)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(lang_label_ct, annot=True, fmt='d', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Count'})
ax.set_title('Language vs Label Heatmap', fontsize=14, fontweight='bold')
ax.set_ylabel('Programming Language', fontsize=12)
ax.set_xlabel('Label', fontsize=12)
plt.tight_layout()
plt.savefig('task_a_solution/plots/05_language_label_heatmap.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: task_a_solution/plots/05_language_label_heatmap.png")
plt.close()

# Sample code examples
print("\n" + "="*80)
print("[8] SAMPLE CODE EXAMPLES")
print("="*80)

for label in [0, 1]:
    label_name = id_to_label[str(label)]
    print(f"\n{'='*60}")
    print(f"Example of {label_name.upper()} code:")
    print(f"{'='*60}")
    sample = df[df['label'] == label].iloc[0]
    print(f"Language: {sample['language']}")
    print(f"Generator: {sample['generator']}")
    print(f"Code length: {sample['code_length']} characters, {sample['code_lines']} lines")
    print(f"\nCode snippet (first 500 chars):")
    print("-" * 60)
    print(sample['code'][:500])
    print("-" * 60)

# Save summary statistics
print("\n" + "="*80)
print("[9] SAVING SUMMARY STATISTICS")
print("="*80)

summary_stats = {
    'total_samples': len(df),
    'label_distribution': label_counts.to_dict(),
    'language_distribution': lang_counts.to_dict(),
    'unique_generators': len(gen_counts),
    'code_length_stats': {
        'mean': float(df['code_length'].mean()),
        'median': float(df['code_length'].median()),
        'std': float(df['code_length'].std()),
        'min': int(df['code_length'].min()),
        'max': int(df['code_length'].max())
    },
    'code_lines_stats': {
        'mean': float(df['code_lines'].mean()),
        'median': float(df['code_lines'].median()),
        'std': float(df['code_lines'].std()),
        'min': int(df['code_lines'].min()),
        'max': int(df['code_lines'].max())
    }
}

with open('task_a_solution/results/eda_summary.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print("✓ Saved: task_a_solution/results/eda_summary.json")

# Save processed data
df.to_csv('task_a_solution/data/task_a_trial_processed.csv', index=False)
print("✓ Saved: task_a_solution/data/task_a_trial_processed.csv")

print("\n" + "="*80)
print("EDA COMPLETE!")
print("="*80)
print(f"\nGenerated {5} plots in task_a_solution/plots/")
print("Generated summary statistics in task_a_solution/results/")
