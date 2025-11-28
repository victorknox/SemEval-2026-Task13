#!/usr/bin/env python3
"""
MEGA ENSEMBLE: Combine ALL approaches
"""

import pandas as pd
import numpy as np

print("=" * 100)
print("MEGA ENSEMBLE: COMBINING ALL APPROACHES")
print("=" * 100)

# Load all predictions
submissions = {
    'CodeBERT': 'task_a_solution/results/final_submission_v2.csv',
    'Robust_Ensemble': 'task_a_solution/results/robust_ensemble_submission.csv',
    'Per_Language': 'task_a_solution/results/per_language_submission.csv',
    'Complexity': 'task_a_solution/results/complexity_based_submission.csv',
    'Anomaly': 'task_a_solution/results/anomaly_detection_submission.csv',
    'Patterns': 'task_a_solution/results/pattern_analysis_submission.csv',
    'Rules': 'task_a_solution/results/rule_based_submission.csv',
    'Flipped': 'task_a_solution/results/flipped_labels_submission.csv',
}

print("\nLoading all submissions...")
all_preds = {}

for name, path in submissions.items():
    try:
        df = pd.read_csv(path)
        all_preds[name] = df.set_index('ID')['label']
        human_pct = (df['label'] == 0).sum() / len(df) * 100
        print(f"  {name:20s}: {human_pct:5.1f}% Human | {100-human_pct:5.1f}% Machine")
    except Exception as e:
        print(f"  {name:20s}: ERROR - {e}")

# Combine predictions
print("\n" + "=" * 100)
print("ENSEMBLE STRATEGIES")
print("=" * 100)

# Get IDs
test_df = pd.read_parquet('Task_A/test.parquet')
ids = test_df['ID'].values

# Strategy 1: Majority Vote (ALL models)
print("\n1. MAJORITY VOTE (all models)")
pred_matrix = np.array([all_preds[name].loc[ids].values for name in all_preds.keys()]).T
majority_vote = (pred_matrix.sum(axis=1) >= len(all_preds) / 2).astype(int)

majority_submission = pd.DataFrame({
    'ID': ids,
    'label': majority_vote
})
majority_submission.to_csv('task_a_solution/results/mega_ensemble_majority.csv', index=False)

human_pct = (majority_vote == 0).sum() / len(majority_vote) * 100
print(f"   Distribution: {human_pct:.1f}% Human, {100-human_pct:.1f}% Machine")

# Strategy 2: Weighted Vote (give more weight to better validation F1)
weights = {
    'CodeBERT': 0.99,
    'Robust_Ensemble': 0.96,
    'Per_Language': 0.95,
    'Complexity': 0.93,
    'Anomaly': 0.68,
    'Patterns': 0.85,
    'Rules': 0.53,
    'Flipped': 0.05  # Low weight since it's inverse
}

print("\n2. WEIGHTED VOTE (by validation F1)")
weighted_scores = np.zeros(len(ids))
total_weight = 0

for name in all_preds.keys():
    if name in weights:
        weight = weights[name]
        preds = all_preds[name].loc[ids].values
        weighted_scores += preds * weight
        total_weight += weight

weighted_vote = (weighted_scores >= total_weight / 2).astype(int)

weighted_submission = pd.DataFrame({
    'ID': ids,
    'label': weighted_vote
})
weighted_submission.to_csv('task_a_solution/results/mega_ensemble_weighted.csv', index=False)

human_pct = (weighted_vote == 0).sum() / len(weighted_vote) * 100
print(f"   Distribution: {human_pct:.1f}% Human, {100-human_pct:.1f}% Machine")

# Strategy 3: Conservative (only if ALL strong models agree on Machine)
print("\n3. CONSERVATIVE (all strong models agree on Machine)")
strong_models = ['CodeBERT', 'Robust_Ensemble', 'Per_Language', 'Complexity']
strong_pred_matrix = np.array([all_preds[name].loc[ids].values for name in strong_models]).T
conservative = (strong_pred_matrix.sum(axis=1) == len(strong_models)).astype(int)  # All say Machine -> 1

conservative_submission = pd.DataFrame({
    'ID': ids,
    'label': conservative
})
conservative_submission.to_csv('task_a_solution/results/mega_ensemble_conservative.csv', index=False)

human_pct = (conservative == 0).sum() / len(conservative) * 100
print(f"   Distribution: {human_pct:.1f}% Human, {100-human_pct:.1f}% Machine")

# Strategy 4: Diversity (if models disagree, predict Human)
print("\n4. DIVERSITY-BASED (disagreement -> Human)")
agreement_score = pred_matrix.std(axis=1)  # Higher std = more disagreement
diversity_vote = ((agreement_score < 0.3) & (pred_matrix.mean(axis=1) < 0.5)).astype(int)

diversity_submission = pd.DataFrame({
    'ID': ids,
    'label': diversity_vote
})
diversity_submission.to_csv('task_a_solution/results/mega_ensemble_diversity.csv', index=False)

human_pct = (diversity_vote == 0).sum() / len(diversity_vote) * 100
print(f"   Distribution: {human_pct:.1f}% Human, {100-human_pct:.1f}% Machine")

# Analyze disagreement
print("\n" + "=" * 100)
print("DISAGREEMENT ANALYSIS")
print("=" * 100)

print(f"\nTotal samples: {len(ids)}")
print(f"All models agree: {(pred_matrix.std(axis=1) == 0).sum()} ({(pred_matrix.std(axis=1) == 0).sum()/len(ids)*100:.1f}%)")
print(f"High disagreement (std>0.4): {(pred_matrix.std(axis=1) > 0.4).sum()} ({(pred_matrix.std(axis=1) > 0.4).sum()/len(ids)*100:.1f}%)")

print("\n" + "=" * 100)
print("SUMMARY - NEW SUBMISSIONS")
print("=" * 100)

print("\n4 NEW mega-ensemble files created:")
print("  1. mega_ensemble_majority.csv - Simple majority vote")
print("  2. mega_ensemble_weighted.csv - Weighted by validation F1")
print("  3. mega_ensemble_conservative.csv - Only if strong models agree")
print("  4. mega_ensemble_diversity.csv - Accounts for disagreement")

print("\n" + "=" * 100)
print("TOTAL SUBMISSIONS AVAILABLE")
print("=" * 100)

import os
results_dir = 'task_a_solution/results'
all_subs = sorted([f for f in os.listdir(results_dir) if 'submission' in f and f.endswith('.csv')])

print(f"\n{len(all_subs)} submission files ready!")
for i, sub in enumerate(all_subs, 1):
    path = os.path.join(results_dir, sub)
    try:
        df = pd.read_csv(path)
        human_pct = (df['label']==0).sum() / len(df) * 100
        print(f"{i:2d}. {sub:45s} | H: {human_pct:5.1f}% | M: {100-human_pct:5.1f}%")
    except:
        pass
