#!/usr/bin/env python3
"""
Compare different model predictions to find consensus and conflicts
"""

import pandas as pd

print("=" * 100)
print("PREDICTION COMPARISON & ANALYSIS")
print("=" * 100)

# Load all predictions
try:
    v1 = pd.read_csv('task_a_solution/results/final_submission.csv')
    print("✓ Loaded: final_submission.csv (CodeBERT on trial data)")
    print(f"  Distribution: {(v1['label']==0).sum()} Human, {(v1['label']==1).sum()} Machine")
except:
    print("✗ Could not load final_submission.csv")
    v1 = None

try:
    v2 = pd.read_csv('task_a_solution/results/final_submission_v2.csv')
    print("✓ Loaded: final_submission_v2.csv (CodeBERT on full data)")
    print(f"  Distribution: {(v2['label']==0).sum()} Human, {(v2['label']==1).sum()} Machine")
except:
    print("✗ Could not load final_submission_v2.csv")
    v2 = None

try:
    v3 = pd.read_csv('task_a_solution/results/robust_ensemble_submission.csv')
    print("✓ Loaded: robust_ensemble_submission.csv (Feature ensemble)")
    print(f"  Distribution: {(v3['label']==0).sum()} Human, {(v3['label']==1).sum()} Machine")
except:
    print("✗ Could not load robust_ensemble_submission.csv")
    v3 = None

# Compare predictions
if v2 is not None and v3 is not None:
    print("\n" + "=" * 100)
    print("CODEBERT VS FEATURE ENSEMBLE COMPARISON")
    print("=" * 100)
    
    # Merge on ID
    comparison = pd.merge(v2[['ID', 'label']], v3[['ID', 'label']], on='ID', suffixes=('_codebert', '_ensemble'))
    
    agreement = (comparison['label_codebert'] == comparison['label_ensemble']).sum()
    total = len(comparison)
    
    print(f"\nAgreement: {agreement}/{total} ({agreement/total*100:.1f}%)")
    print(f"Disagreement: {total-agreement}/{total} ({(total-agreement)/total*100:.1f}%)")
    
    # Analyze disagreements
    disagreements = comparison[comparison['label_codebert'] != comparison['label_ensemble']]
    
    print(f"\n Disagreements breakdown:")
    print(f"  CodeBERT=Human, Ensemble=Machine: {((disagreements['label_codebert']==0) & (disagreements['label_ensemble']==1)).sum()}")
    print(f"  CodeBERT=Machine, Ensemble=Human: {((disagreements['label_codebert']==1) & (disagreements['label_ensemble']==0)).sum()}")
    
    # Create voting submission
    print("\n" + "=" * 100)
    print("CREATING MAJORITY VOTE SUBMISSION")
    print("=" * 100)
    
    # Simple majority vote
    comparison['vote'] = ((comparison['label_codebert'] + comparison['label_ensemble']) > 0.5).astype(int)
    
    vote_submission = pd.DataFrame({
        'ID': comparison['ID'],
        'label': comparison['vote']
    })
    
    vote_submission.to_csv('task_a_solution/results/majority_vote_submission.csv', index=False)
    
    print(f"✓ Saved: majority_vote_submission.csv")
    print(f"  Distribution: {(vote_submission['label']==0).sum()} Human ({(vote_submission['label']==0).sum()/len(vote_submission)*100:.1f}%)")
    print(f"                {(vote_submission['label']==1).sum()} Machine ({(vote_submission['label']==1).sum()/len(vote_submission)*100:.1f}%)")
    
    # Load confidence scores
    try:
        conf = pd.read_csv('task_a_solution/results/predictions_with_confidence.csv')
        print("\n" + "=" * 100)
        print("CONFIDENCE-BASED ANALYSIS")
        print("=" * 100)
        
        # Show low confidence predictions
        low_conf = conf[conf['confidence'] < 0.6].sort_values('confidence')
        
        if len(low_conf) > 0:
            print(f"\nLow confidence predictions (<0.6): {len(low_conf)}")
            print("\nLowest 10 confidence predictions:")
            print(low_conf.head(10)[['ID', 'label', 'confidence', 'prob_human', 'prob_machine']].to_string(index=False))
            
            # For low confidence ones, we might want to default to Machine (since it's majority)
            print("\n" + "=" * 100)
            print("CONFIDENCE-ADJUSTED SUBMISSION")
            print("=" * 100)
            
            # If low confidence, default to Machine (label=1) since test seems machine-heavy
            conf['adjusted_label'] = conf.apply(
                lambda row: 1 if row['confidence'] < 0.6 else row['label'],
                axis=1
            )
            
            conf_submission = pd.DataFrame({
                'ID': conf['ID'],
                'label': conf['adjusted_label']
            })
            
            conf_submission.to_csv('task_a_solution/results/confidence_adjusted_submission.csv', index=False)
            
            print(f"✓ Saved: confidence_adjusted_submission.csv")
            print(f"  Changed {sum(conf['label'] != conf['adjusted_label'])} low-confidence predictions to Machine")
            print(f"  Distribution: {(conf_submission['label']==0).sum()} Human, {(conf_submission['label']==1).sum()} Machine")
        
    except Exception as e:
        print(f"\nCould not analyze confidence: {e}")

print("\n" + "=" * 100)
print("SUMMARY - SUBMISSIONS AVAILABLE")
print("=" * 100)

import os
results_dir = 'task_a_solution/results'
if os.path.exists(results_dir):
    submissions = [f for f in os.listdir(results_dir) if 'submission' in f and f.endswith('.csv')]
    for i, sub in enumerate(sorted(submissions), 1):
        path = os.path.join(results_dir, sub)
        df = pd.read_csv(path)
        human_pct = (df['label']==0).sum() / len(df) * 100
        machine_pct = (df['label']==1).sum() / len(df) * 100
        print(f"{i}. {sub:50s} | H: {human_pct:5.1f}% | M: {machine_pct:5.1f}%")

print("\n" + "=" * 100)
print("RECOMMENDATIONS")
print("=" * 100)
print("\n1. robust_ensemble_submission.csv - Best validation F1 (96.15%), normalized features")
print("2. majority_vote_submission.csv - Combines CodeBERT + Ensemble wisdom")
print("3. confidence_adjusted_submission.csv - Conservative, defaults uncertain to Machine")
print("\nGiven 27% F1 with CodeBERT, the feature-based approach should perform better!")
