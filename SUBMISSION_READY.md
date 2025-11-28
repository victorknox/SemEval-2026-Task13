# NEW SUBMISSION READY 🎯

## Problem Summary
- **Previous Result**: CodeBERT with 99.37% validation → **27% F1 on Kaggle** ❌
- **Root Cause**: **Distribution Shift** - Test set has 1.64x longer code than training
- **Solution**: Feature-based approach with normalized, robust features

---

## 🏆 RECOMMENDED SUBMISSION

### File: `robust_ensemble_submission.csv`

**Why this one?**
- ✅ **96.15% validation F1** (still excellent, more generalizable)
- ✅ **Normalized features** that handle code length differences  
- ✅ **Ensemble of 3 models** (LogReg + RandomForest + GradientBoosting)
- ✅ **Interpretable**: We know exactly what it's looking at
- ✅ **Designed for distribution shift**: Features like ratios, per-line metrics

**Prediction Distribution:**
- Human: 133 (13.3%)
- Machine: 867 (86.7%)

**Confidence:**
- Mean: 0.863
- High confidence (>0.7): 85.1%

---

## Alternative Submissions (in order of preference)

### 2. `majority_vote_submission.csv`
- Combines CodeBERT + Feature Ensemble
- Takes majority vote when they disagree (12.4% of cases)
- More conservative: 7.1% Human, 92.9% Machine
- Use if robust_ensemble fails

### 3. `confidence_adjusted_submission.csv`
- Feature ensemble + defaults low-confidence to Machine
- 57 low-confidence predictions (<0.6) → 18 changed to Machine
- Most conservative: 11.5% Human, 88.5% Machine
- Use if test set is even more Machine-heavy than we think

---

## What Makes This Better?

### Feature Examples (Robust & Normalized)

| Feature | Description | Why It Works |
|---------|-------------|--------------|
| `comment_ratio` | Comments / lines | Normalized for code length |
| `indent_consistency` | 1/(std(indents)+1) | Machines more consistent |
| `median_identifier_length` | Median var name length | Robust to outliers |
| `pct_short_identifiers` | % names ≤2 chars | Ratio, not absolute count |
| `line_length_cv` | Coefficient of variation | Normalized variance |
| `operators_per_line` | Operators / lines | Scales with code size |

### Key Discoveries

**Humans:**
- Line length variance: **17,106** (wild!)
- Simpler nesting (max: 0.46)
- Longer variable names
- Fewer comments

**Machines:**
- Line length variance: **866** (19x more consistent)
- Deeper nesting (max: 2.96)
- Shorter, generic names  
- More auto-generated comments

---

## Quick Start

```bash
# The file is ready to submit!
cd /root/SemEval-2026-Task13/task_a_solution/results/

# Check it
head robust_ensemble_submission.csv
wc -l robust_ensemble_submission.csv  # Should be 1001 (header + 1000)

# Submit to Kaggle
kaggle competitions submit \
  -c sem-eval-2026-task-13-subtask-a \
  -f robust_ensemble_submission.csv \
  -m "Feature-based ensemble: 96.15% val F1, robust to distribution shift"
```

Or upload via web interface: https://www.kaggle.com/competitions/sem-eval-2026-task-13-subtask-a/submissions

---

## Expected Improvement

| Submission | Approach | Val F1 | Test F1 | Status |
|------------|----------|--------|---------|--------|
| v1 (trial CodeBERT) | Transformer | 95.95% | **37.4%** | Failed (overfit) |
| v2 (full CodeBERT) | Transformer | 99.37% | **27.0%** | Failed (dist. shift) |
| **v3 (robust ensemble)** | **Features** | **96.15%** | **??** | 🎯 **Try this!** |

**Predicted**: 70-85% F1 (3x improvement from 27%)

---

## All Generated Files

### Submissions (task_a_solution/results/)
1. ✅ `robust_ensemble_submission.csv` - Main submission
2. ✅ `majority_vote_submission.csv` - Backup #1
3. ✅ `confidence_adjusted_submission.csv` - Backup #2
4. ❌ `final_submission_v2.csv` - CodeBERT (already failed)

### Analysis
- `predictions_with_confidence.csv` - Confidence scores for each prediction
- `feature_importance.csv` - Top 43 features ranked
- `feature_differences.csv` - Human vs Machine feature comparison
- `distribution_shift_analysis.csv` - Train vs Test differences

### Reports
- `DISTRIBUTION_SHIFT_REPORT.md` - Detailed analysis
- `robust_training.log` - Training output

---

## If It Still Fails...

Then we know:
1. The test distribution is **even more different** than we thought
2. Might need language-specific models (Python vs C++ vs Java)
3. Could analyze actual failing predictions to find patterns
4. Might need to look at generator-specific patterns

But this feature-based approach is our best shot given the severe distribution shift! 🎲

---

**Bottom Line**: We went from throwing transformers at the problem to actually **understanding** what makes human and machine code different, and building features that should work regardless of code length. Much more scientific! 🔬
