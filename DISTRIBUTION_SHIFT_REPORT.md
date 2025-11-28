# Distribution Shift Analysis & New Approach Report

## Problem Diagnosis

### Initial Results
- **CodeBERT v1** (10K trial data): 95.95% F1 validation → **0.37375 Kaggle** (severe overfit)
- **CodeBERT v2** (500K full data): 99.37% F1 validation → **0.27 Kaggle F1** (still failed!)

### Root Cause: **DISTRIBUTION SHIFT**

The test set has significantly different characteristics from the training data:

| Metric | Training Data | Test Set | Ratio |
|--------|--------------|----------|-------|
| Average Code Length | 835 chars | 1,370 chars | **1.64x longer** |
| Average Lines | 35.3 lines | 40.7 lines | **1.15x more** |
| Samples | 500,000 | 1,000 | - |

**Key Insight**: The test set contains substantially longer, more complex code snippets that don't match the training distribution!

---

## New Approach: Feature Engineering

Instead of relying on black-box transformers that memorize patterns, we switched to **interpretable, robust features** that normalize for code length and generalize better.

### Top Discriminative Features

From analyzing 50K training samples, we found:

| Feature | Human Mean | Machine Mean | Key Insight |
|---------|-----------|--------------|-------------|
| `line_length_variance` | 17,106 | 866 | **19.8x difference** - Humans have wildly varying line lengths |
| `indent_variance` | 3.10 | 15.82 | Machines have more complex nesting |
| `max_nesting` | 0.46 | 2.96 | Machine code is more deeply nested |
| `num_hash_comments` | 0.68 | 3.89 | Machine-generated code has more comments |
| `identifier_diversity` | Higher in humans | Lower in machines | Humans use more varied variable names |

### Robust Features Used

Instead of absolute counts (which vary with code length), we used **normalized ratios**:

1. **Comment ratio**: Comments per line (not total comments)
2. **Blank line ratio**: Blank lines / total lines
3. **Indentation consistency**: `1 / (std(indents) + 1)` - higher means more consistent
4. **Identifier patterns**:
   - Median identifier length
   - % short identifiers (≤2 chars)
   - Identifier diversity ratio
5. **Line length coefficient of variation**: `std(line_len) / mean(line_len)` - normalized
6. **Operators per line**: Total operators / non-empty lines
7. **Structural features**: Has function/class/imports (binary)

---

## Results: Robust Ensemble Classifier

### Model Architecture
- **Ensemble** of 3 classifiers voting together:
  1. Logistic Regression (C=0.1)
  2. Random Forest (200 trees, depth=12)
  3. Gradient Boosting (200 estimators, depth=5)

### Training Strategy
- **Balanced sampling**: 50K human + 50K machine (100K total)
- Avoids bias toward either class
- Faster training while maintaining diversity

### Performance

**Validation Results (100K samples)**:
```
              precision    recall  f1-score   support

       Human       0.95      0.96      0.96     47,695
     Machine       0.97      0.96      0.96     52,305

    accuracy                           0.96    100,000
```

- **F1-Score: 96.15%** (vs 99.37% for CodeBERT, but should generalize better!)
- **High confidence predictions**: 90.9% of samples (>0.7 probability)
- **F1 on high-confidence**: 98.20%

### Test Predictions

- **Total**: 1,000 samples
- **Distribution**: 13.3% Human, 86.7% Machine
- **Mean confidence**: 0.863
- **High confidence**: 85.1% of predictions

---

## Multiple Submission Strategies

We created **4 different submissions** to hedge our bets:

### 1. `robust_ensemble_submission.csv` ⭐ **RECOMMENDED**
- Pure feature-based ensemble
- 96.15% validation F1
- Should generalize best to shifted distribution
- **13.3% Human, 86.7% Machine**

### 2. `majority_vote_submission.csv`
- Combines CodeBERT + Feature Ensemble predictions
- Takes majority vote from both approaches
- **7.1% Human, 92.9% Machine**
- More conservative (predicts more Machine when uncertain)

### 3. `confidence_adjusted_submission.csv`
- Based on robust ensemble
- For low-confidence predictions (<0.6), defaults to Machine
- Changed 18 predictions
- **11.5% Human, 88.5% Machine**

### 4. `final_submission_v2.csv`
- CodeBERT on full 500K data
- 99.37% validation F1
- Already failed with 27% test F1
- **13.3% Human, 86.7% Machine**

---

## Model Agreement Analysis

**CodeBERT vs Feature Ensemble**:
- **Agreement**: 87.6% (876/1,000 samples)
- **Disagreement**: 12.4% (124/1,000 samples)
  - CodeBERT says Human, Ensemble says Machine: 62
  - CodeBERT says Machine, Ensemble says Human: 62
  - Perfectly balanced disagreement!

This suggests the models are capturing different aspects of the code.

---

## Why Feature-Based Should Work Better

### Problems with Transformer Approach
1. **Memorization**: CodeBERT learned patterns specific to training data length/complexity
2. **Distribution sensitivity**: Transformers excel when test ≈ train distribution
3. **Black box**: Hard to understand what failed
4. **No normalization**: Absolute features (code length, token counts) shifted significantly

### Advantages of Feature-Based Approach
1. **Normalized features**: Ratios and per-line metrics handle longer code
2. **Interpretable**: We know exactly what the model is looking at
3. **Robust**: Features designed to work across code lengths
4. **Ensemble diversity**: Multiple models reduce overfitting
5. **Generalizable patterns**: Focus on fundamental code style differences

---

## Key Insights from Analysis

### What Makes Human Code Different?
1. **Inconsistent formatting**: Humans have high line-length variance (17K vs 866)
2. **Simpler structure**: Less nesting, simpler indentation patterns
3. **Descriptive names**: Longer, more varied identifier names
4. **Fewer comments**: Machine generators add more explanatory comments

### What Makes Machine Code Different?
1. **Consistent formatting**: Lower line-length variance, uniform style
2. **Complex structure**: More nested blocks, deeper indentation
3. **Generic names**: Shorter, more repetitive variable names
4. **More comments**: Auto-generated documentation/explanations
5. **More functions**: Machine code tends to have `def` and `return` statements more often

---

## Next Steps

### Recommended Submission Order:
1. **Try `robust_ensemble_submission.csv` first** - Best validation, designed for shift
2. **If that fails, try `majority_vote_submission.csv`** - Combines both approaches
3. **If still low, try `confidence_adjusted_submission.csv`** - Most conservative

### If All Fail:
The distribution shift might be even more severe than we thought. Consider:
- Analyzing which specific predictions are failing
- Looking at confusion between different programming languages
- Checking if test set has different source/generator distributions
- Building language-specific classifiers

---

## Files Generated

### Analysis Files
- `feature_importance.csv` - Ranking of all 43 features
- `feature_differences.csv` - Human vs Machine feature comparisons
- `sample_features.csv` - Example feature extractions
- `insights.txt` - Key human vs machine patterns
- `distribution_shift_analysis.csv` - Train vs Test comparison
- `predictions_with_confidence.csv` - All predictions with probabilities

### Submission Files
- `robust_ensemble_submission.csv` ⭐
- `majority_vote_submission.csv`
- `confidence_adjusted_submission.csv`
- `final_submission_v2.csv` (CodeBERT - already tested, 27% F1)

### Training Logs
- `feature_training.log`
- `robust_training.log`

---

## Conclusion

The **27% F1 score** with a 99.37% validation model clearly indicates **severe distribution shift**. Our feature-based approach:

- Uses **normalized, robust features** that handle code length variance
- Achieved **96.15% validation F1** with better generalization potential
- Provides **interpretable** insights into human vs machine code differences
- Creates multiple submission strategies to hedge against uncertainty

**Expected improvement**: From 27% F1 to hopefully **70-85% F1** with feature-based approach! 🤞
