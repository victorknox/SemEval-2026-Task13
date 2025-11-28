#!/usr/bin/env python3
"""
RULE-BASED APPROACH: Detect obvious machine-generated patterns
"""

import pandas as pd
import re

def is_machine_generated(code):
    """Detect obvious machine generation markers"""
    score = 0
    
    # Strong signals for MACHINE
    if 'endoftext' in code.lower() or 'end of text' in code.lower():
        score += 10  # Very strong signal
    
    if '```python' in code or '```java' in code or '```cpp' in code or '```c++' in code:
        score += 8  # Code block markers
    
    if 'python\n' in code[:50] or 'java\n' in code[:50] or 'cpp\n' in code[:50]:
        score += 5  # Language marker at start
    
    if '__name__ == "__main__"' in code or '__name__ == \'__main__\'' in code:
        score += 3
    
    if 'raw_input' in code:
        score += 3  # Outdated Python 2
    
    # Check for markdown-style comments
    if re.search(r'```\w+', code):
        score += 5
    
    # Check for documentation patterns
    if '"""' in code and code.count('"""') >= 4:  # Multiple docstrings
        score += 2
    
    # Strong signals for HUMAN
    if 'class solution' in code.lower() or 'class Solution' in code:
        score -= 5  # Competitive programming pattern
    
    if '__starting_point' in code:
        score -= 5  # Another competitive programming marker
    
    # Very simple/short code is often human
    if len(code) < 150 and 'def' not in code:
        score -= 2
    
    # Predict based on score
    return 1 if score > 0 else 0

print("=" * 100)
print("RULE-BASED DETECTION")
print("=" * 100)

# Load data
test_df = pd.read_parquet('Task_A/test.parquet')
val_df = pd.read_parquet('Task_A/validation.parquet')

# Test on validation first
print("\nTesting on validation...")
val_sample = val_df.sample(n=5000, random_state=42)
val_preds = [is_machine_generated(code) for code in val_sample['code']]

from sklearn.metrics import classification_report, f1_score
print("\nValidation results:")
print(classification_report(val_sample['label'], val_preds, target_names=['Human', 'Machine']))
print(f"F1-Score: {f1_score(val_sample['label'], val_preds):.4f}")

# Apply to test
print("\nApplying to test set...")
test_preds = [is_machine_generated(code) for code in test_df['code']]

# Check some examples
print("\n" + "=" * 100)
print("EXAMPLES OF DETECTED PATTERNS")
print("=" * 100)

for i in range(min(10, len(test_df))):
    code = test_df.iloc[i]['code']
    pred = test_preds[i]
    pred_label = "MACHINE" if pred == 1 else "HUMAN"
    
    # Show why
    reasons = []
    if 'endoftext' in code.lower():
        reasons.append("has 'endoftext'")
    if '```' in code:
        reasons.append("has code blocks")
    if '__name__ == "__main__"' in code:
        reasons.append("has __main__ pattern")
    if 'class solution' in code.lower():
        reasons.append("has 'class solution'")
    if '__starting_point' in code:
        reasons.append("has '__starting_point'")
    
    if reasons:
        print(f"\nID {test_df.iloc[i]['ID']}: Predicted {pred_label}")
        print(f"  Reasons: {', '.join(reasons)}")
        print(f"  First 200 chars: {code[:200]}")

submission = pd.DataFrame({
    'ID': test_df['ID'],
    'label': test_preds
})

submission.to_csv('task_a_solution/results/rule_based_submission.csv', index=False)

human_pct = (submission['label'] == 0).sum() / len(submission) * 100
print(f"\n✓ Saved: rule_based_submission.csv")
print(f"  Distribution: {human_pct:.1f}% Human, {100-human_pct:.1f}% Machine")
