#!/usr/bin/env python3
"""
CRITICAL DISCOVERY: Test set has CODE + PROBLEM DESCRIPTIONS mixed!
We need to EXTRACT just the code part!
"""

import pandas as pd
import re

test_df = pd.read_parquet('Task_A/test.parquet')

print("=" * 100)
print("ANALYZING TEST SET STRUCTURE")
print("=" * 100)

# Check how many have problem descriptions
has_problem_marker = 0
has_readme_path = 0
has_problem_word = 0
has_input_output = 0

for idx, row in test_df.iterrows():
    code = row['code']
    
    if '/readme.md' in code.lower() or '/problem' in code.lower():
        has_readme_path += 1
    
    if 'Problem A' in code or 'Problem B' in code or 'Problem C' in code:
        has_problem_marker += 1
    
    if '-----Input-----' in code or '-----Output-----' in code:
        has_input_output += 1
    
    if 'problem' in code.lower():
        has_problem_word += 1

print(f"\nTest set contamination:")
print(f"  Has /readme.md or /problem path: {has_readme_path}/{len(test_df)} ({has_readme_path/len(test_df)*100:.1f}%)")
print(f"  Has 'Problem A/B/C' marker: {has_problem_marker}/{len(test_df)} ({has_problem_marker/len(test_df)*100:.1f}%)")
print(f"  Has '-----Input-----/Output-----': {has_input_output}/{len(test_df)} ({has_input_output/len(test_df)*100:.1f}%)")
print(f"  Contains word 'problem': {has_problem_word}/{len(test_df)} ({has_problem_word/len(test_df)*100:.1f}%)")

# Let's try to extract ONLY the code part
print("\n" + "=" * 100)
print("ATTEMPTING CODE EXTRACTION")
print("=" * 100)

def extract_code_only(text):
    """Try to extract only the code part, removing problem descriptions"""
    
    # Split by common markers
    # Usually code comes before the /xxx/readme.md marker
    if '/readme.md' in text or '/problem' in text.lower():
        # Find the first occurrence of a path marker
        match = re.search(r'/\d+\..*?readme\.md|/\d+\..*?problem', text, re.IGNORECASE)
        if match:
            # Take everything before this marker
            code_part = text[:match.start()].strip()
            if code_part:
                return code_part
    
    # If there's -----Input----- or -----Output-----, code usually comes before
    if '-----Input-----' in text or '-----Output-----' in text:
        parts = re.split(r'-----Input-----|-----Output-----', text)
        if parts and parts[0].strip():
            return parts[0].strip()
    
    # If there's "Problem A/B/C", code might be before or after
    # Let's take the part before
    if 'Problem A' in text or 'Problem B' in text or 'Problem C' in text:
        match = re.search(r'Problem [ABC]', text)
        if match:
            code_part = text[:match.start()].strip()
            if len(code_part) > 50:  # Only if substantial
                return code_part
    
    # Otherwise return as-is
    return text

# Test extraction on first few examples
print("\nTesting extraction on first 5 samples:\n")

for i in range(5):
    original = test_df.iloc[i]['code']
    extracted = extract_code_only(original)
    
    print(f"ID: {test_df.iloc[i]['ID']}")
    print(f"Original length: {len(original)} chars")
    print(f"Extracted length: {len(extracted)} chars")
    print(f"Reduction: {(1 - len(extracted)/len(original))*100:.1f}%")
    print(f"\nExtracted code:")
    print(extracted)
    print("\n" + "-" * 80 + "\n")

# Create cleaned test set
print("=" * 100)
print("CREATING CLEANED TEST SET")
print("=" * 100)

test_df['code_cleaned'] = test_df['code'].apply(extract_code_only)

# Stats
original_lens = test_df['code'].str.len()
cleaned_lens = test_df['code_cleaned'].str.len()

print(f"\nOriginal avg length: {original_lens.mean():.1f} chars")
print(f"Cleaned avg length: {cleaned_lens.mean():.1f} chars")
print(f"Average reduction: {(1 - cleaned_lens.mean()/original_lens.mean())*100:.1f}%")

# Save cleaned version
test_cleaned = test_df[['ID', 'code_cleaned']].copy()
test_cleaned.columns = ['ID', 'code']
test_cleaned.to_parquet('Task_A/test_cleaned.parquet', index=False)

print(f"\n✓ Saved: Task_A/test_cleaned.parquet")
print(f"  Now we can re-run predictions on CLEAN code only!")

# Show distribution of reductions
huge_reduction = (cleaned_lens < original_lens * 0.5).sum()
print(f"\nSamples with >50% reduction: {huge_reduction} ({huge_reduction/len(test_df)*100:.1f}%)")
