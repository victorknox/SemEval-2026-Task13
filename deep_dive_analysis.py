#!/usr/bin/env python3
"""
Deep dive: What's REALLY different about the test set?
Let's look at actual code examples and patterns
"""

import pandas as pd
import numpy as np
import re

print("=" * 100)
print("DEEP DIVE: WHAT'S ACTUALLY IN THE TEST SET?")
print("=" * 100)

# Load data
train_df = pd.read_parquet('Task_A/train.parquet')
val_df = pd.read_parquet('Task_A/validation.parquet')
test_df = pd.read_parquet('Task_A/test.parquet')

print(f"\nDataset sizes: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

# Look at actual code examples
print("\n" + "=" * 100)
print("TEST SET CODE EXAMPLES (First 5 - FULL CODE)")
print("=" * 100)

for i in range(min(5, len(test_df))):
    code = test_df.iloc[i]['code']
    print(f"\n{'='*80}")
    print(f"ID: {test_df.iloc[i]['ID']}")
    print(f"Length: {len(code)} chars, {code.count(chr(10))+1} lines")
    print(f"{'='*80}")
    print(code)
    print(f"{'='*80}\n")

# Compare with validation examples
print("\n" + "=" * 100)
print("VALIDATION SET CODE EXAMPLES (Random 3 from EACH class)")
print("=" * 100)

# Sample from each class
val_human = val_df[val_df['label'] == 0].sample(n=3, random_state=42)
val_machine = val_df[val_df['label'] == 1].sample(n=3, random_state=42)

print("\n--- HUMAN-WRITTEN CODE ---")
for i, row in val_human.iterrows():
    print(f"\n[Human Example - {len(row['code'])} chars]")
    print(row['code'][:500])
    print("...\n")

print("\n--- MACHINE-GENERATED CODE ---")
for i, row in val_machine.iterrows():
    print(f"\n[Machine Example - {len(row['code'])} chars]")
    print(row['code'][:500])
    print("...\n")

# Analyze specific patterns
print("\n" + "=" * 100)
print("PATTERN ANALYSIS: What's different?")
print("=" * 100)

def analyze_patterns(df, name):
    print(f"\n{name}:")
    
    # Language detection
    def detect_language(code):
        if 'def ' in code or 'import ' in code or '__name__' in code:
            return 'Python'
        elif '#include' in code or 'int main' in code:
            return 'C/C++'
        elif 'public class' in code or 'public static void main' in code:
            return 'Java'
        else:
            return 'Unknown'
    
    langs = df['code'].apply(detect_language).value_counts()
    print(f"  Languages: {dict(langs)}")
    
    # Check for specific patterns
    has_readme = df['code'].str.contains('README|readme|Problem|problem', case=False, regex=True).sum()
    has_url = df['code'].str.contains('http://|https://|www\.', regex=True).sum()
    has_markdown = df['code'].str.contains(r'#+\s|\*\*|```', regex=True).sum()
    very_long = (df['code'].str.len() > 2000).sum()
    very_short = (df['code'].str.len() < 200).sum()
    
    print(f"  Contains README/Problem text: {has_readme} ({has_readme/len(df)*100:.1f}%)")
    print(f"  Contains URLs: {has_url} ({has_url/len(df)*100:.1f}%)")
    print(f"  Contains Markdown: {has_markdown} ({has_markdown/len(df)*100:.1f}%)")
    print(f"  Very long (>2000 chars): {very_long} ({very_long/len(df)*100:.1f}%)")
    print(f"  Very short (<200 chars): {very_short} ({very_short/len(df)*100:.1f}%)")

train_sample = train_df.sample(n=5000, random_state=42)
analyze_patterns(train_sample, "Training Set (sample)")
analyze_patterns(val_df.sample(n=5000, random_state=42), "Validation Set (sample)")
analyze_patterns(test_df, "Test Set (ALL)")

# Check if test set has mixed content (code + problem descriptions)
print("\n" + "=" * 100)
print("HYPOTHESIS: Test set might have MIXED content (code + problem descriptions)?")
print("=" * 100)

for i in range(min(10, len(test_df))):
    code = test_df.iloc[i]['code']
    has_problem = any(word in code.lower() for word in ['problem', 'input', 'output', 'example', 'description', 'note'])
    has_code = any(word in code for word in ['def ', 'class ', 'import', 'for ', 'while ', 'if ', '=='])
    
    print(f"ID {test_df.iloc[i]['ID']:6d}: HasProblem={has_problem:5} | HasCode={has_code:5} | Len={len(code):5}")

# Check training vs test language distribution more carefully
print("\n" + "=" * 100)
print("DETAILED LANGUAGE ANALYSIS")
print("=" * 100)

def detailed_language_check(code):
    python_score = sum([
        'def ' in code,
        'import ' in code,
        '__name__' in code,
        'print(' in code,
        '.append(' in code,
        'range(' in code
    ])
    
    cpp_score = sum([
        '#include' in code,
        'int main' in code,
        'std::' in code,
        'cout' in code,
        'cin' in code
    ])
    
    java_score = sum([
        'public class' in code,
        'public static void main' in code,
        'System.out' in code,
        '.println' in code
    ])
    
    if python_score > cpp_score and python_score > java_score:
        return 'Python'
    elif cpp_score > python_score and cpp_score > java_score:
        return 'C++'
    elif java_score > python_score and java_score > cpp_score:
        return 'Java'
    else:
        return 'Mixed/Other'

test_langs = test_df['code'].apply(detailed_language_check).value_counts()
print(f"\nTest set languages:")
for lang, count in test_langs.items():
    print(f"  {lang}: {count} ({count/len(test_df)*100:.1f}%)")

train_langs = train_sample['code'].apply(detailed_language_check).value_counts()
print(f"\nTraining set languages (sample):")
for lang, count in train_langs.items():
    print(f"  {lang}: {count} ({count/len(train_sample)*100:.1f}%)")

# Specific test case analysis
print("\n" + "=" * 100)
print("CHECKING IF TEST SET IS ACTUALLY PURE CODE")
print("=" * 100)

for i in [0, 1, 2]:  # First 3
    code = test_df.iloc[i]['code']
    lines = code.split('\n')
    
    print(f"\nID {test_df.iloc[i]['ID']} - Line by line analysis:")
    print(f"Total lines: {len(lines)}")
    
    code_lines = 0
    text_lines = 0
    blank_lines = 0
    
    for line in lines[:20]:  # First 20 lines
        stripped = line.strip()
        if not stripped:
            blank_lines += 1
        elif any(c in stripped for c in ['def ', 'class ', 'import', 'for ', 'if ', '==', '(', ')', '{', '}']):
            code_lines += 1
            print(f"  [CODE] {line[:80]}")
        else:
            text_lines += 1
            print(f"  [TEXT?] {line[:80]}")
    
    print(f"Summary (first 20): Code={code_lines}, Text={text_lines}, Blank={blank_lines}")
