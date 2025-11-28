#!/usr/bin/env python3
"""
Generate final submission for SemEval-2026 Task 13 Subtask A
Using the best model: CodeBERT (95.95% F1-score)
"""

import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SemEval-2026 Task 13 Subtask A - Final Submission Generator")
print("=" * 80)

# Configuration
MODEL_PATH = "task_a_solution/models/codebert_final"
TEST_DATA = "Task_A/test.parquet"
OUTPUT_FILE = "task_a_solution/results/final_submission.csv"
BATCH_SIZE = 16
MAX_LENGTH = 512

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load test data
print(f"\nLoading test data from {TEST_DATA}...")
test_df = pd.read_parquet(TEST_DATA)
print(f"Test samples: {len(test_df)}")
print(f"Columns: {test_df.columns.tolist()}")

# Load model and tokenizer
print(f"\nLoading CodeBERT model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()
print("✓ Model loaded successfully")

# Generate predictions
print("\nGenerating predictions...")
all_predictions = []

with torch.no_grad():
    for i in tqdm(range(0, len(test_df), BATCH_SIZE)):
        batch = test_df.iloc[i:i+BATCH_SIZE]
        texts = batch['code'].tolist()
        
        # Tokenize
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        ).to(device)
        
        # Predict
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        all_predictions.extend(predictions)

# Create submission dataframe
print("\nCreating submission file...")
submission_df = pd.DataFrame({
    'ID': test_df['ID'].values,
    'label': all_predictions
})

# Save submission
submission_df.to_csv(OUTPUT_FILE, index=False)
print(f"✓ Submission saved to: {OUTPUT_FILE}")

# Display statistics
print("\n" + "=" * 80)
print("SUBMISSION STATISTICS")
print("=" * 80)
print(f"Total predictions: {len(submission_df)}")
print(f"\nLabel distribution:")
label_counts = submission_df['label'].value_counts().sort_index()
for label, count in label_counts.items():
    label_name = "Human" if label == 0 else "Machine"
    percentage = (count / len(submission_df)) * 100
    print(f"  {label} ({label_name}): {count:4d} ({percentage:5.2f}%)")

print("\nFirst 10 predictions:")
print(submission_df.head(10).to_string(index=False))

print("\n" + "=" * 80)
print("SUBMISSION FILE READY FOR KAGGLE!")
print("=" * 80)
print(f"\nFile: {OUTPUT_FILE}")
print(f"Format: ID, label")
print(f"Model: CodeBERT (95.95% F1-score on validation)")
print("\nUpload this file to the Kaggle competition to submit your predictions!")
print("=" * 80)
