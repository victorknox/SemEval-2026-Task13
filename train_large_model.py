#!/usr/bin/env python3
"""
Train a large, generalizable model on the FULL training dataset
Using DeepSeek-Coder-1.3B or similar large model optimized for code
"""

import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("LARGE MODEL TRAINING - SemEval-2026 Task A")
print("=" * 100)

# Configuration - Try a larger, better model
# MODEL_NAME = "microsoft/codebert-base"  # 125M params - already tried, poor generalization
MODEL_NAME = "Salesforce/codet5p-770m"  # 770M params - Better for code understanding
# Other options:
# "Salesforce/codet5-large" # 770M params
# "bigcode/starcoderbase-1b" # 1B params - might be too large
# "deepseek-ai/deepseek-coder-1.3b-base" # 1.3B params

OUTPUT_DIR = "task_a_solution/models/large_model_full_data"
RESULTS_FILE = "task_a_solution/results/large_model_results.json"
TRAIN_FILE = "Task_A/train.parquet"
VAL_FILE = "Task_A/validation.parquet"

# Training settings
MAX_SAMPLES = 100000  # Use 100K samples for faster iteration (20% of data)
MAX_LENGTH = 512
BATCH_SIZE = 4  # Smaller for larger model
GRADIENT_ACCUMULATION = 8  # Effective batch size = 4 * 8 = 32
EPOCHS = 3  # A few epochs
LEARNING_RATE = 2e-5

# Device check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Load data
print(f"\nLoading training data from {TRAIN_FILE}...")
train_df = pd.read_parquet(TRAIN_FILE)
print(f"Full training samples: {len(train_df):,}")

if MAX_SAMPLES:
    train_df = train_df.sample(n=MAX_SAMPLES, random_state=42)
    print(f"Using {len(train_df):,} samples for faster training")

print(f"\nLoading validation data from {VAL_FILE}...")
val_df = pd.read_parquet(VAL_FILE)
print(f"Full validation samples: {len(val_df):,}")

# Limit validation for faster eval
val_df = val_df.sample(n=min(10000, len(val_df)), random_state=42)
print(f"Using {len(val_df):,} validation samples")

# Show distribution
print(f"\nTraining label distribution:")
print(train_df['label'].value_counts())
print(f"\nValidation label distribution:")
print(val_df['label'].value_counts())

# Load model and tokenizer
print(f"\nLoading model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    problem_type="single_label_classification"
)
print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

# Prepare datasets
def tokenize_function(examples):
    return tokenizer(
        examples['code'],
        padding=False,
        truncation=True,
        max_length=MAX_LENGTH
    )

print("\nTokenizing datasets...")
train_dataset = Dataset.from_pandas(train_df[['code', 'label']])
val_dataset = Dataset.from_pandas(val_df[['code', 'label']])

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['code'])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=['code'])

print(f"✓ Tokenization complete")

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    
    return {
        'accuracy': acc,
        'f1': f1,
    }

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="steps",
    eval_steps=2000,
    save_strategy="steps",
    save_steps=2000,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    warmup_steps=500,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
    report_to="none",
    save_total_limit=2,
)

print("\nTraining configuration:")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION})")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Max length: {MAX_LENGTH}")
print(f"  Training samples: {len(train_dataset):,}")
print(f"  Steps per epoch: {len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION):,}")

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train
print("\n" + "=" * 100)
print("STARTING TRAINING")
print("=" * 100)

train_result = trainer.train()

print("\n" + "=" * 100)
print("TRAINING COMPLETE")
print("=" * 100)

# Save model
print(f"\nSaving model to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✓ Model saved")

# Evaluate on full validation set
print("\nEvaluating on validation set...")
predictions_output = trainer.predict(val_dataset)
predictions = np.argmax(predictions_output.predictions, axis=1)
labels = val_dataset['label']

# Calculate metrics
acc = accuracy_score(labels, predictions)
f1 = f1_score(labels, predictions, average='macro')
f1_weighted = f1_score(labels, predictions, average='weighted')

# Try ROC-AUC
try:
    probs = torch.softmax(torch.tensor(predictions_output.predictions), dim=1)[:, 1].numpy()
    roc_auc = roc_auc_score(labels, probs)
except:
    roc_auc = 0.0

print("\n" + "=" * 100)
print("FINAL RESULTS")
print("=" * 100)
print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"F1-Score (Macro):  {f1:.4f} ({f1*100:.2f}%)")
print(f"F1-Score (Weighted):  {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")
print(f"ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(labels, predictions, target_names=['Human', 'Machine']))

# Save results
results = {
    'model': MODEL_NAME,
    'train_samples': len(train_dataset),
    'val_samples': len(val_dataset),
    'epochs': EPOCHS,
    'accuracy': float(acc),
    'f1_macro': float(f1),
    'f1_weighted': float(f1_weighted),
    'roc_auc': float(roc_auc),
    'training_time': train_result.metrics['train_runtime'],
}

with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to {RESULTS_FILE}")
print("=" * 100)
