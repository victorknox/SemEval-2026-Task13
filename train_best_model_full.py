#!/usr/bin/env python3
"""
Train the BEST model on FULL 500K training dataset
Optimized for H100 GPU with <1 hour training time
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
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("BEST MODEL - FULL DATASET TRAINING")
print("SemEval-2026 Task 13 Subtask A")
print("=" * 100)

# Configuration - Using CodeBERT but with FULL data and optimized settings
MODEL_NAME = "microsoft/codebert-base"  # 125M params - proven, fast, fits well
OUTPUT_DIR = "task_a_solution/models/codebert_full_500k"
RESULTS_FILE = "task_a_solution/results/codebert_full_500k_results.json"
TRAIN_FILE = "Task_A/train.parquet"
VAL_FILE = "Task_A/validation.parquet"

# Optimized settings for <1 hour training on 500K samples
MAX_LENGTH = 384  # Shorter for speed (most code fits in 384 tokens)
BATCH_SIZE = 32  # Larger batch for H100
GRADIENT_ACCUMULATION = 2  # Effective batch = 64
EPOCHS = 1  # Single epoch on 500K is enough
LEARNING_RATE = 3e-5  # Slightly higher for single epoch
VAL_SAMPLES = 20000  # Use 20K validation for faster eval

# Device check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

start_time = time.time()

# Load data
print(f"\nLoading training data from {TRAIN_FILE}...")
train_df = pd.read_parquet(TRAIN_FILE)
print(f"Training samples: {len(train_df):,}")

print(f"\nLoading validation data from {VAL_FILE}...")
val_df = pd.read_parquet(VAL_FILE)
print(f"Full validation: {len(val_df):,}")

# Use subset of validation for faster evaluation
val_df = val_df.sample(n=VAL_SAMPLES, random_state=42)
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

# Calculate steps
total_steps = len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION)
eval_steps = max(total_steps // 5, 500)  # Eval 5 times per epoch
save_steps = eval_steps

print("\nTraining configuration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Training samples: {len(train_dataset):,}")
print(f"  Validation samples: {len(val_dataset):,}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size per device: {BATCH_SIZE}")
print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION}")
print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Max sequence length: {MAX_LENGTH}")
print(f"  Total steps: {total_steps:,}")
print(f"  Eval every: {eval_steps:,} steps")

# Training arguments - optimized for speed
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_strategy="steps",
    save_steps=save_steps,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
    dataloader_num_workers=4,
    report_to="none",
    save_total_limit=2,
    gradient_checkpointing=False,  # Faster without checkpointing on H100
)

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
print("STARTING TRAINING ON 500,000 SAMPLES")
print("=" * 100)

train_result = trainer.train()

training_time = time.time() - start_time

print("\n" + "=" * 100)
print("TRAINING COMPLETE")
print("=" * 100)
print(f"Training time: {training_time/60:.2f} minutes")

# Save model
print(f"\nSaving model to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✓ Model saved")

# Evaluate on validation set
print("\nEvaluating on validation set...")
predictions_output = trainer.predict(val_dataset)
predictions = np.argmax(predictions_output.predictions, axis=1)
labels = val_dataset['label']

# Calculate metrics
acc = accuracy_score(labels, predictions)
f1 = f1_score(labels, predictions, average='macro')
f1_weighted = f1_score(labels, predictions, average='weighted')

print("\n" + "=" * 100)
print("FINAL RESULTS")
print("=" * 100)
print(f"Training time: {training_time/60:.2f} minutes")
print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"F1-Score (Macro):  {f1:.4f} ({f1*100:.2f}%)")
print(f"F1-Score (Weighted):  {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(labels, predictions, target_names=['Human', 'Machine']))

# Error analysis
errors = np.where(predictions != labels)[0]
print(f"\nErrors: {len(errors)} / {len(labels)} ({len(errors)/len(labels)*100:.2f}%)")

# Save results
results = {
    'model': MODEL_NAME,
    'train_samples': len(train_dataset),
    'val_samples': len(val_dataset),
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE * GRADIENT_ACCUMULATION,
    'max_length': MAX_LENGTH,
    'accuracy': float(acc),
    'f1_macro': float(f1),
    'f1_weighted': float(f1_weighted),
    'training_time_minutes': training_time / 60,
    'errors': int(len(errors)),
}

with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to {RESULTS_FILE}")
print("=" * 100)
print(f"Total time: {training_time/60:.2f} minutes")
print("=" * 100)
