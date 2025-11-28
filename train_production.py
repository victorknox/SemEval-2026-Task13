#!/usr/bin/env python3
"""
PRODUCTION-READY TRAINING SCRIPT
Train CodeBERT on full 500K dataset with proper error handling
Optimized for H100, <1 hour training time
"""

import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json
import numpy as np
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
set_seed(42)

def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)

def check_gpu():
    """Check GPU availability and specs"""
    if not torch.cuda.is_available():
        print("ERROR: No GPU available!")
        sys.exit(1)
    
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"Device: {device}")
    print(f"GPU: {gpu_name}")
    print(f"GPU Memory: {gpu_memory:.2f} GB")
    
    return device

def load_data(train_file, val_file, val_samples=20000):
    """Load and validate data"""
    print_section("LOADING DATA")
    
    try:
        print(f"Loading training data: {train_file}")
        train_df = pd.read_parquet(train_file)
        print(f"✓ Training samples: {len(train_df):,}")
        
        print(f"\nLoading validation data: {val_file}")
        val_df = pd.read_parquet(val_file)
        print(f"✓ Full validation: {len(val_df):,}")
        
        # Sample validation for faster eval
        if len(val_df) > val_samples:
            val_df = val_df.sample(n=val_samples, random_state=42)
            print(f"✓ Using {len(val_df):,} validation samples")
        
        # Validate columns
        required_cols = ['code', 'label']
        for col in required_cols:
            if col not in train_df.columns:
                raise ValueError(f"Missing column: {col}")
        
        # Check for nulls
        if train_df[required_cols].isnull().any().any():
            print("WARNING: Null values found, dropping...")
            train_df = train_df.dropna(subset=required_cols)
        
        if val_df[required_cols].isnull().any().any():
            val_df = val_df.dropna(subset=required_cols)
        
        # Show distributions
        print(f"\nTraining label distribution:")
        print(train_df['label'].value_counts().sort_index())
        print(f"\nValidation label distribution:")
        print(val_df['label'].value_counts().sort_index())
        
        return train_df, val_df
        
    except Exception as e:
        print(f"ERROR loading data: {e}")
        sys.exit(1)

def prepare_datasets(train_df, val_df, tokenizer, max_length=384):
    """Tokenize and prepare datasets"""
    print_section("PREPARING DATASETS")
    
    try:
        print("Creating HuggingFace datasets...")
        train_dataset = Dataset.from_pandas(train_df[['code', 'label']])
        val_dataset = Dataset.from_pandas(val_df[['code', 'label']])
        print(f"✓ Train dataset: {len(train_dataset):,} examples")
        print(f"✓ Val dataset: {len(val_dataset):,} examples")
        
        print(f"\nTokenizing with max_length={max_length}...")
        
        def tokenize_function(examples):
            return tokenizer(
                examples['code'],
                padding=False,
                truncation=True,
                max_length=max_length
            )
        
        # Tokenize with progress
        print("Tokenizing training data...")
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            remove_columns=['code'],
            desc="Tokenizing train"
        )
        
        print("Tokenizing validation data...")
        val_dataset = val_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            remove_columns=['code'],
            desc="Tokenizing val"
        )
        
        print("✓ Tokenization complete")
        
        return train_dataset, val_dataset
        
    except Exception as e:
        print(f"ERROR preparing datasets: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    
    return {
        'accuracy': acc,
        'f1': f1,
    }

def main():
    """Main training function"""
    
    print_section("BEST MODEL TRAINING - FULL 500K DATASET")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    MODEL_NAME = "microsoft/codebert-base"
    OUTPUT_DIR = "task_a_solution/models/codebert_full_500k"
    RESULTS_FILE = "task_a_solution/results/codebert_full_500k_results.json"
    TRAIN_FILE = "Task_A/train.parquet"
    VAL_FILE = "Task_A/validation.parquet"
    
    # Hyperparameters
    MAX_LENGTH = 384
    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION = 2
    EPOCHS = 1
    LEARNING_RATE = 3e-5
    VAL_SAMPLES = 20000
    
    start_time = time.time()
    
    # Check GPU
    device = check_gpu()
    
    # Load data
    train_df, val_df = load_data(TRAIN_FILE, VAL_FILE, VAL_SAMPLES)
    
    # Load model
    print_section("LOADING MODEL")
    print(f"Model: {MODEL_NAME}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            problem_type="single_label_classification"
        )
        print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(train_df, val_df, tokenizer, MAX_LENGTH)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training setup
    print_section("TRAINING CONFIGURATION")
    
    total_steps = len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION)
    eval_steps = max(total_steps // 5, 500)
    save_steps = eval_steps
    
    print(f"Model: {MODEL_NAME}")
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size per device: {BATCH_SIZE}")
    print(f"Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Max sequence length: {MAX_LENGTH}")
    print(f"Total steps: {total_steps:,}")
    print(f"Eval every: {eval_steps:,} steps")
    print(f"Output dir: {OUTPUT_DIR}")
    
    # Training arguments
    try:
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
            gradient_checkpointing=False,
        )
    except Exception as e:
        print(f"ERROR creating training arguments: {e}")
        sys.exit(1)
    
    # Initialize trainer
    print("\nInitializing Trainer...")
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        print("✓ Trainer initialized")
    except Exception as e:
        print(f"ERROR initializing trainer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Train
    print_section("TRAINING")
    print(f"Training on {len(train_dataset):,} samples...")
    print(f"This will take approximately 40-50 minutes...")
    
    try:
        train_result = trainer.train()
        training_time = time.time() - start_time
        print(f"\n✓ Training complete in {training_time/60:.2f} minutes")
    except Exception as e:
        print(f"ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save model
    print_section("SAVING MODEL")
    try:
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"✓ Model saved to: {OUTPUT_DIR}")
    except Exception as e:
        print(f"ERROR saving model: {e}")
        sys.exit(1)
    
    # Evaluate
    print_section("EVALUATION")
    try:
        print("Generating predictions on validation set...")
        predictions_output = trainer.predict(val_dataset)
        predictions = np.argmax(predictions_output.predictions, axis=1)
        labels = val_dataset['label']
        
        # Calculate metrics
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        
        print(f"\nAccuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"F1-Score (Macro):  {f1:.4f} ({f1*100:.2f}%)")
        print(f"F1-Score (Weighted):  {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")
        
        print("\nClassification Report:")
        print(classification_report(labels, predictions, target_names=['Human', 'Machine']))
        
        # Error analysis
        errors = np.where(predictions != labels)[0]
        print(f"\nErrors: {len(errors):,} / {len(labels):,} ({len(errors)/len(labels)*100:.2f}%)")
        
    except Exception as e:
        print(f"ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save results
    print_section("SAVING RESULTS")
    try:
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
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved to: {RESULTS_FILE}")
        
    except Exception as e:
        print(f"ERROR saving results: {e}")
    
    # Final summary
    print_section("TRAINING COMPLETE")
    print(f"Total time: {training_time/60:.2f} minutes")
    print(f"Final F1 Score: {f1*100:.2f}%")
    print(f"Model saved: {OUTPUT_DIR}")
    print(f"Results saved: {RESULTS_FILE}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
