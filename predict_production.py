#!/usr/bin/env python3
"""
PRODUCTION-READY PREDICTION SCRIPT
Generate predictions on test set using trained model
"""

import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)

def main():
    """Main prediction function"""
    
    print_section("GENERATING FINAL PREDICTIONS")
    
    # Configuration
    MODEL_PATH = "task_a_solution/models/codebert_full_500k"
    TEST_DATA = "Task_A/test.parquet"
    OUTPUT_FILE = "task_a_solution/results/final_submission_v2.csv"
    BATCH_SIZE = 32
    MAX_LENGTH = 384
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Please train the model first using train_production.py")
        sys.exit(1)
    
    # Load test data
    print_section("LOADING TEST DATA")
    try:
        print(f"Loading: {TEST_DATA}")
        test_df = pd.read_parquet(TEST_DATA)
        print(f"✓ Test samples: {len(test_df):,}")
        print(f"✓ Columns: {test_df.columns.tolist()}")
        
        if 'ID' not in test_df.columns or 'code' not in test_df.columns:
            print("ERROR: Required columns missing (ID, code)")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR loading test data: {e}")
        sys.exit(1)
    
    # Load model
    print_section("LOADING MODEL")
    try:
        print(f"Loading model from: {MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        print(f"✓ Model loaded successfully")
        print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Generate predictions
    print_section("GENERATING PREDICTIONS")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max length: {MAX_LENGTH}")
    
    all_predictions = []
    
    try:
        with torch.no_grad():
            for i in tqdm(range(0, len(test_df), BATCH_SIZE), desc="Predicting"):
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
        
        print(f"✓ Generated {len(all_predictions):,} predictions")
        
    except Exception as e:
        print(f"ERROR during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create submission
    print_section("CREATING SUBMISSION FILE")
    try:
        submission_df = pd.DataFrame({
            'ID': test_df['ID'].values,
            'label': all_predictions
        })
        
        # Validate
        assert len(submission_df) == len(test_df), "Prediction count mismatch"
        assert set(submission_df['label'].unique()).issubset({0, 1}), "Invalid labels"
        assert not submission_df.isnull().any().any(), "Null values in submission"
        
        # Save
        submission_df.to_csv(OUTPUT_FILE, index=False)
        print(f"✓ Submission saved to: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"ERROR creating submission: {e}")
        sys.exit(1)
    
    # Statistics
    print_section("SUBMISSION STATISTICS")
    print(f"Total predictions: {len(submission_df):,}")
    print(f"\nLabel distribution:")
    label_counts = submission_df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        label_name = "Human" if label == 0 else "Machine"
        percentage = (count / len(submission_df)) * 100
        print(f"  {label} ({label_name}): {count:4d} ({percentage:5.2f}%)")
    
    print(f"\nFirst 10 predictions:")
    print(submission_df.head(10).to_string(index=False))
    
    print_section("SUBMISSION READY")
    print(f"File: {OUTPUT_FILE}")
    print(f"Format: ID, label")
    print(f"Model: CodeBERT trained on 500K samples")
    print(f"\nUpload to Kaggle:")
    print(f"  kaggle competitions submit -c sem-eval-2026-task-13-subtask-a \\")
    print(f"    -f {OUTPUT_FILE} \\")
    print(f'    -m "CodeBERT trained on full 500K dataset"')
    print("=" * 100)
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nPrediction interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
