"""
Model 2: CodeBERT Fine-tuning
SemEval-2026 Task 13 - Task A: Binary Machine-Generated Code Detection
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_auc_score
)
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MODEL 2: CODEBERT FINE-TUNING")
print("="*80)

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✓ Using device: {device}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Load data
print("\n[1] Loading data...")
df = pd.read_parquet('task_A/task_a_trial.parquet')
print(f"✓ Loaded {len(df):,} samples")

# Load label mappings
with open('task_A/id_to_label.json', 'r') as f:
    id_to_label = json.load(f)

# Prepare data
print("\n[2] Preparing data...")
X_train, X_test, y_train, y_test = train_test_split(
    df['code'].values, df['label'].values,
    test_size=0.2, random_state=42, stratify=df['label']
)
print(f"✓ Train samples: {len(X_train):,}")
print(f"✓ Test samples: {len(X_test):,}")

# Initialize tokenizer and model
print("\n[3] Initializing CodeBERT...")
model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    problem_type="single_label_classification"
)
print(f"✓ Model: {model_name}")
print(f"✓ Model parameters: {model.num_parameters():,}")

# Create dataset class
class CodeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create datasets
print("\n[4] Creating datasets...")
train_dataset = CodeDataset(X_train, y_train, tokenizer)
test_dataset = CodeDataset(X_test, y_test, tokenizer)
print(f"✓ Train dataset size: {len(train_dataset)}")
print(f"✓ Test dataset size: {len(test_dataset)}")

# Define metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Training arguments
print("\n[5] Setting up training arguments...")
training_args = TrainingArguments(
    output_dir='task_a_solution/models/codebert',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='task_a_solution/models/codebert/logs',
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    fp16=True,  # Use mixed precision for faster training
    dataloader_num_workers=4,
    report_to="none"
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train model
print("\n[6] Training CodeBERT...")
print("=" * 80)
start_time = time.time()
train_result = trainer.train()
training_time = time.time() - start_time
print(f"\n✓ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

# Save model
print("\n[7] Saving model...")
trainer.save_model('task_a_solution/models/codebert_final')
tokenizer.save_pretrained('task_a_solution/models/codebert_final')
print("✓ Saved: task_a_solution/models/codebert_final")

# Evaluate on test set
print("\n[8] Evaluating on test set...")
predictions = trainer.predict(test_dataset)
y_pred = predictions.predictions.argmax(-1)
y_pred_proba = torch.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average='macro'
)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*80)
print("CODEBERT MODEL RESULTS")
print("="*80)
print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\n" + "-"*80)
print("Classification Report:")
print("-"*80)
target_names = [id_to_label[str(i)] for i in sorted(np.unique(y_test))]
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Save results
results = {
    'model_name': 'CodeBERT',
    'model_checkpoint': model_name,
    'timestamp': datetime.now().isoformat(),
    'metrics': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc)
    },
    'training_time': float(training_time),
    'train_samples': int(len(X_train)),
    'test_samples': int(len(X_test)),
    'confusion_matrix': cm.tolist(),
    'training_args': {
        'epochs': training_args.num_train_epochs,
        'batch_size': training_args.per_device_train_batch_size,
        'learning_rate': training_args.learning_rate
    }
}

with open('task_a_solution/results/codebert_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n✓ Saved: task_a_solution/results/codebert_results.json")

# Save predictions
pred_df = pd.DataFrame({
    'true_label': y_test,
    'predicted_label': y_pred,
    'predicted_proba': y_pred_proba
})
pred_df.to_csv('task_a_solution/results/codebert_predictions.csv', index=False)
print("✓ Saved: task_a_solution/results/codebert_predictions.csv")

print("\n" + "="*80)
print("CODEBERT TRAINING COMPLETE!")
print("="*80)
print(f"\nMacro F1-Score: {f1:.4f}")
print(f"Improvement over baseline: {f1 - 0.8774:.4f}")
