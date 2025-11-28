# Task A Solution - Quick Start Guide

## 🎯 Best Model Performance

**CodeBERT achieved 95.95% Macro F1-Score** ⭐

## 📁 Directory Structure

```
task_a_solution/
├── code/               # All Python scripts (reproducible)
│   ├── 01_eda.py
│   ├── 02_baseline_model.py
│   ├── 03_distilbert_model.py
│   ├── 04_codebert_model.py
│   └── 05_model_comparison.py
├── models/             # Trained models
│   ├── baseline_model.pkl
│   ├── distilbert_final/
│   └── codebert_final/  ← BEST MODEL
├── results/            # JSON results & predictions
│   ├── comprehensive_summary.json
│   ├── model_comparison.csv
│   └── *_predictions.csv
├── plots/              # 16 visualizations
└── REPORT.md          # Comprehensive report
```

## 🚀 Quick Start

### Run Everything
```bash
cd /root/SemEval-2026-Task13
source .venv/bin/activate

# Run entire pipeline
python task_a_solution/code/01_eda.py
python task_a_solution/code/02_baseline_model.py
python task_a_solution/code/03_distilbert_model.py
python task_a_solution/code/04_codebert_model.py
python task_a_solution/code/05_model_comparison.py
```

### Use Best Model (CodeBERT)
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
tokenizer = AutoTokenizer.from_pretrained('task_a_solution/models/codebert_final')
model = AutoModelForSequenceClassification.from_pretrained('task_a_solution/models/codebert_final')

# Predict
code = "def hello(): print('world')"
inputs = tokenizer(code, return_tensors='pt', truncation=True, max_length=512)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=1).item()
print(f"Prediction: {'Machine' if prediction == 1 else 'Human'}")
```

## 📊 Results Summary

| Model | F1-Score | ROC-AUC | Training Time |
|-------|----------|---------|---------------|
| Baseline (TF-IDF + LR) | 87.74% | 94.94% | 9s |
| DistilBERT | 87.99% | 95.53% | 80s |
| **CodeBERT** ⭐ | **95.95%** | **99.24%** | 102s |

## 📈 Key Visualizations

1. `plots/model_comparison_f1.png` - F1-Score comparison
2. `plots/model_comparison_all_metrics.png` - All metrics
3. `plots/model_comparison_confusion_matrices.png` - Confusion matrices
4. `plots/model_comparison_roc_curves.png` - ROC curves
5. `plots/model_error_analysis.png` - Error analysis

## 📝 Full Report

See `REPORT.md` for comprehensive documentation including:
- Task selection rationale
- Exploratory data analysis
- Methodology and model details
- Results and visualizations
- Key findings and conclusions

## ✅ Completed Tasks

- [x] Task selection (chose Task A - easiest)
- [x] Exploratory Data Analysis (5 plots)
- [x] Baseline model (87.74% F1)
- [x] DistilBERT model (87.99% F1)
- [x] CodeBERT model (95.95% F1) ⭐
- [x] Model comparison (7 plots)
- [x] Comprehensive report
- [x] Ready for submission

**Total Time:** ~3 hours
**Best F1-Score:** 95.95%
**Status:** ✅ Complete
