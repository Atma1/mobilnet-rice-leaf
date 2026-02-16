# mobilnet-rice-leaf

Multi-Scenario MobileNetV3 Transfer Learning for Rice Leaf Disease Classification

## Overview

This project implements a comprehensive transfer learning experiment using **MobileNetV3Large** for rice leaf disease classification. It trains and evaluates **18 different scenarios** combining various training configurations.

## Files

- **`mobilenetv3.ipynb`** - Jupyter notebook with interactive training and visualization
- **`mobilenetv3.py`** - Python script version (non-interactive, auto-saves all plots)

## Dataset

- **4 selected classes** from rice leaf diseases:
  - bacterial_leaf_blight
  - brown_spot
  - leaf_blast
  - rice_leaf_healthy

## Training Scenarios (18 total)

Combinations of:
- **3 Data Splits**: 90:10, 80:20, 70:30 (train:validation)
- **2 Optimizers**: Adam, SGD
- **3 Learning Rates**: 0.1, 0.01, 0.001

## Output Metrics

For each scenario, the following metrics are calculated and stored:
- Training Accuracy & Loss
- Validation Accuracy & Loss
- **Precision** (weighted average)
- **Recall** (weighted average)
- **F1 Score** (weighted average)

## Usage

### Jupyter Notebook
```bash
jupyter notebook mobilenetv3.ipynb
```

### Python Script
```bash
python mobilenetv3.py
```

## Output Files

The training process generates:
- `scenario_results.csv` - All metrics for 18 scenarios
- `training_history.json` - Detailed epoch-by-epoch history
- `top_10_scenarios.png` - Bar chart of best scenarios
- `top_3_training_curves.png` - Training curves visualization
- `best_scenario_confusion_matrix.png` - Confusion matrix
- Model files in `models/` directory