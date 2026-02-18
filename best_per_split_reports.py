#!/usr/bin/env python3
"""
Best Per Split Reports (Val Acc + F1) - Standalone Script

This script generates classification reports and confusion matrices for the best 
performing scenario per split ratio based on validation accuracy and F1 score.

It corresponds to section 7B from the mobilenetv3.ipynb notebook.

USAGE:
    python best_per_split_reports.py

PREREQUISITES:
    Before running this script, you must have:
    1. Completed training using mobilenetv3.py or mobilenetv3.ipynb
    2. Generated scenario_results.csv in the OUTPUT_DIR
    3. Generated training_history.json in the OUTPUT_DIR
    4. Saved model files in the MODELS_DIR

INPUTS:
    - scenario_results.csv: CSV file containing results from all training scenarios
    - training_history.json: JSON file with detailed training history
    - model_scenario_<id>_<split>_<optimizer>_lr<lr>.keras: Trained model files

OUTPUTS:
    For each best split scenario, generates:
    - classification_report_split_<ratio>_scenario_<id>.csv
    - confusion_matrix_split_<ratio>_scenario_<id>.png

PROCESS:
    1. Load scenario results from CSV
    2. Rank scenarios by split ratio, validation accuracy, and F1 score
    3. Select best performing scenario per split ratio
    4. For each best scenario:
       - Load the trained model
       - Generate predictions on validation data
       - Create classification report and save as CSV
       - Create confusion matrix visualization and save as PNG
"""

import os
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as preprocess_mobilenet

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Define AUTOTUNE for performance optimization
AUTOTUNE = tf.data.AUTOTUNE

# Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')


# ------------------ CONFIGURATION ------------------

class Config:
    BASE_INPUT = '/mgpfs/home/asusanto/_scratch/mobilnet-rice-leaf/dataset/Rice_Leaf_AUG/Rice_Leaf_AUG'
    WORK_DIR = '/mgpfs/home/asusanto/_scratch/mobilnet-rice-leaf/work'
    OUTPUT_DIR = '/mgpfs/home/asusanto/_scratch/mobilnet-rice-leaf/work/results'
    MODELS_DIR = '/mgpfs/home/asusanto/_scratch/mobilnet-rice-leaf/work/models'
    DATASET_DIR = '/mgpfs/home/asusanto/_scratch/mobilnet-rice-leaf/work/dataset'
    
    # Target classes for filtering (4 out of 10 classes)
    TARGET_CLASSES = ['bacterial_leaf_blight', 'brown_spot', 'leaf_blast', 'healthy_rice_leaf']
    
    BATCH_SIZE = 16
    IMG_SIZE_MOBILE = 224
    
    # Model architecture
    DROPOUT_RATE = 0.5
    DENSE_UNITS = 256
    NUM_CLASSES = 4  # Updated for 4 classes
    
    SEED = 42


config = Config()

print("Configuration loaded:")
print(f"  - Target classes: {config.TARGET_CLASSES}")
print(f"  - Number of classes: {config.NUM_CLASSES}")
print(f"  - Image size: {config.IMG_SIZE_MOBILE}x{config.IMG_SIZE_MOBILE}")
print(f"  - Batch size: {config.BATCH_SIZE}")
print(f"  - Output directory: {config.OUTPUT_DIR}")


# ------------------ HELPER FUNCTIONS ------------------

def preprocess_mobile(img, label):
    """
    Preprocess images using MobileNetV3 preprocessing.
    
    Args:
        img: Input image tensor
        label: Label tensor
    
    Returns:
        Preprocessed image and label
    """
    img = preprocess_mobilenet(img)
    return img, label


def create_data_generators(data_dir, train_ratio, img_size, batch_size):
    """
    Create train/validation datasets with specified split ratio.
    Uses validation_split parameter for consistent splitting.
    
    Args:
        data_dir: Data directory path (should be config.DATASET_DIR)
        train_ratio: Ratio for training (e.g., 0.9 for 90:10 split)
        img_size: Target image size
        batch_size: Batch size for datasets
    
    Returns:
        train_ds, val_ds, class_names
    """
    # Create training dataset
    train_ds_mobile = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=1.0 - train_ratio,
        subset='training',
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True,
        seed=SEED
    )
    
    # Create validation dataset (same split, different subset)
    val_ds_mobile = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=1.0 - train_ratio,
        subset='validation',
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False,
        seed=SEED
    )
    
    # Get class names
    class_names = train_ds_mobile.class_names
    
    # Apply preprocessing
    train_ds = train_ds_mobile.map(preprocess_mobile, AUTOTUNE).shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds_mobile.map(preprocess_mobile, AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names


# ------------------ 7B. BEST PER SPLIT REPORTS (VAL ACC + F1) ------------------

def main():
    """Main function to generate best per split reports."""
    
    # Load results CSV
    results_csv = os.path.join(config.OUTPUT_DIR, 'scenario_results.csv')
    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"Missing results CSV: {results_csv}")
    
    df_results = pd.read_csv(results_csv)
    
    # Check if we have any results
    if len(df_results) == 0:
        print("⚠ No results to evaluate best per split")
        return
    
    # Validate required columns
    required_cols = {'Scenario_ID', 'Split_Ratio', 'Val_Accuracy', 'F1_Score'}
    missing_cols = required_cols.difference(df_results.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in df_results: {sorted(missing_cols)}")
    
    # Load training history JSON to get detailed scenario information
    history_json = os.path.join(config.OUTPUT_DIR, 'training_history.json')
    if not os.path.exists(history_json):
        raise FileNotFoundError(f"Missing training history JSON: {history_json}")
    
    with open(history_json, 'r') as f:
        successful_results = json.load(f)
    
    # Rank scenarios by split ratio, validation accuracy, and F1 score
    ranked = df_results.sort_values(
        ['Split_Ratio', 'Val_Accuracy', 'F1_Score'],
        ascending=[True, False, False]
    )
    best_per_split = ranked.groupby('Split_Ratio', as_index=False).head(1)
    
    print(f"\n{'='*70}")
    print("BEST SCENARIOS PER SPLIT (VAL ACC + F1)")
    print(f"{'='*70}")
    print(best_per_split[['Scenario_ID', 'Split_Ratio', 'Optimizer', 'Learning_Rate', 'Val_Accuracy', 'F1_Score']].to_string(index=False))
    
    # Generate reports for each best scenario
    for _, row in best_per_split.iterrows():
        scenario_id = int(row['Scenario_ID'])
        split_ratio = row['Split_Ratio']
        
        # Find the matching result in successful_results
        matches = [r for r in successful_results if r['scenario_id'] == scenario_id]
        if not matches:
            raise ValueError(f"Scenario ID {scenario_id} not found in successful_results")
        best_result = matches[0]
        
        # Parse split ratio to get numeric value
        if isinstance(best_result['split_ratio'], str):
            split_num = float(best_result['split_ratio'].split(':')[0]) / 100.0
        else:
            split_num = best_result['split_ratio']
        
        # Create data generators for this split
        _, val_gen, class_names = create_data_generators(
            config.DATASET_DIR,
            train_ratio=split_num,
            img_size=config.IMG_SIZE_MOBILE,
            batch_size=config.BATCH_SIZE
        )
        
        # Construct model path
        scenario_model_path = os.path.join(
            config.MODELS_DIR,
            f"model_scenario_{scenario_id:02d}_{split_ratio.replace(':', '-')}_{best_result['optimizer']}_lr{best_result['learning_rate']}.keras"
        )
        if not os.path.exists(scenario_model_path):
            raise FileNotFoundError(f"Missing model file: {scenario_model_path}")
        
        # Load the model
        model = tf.keras.models.load_model(scenario_model_path)
        print(f"\n✓ Loaded model for split {split_ratio}: {scenario_model_path}")
        
        # Make predictions
        y_pred = model.predict(val_gen, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Get true labels
        val_ds_for_labels = tf.keras.utils.image_dataset_from_directory(
            config.DATASET_DIR,
            validation_split=1.0 - split_num,
            subset='validation',
            seed=SEED,
            image_size=(config.IMG_SIZE_MOBILE, config.IMG_SIZE_MOBILE),
            batch_size=config.BATCH_SIZE,
            shuffle=False
        )
        
        y_true_batches = []
        for x, y in val_ds_for_labels:
            y_true_batches.append(y.numpy())
        y_true = np.concatenate(y_true_batches, axis=0)
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
        y_true = y_true[:len(y_pred_classes)]
        
        # Generate classification report
        report_text = classification_report(
            y_true,
            y_pred_classes,
            target_names=class_names,
            labels=np.arange(len(class_names)),
            zero_division=0
        )
        report_dict = classification_report(
            y_true,
            y_pred_classes,
            target_names=class_names,
            labels=np.arange(len(class_names)),
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report_dict).transpose()
        report_csv = os.path.join(
            config.OUTPUT_DIR,
            f"classification_report_split_{split_ratio.replace(':', '-')}_scenario_{scenario_id:02d}.csv"
        )
        report_df.to_csv(report_csv, index=True)
        print("\nClassification Report:")
        print(report_text)
        print(f"✓ Classification report saved: {report_csv}")
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title(
            f"Confusion Matrix - Best Split {split_ratio} (Scenario {scenario_id})\n"
            f"{best_result['optimizer']}, LR={best_result['learning_rate']}",
            fontsize=13,
            fontweight='bold',
            pad=15
        )
        plt.ylabel('True Label', fontsize=11, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
        plt.tight_layout()
        
        fig_path = os.path.join(
            config.OUTPUT_DIR,
            f"confusion_matrix_split_{split_ratio.replace(':', '-')}_scenario_{scenario_id:02d}.png"
        )
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved: {fig_path}")
        plt.close()
        
        # Clean up
        del model
        tf.keras.backend.clear_session()
    
    print(f"\n{'='*70}")
    print("BEST PER SPLIT REPORTS GENERATION COMPLETED")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
