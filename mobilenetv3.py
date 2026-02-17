#!/usr/bin/env python3
"""
Multi-Scenario MobileNetV3 Transfer Learning - Python Script Version

This is the Python script version of the mobilenetv3.ipynb notebook.
All functionality is the same except visualizations are saved instead of shown.
"""

# ======================================================================
# CODE CELL 2
# ======================================================================

import sys
import subprocess
import traceback

import os

# =====================================================
# MAIN TRAINING CODE
# =====================================================
import shutil
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as preprocess_mobilenet
from tensorflow.keras import layers, models, optimizers, callbacks
from pathlib import Path
import json
from datetime import datetime

tf.get_logger().setLevel('ERROR')

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Define AUTOTUNE for performance optimization
AUTOTUNE = tf.data.AUTOTUNE

# ------------------ 1. CONFIGURATION ------------------
class Config:
    BASE_INPUT = '/mgpfs/home/asusanto/_scratch/mobilnet-rice-leaf/dataset/Rice_Leaf_AUG/Rice_Leaf_AUG'
    WORK_DIR = '/mgpfs/home/asusanto/_scratch/mobilnet-rice-leaf/work'
    OUTPUT_DIR = '/mgpfs/home/asusanto/_scratch/mobilnet-rice-leaf/work/results'
    MODELS_DIR =  '/mgpfs/home/asusanto/_scratch/mobilnet-rice-leaf/work/models'
    DATASET_DIR =  '/mgpfs/home/asusanto/_scratch/mobilnet-rice-leaf/work/dataset'
    
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

print(f"Configuration loaded:")
print(f"  - Target classes: {config.TARGET_CLASSES}")
print(f"  - Number of classes: {config.NUM_CLASSES}")
print(f"  - Image size: {config.IMG_SIZE_MOBILE}x{config.IMG_SIZE_MOBILE}")
print(f"  - Batch size: {config.BATCH_SIZE}")
print(f"  - Output directory: {config.OUTPUT_DIR}")

# ======================================================================
# CODE CELL 3
# ======================================================================

# ------------------ 2. DATASET FILTERING & PREPARATION ------------------
def normalize_class_name(name):
    """
    Normalize folder names to snake_case (lowercase, underscores).
    """
    normalized = name.strip().lower().replace(" ", "_").replace("-", "_")
    normalized = "".join(ch for ch in normalized if ch.isalnum() or ch == "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")

def build_normalized_dir_map(source_dir):
    """
    Map normalized folder names to their actual folder names in source_dir.
    """
    if not os.path.exists(source_dir):
        return {}
    mapping = {}
    for entry in os.listdir(source_dir):
        entry_path = os.path.join(source_dir, entry)
        if os.path.isdir(entry_path):
            mapping[normalize_class_name(entry)] = entry
    return mapping

def filter_dataset_classes(source_dir, dest_dir, target_classes):
    """
    Copy only the target classes from source to destination.
    Preserves train/test directory structure.
    """
        
    if not os.path.exists(source_dir) and not os.path.exists(dest_dir):
        return
        
    source_map = build_normalized_dir_map(source_dir)
        
    for class_name in target_classes:
        normalized_name = normalize_class_name(class_name)
        source_folder = source_map.get(normalized_name)
        if not source_folder:
            continue
        source_class = os.path.join(source_dir, source_folder)
        dest_class = os.path.join(dest_dir, normalized_name)
        
        if os.path.exists(source_class):
            os.makedirs(dest_class, exist_ok=True)
            if not os.listdir(dest_class):  # Only copy if destination is empty
                shutil.copytree(source_class, dest_class, dirs_exist_ok=True)
                print(f"  Copied {source_folder} to {dest_class}/")
    
    print(f"✓ Dataset filtered to {len(target_classes)} classes")

# Copy and filter dataset
if not os.path.exists(config.WORK_DIR):
 # Create output directory for results
    print(f"Creating working directory at: {config.WORK_DIR}")
    os.makedirs(config.WORK_DIR)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.DATASET_DIR, exist_ok=True)
    print("Filtering and copying dataset to working directory...")
    filter_dataset_classes(config.BASE_INPUT, config.DATASET_DIR, config.TARGET_CLASSES)
else:
    print("✓ Working directory already exists")

# ======================================================================
# CODE CELL 4
# ======================================================================

# ------------------ 3. DATA SPLITTING & GENERATORS ------------------

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

def estimate_samples(dataset, batch_size):
    """
    Estimate sample count from dataset cardinality.
    Returns None if cardinality is unknown or infinite.
    """
    steps = tf.data.experimental.cardinality(dataset).numpy()
    if steps < 0:
        return None
    return int(steps) * batch_size

print("✓ Data generator function ready")

# ======================================================================
# CODE CELL 5
# ======================================================================

# ------------------ 4. DEFINE 10 TRAINING SCENARIOS ------------------

# Phase 1 Essential Fixes: 5 configs × 2 optimizers = 10 scenarios
# Testing multiple LRs (5e-4, 1e-3, 2e-3) with focus on 1e-3 baseline
scenarios = []

scenario_configs = [
    # Scenario 1: 90:10 split - test multiple learning rates
    {'split_ratio': 0.90, 'split_name': '90:10', 'lr': 5e-4, 'epochs': 30},   # Conservative
    {'split_ratio': 0.90, 'split_name': '90:10', 'lr': 1e-3, 'epochs': 30},   # Recommended (matches notebook)
    {'split_ratio': 0.90, 'split_name': '90:10', 'lr': 2e-3, 'epochs': 30},   # Aggressive
    
    # Scenario 2: 80:20 split - baseline with recommended LR
    {'split_ratio': 0.80, 'split_name': '80:20', 'lr': 1e-3, 'epochs': 30},
    
    # Scenario 3: 70:30 split - baseline with recommended LR
    {'split_ratio': 0.70, 'split_name': '70:30', 'lr': 1e-3, 'epochs': 30},
]

optimizers_list = ['Adam', 'SGD']

# Generate all 10 combinations (5 configs × 2 optimizers)
scenario_id = 1
for sc in scenario_configs:
    for opt in optimizers_list:
        scenarios.append({
            'id': scenario_id,
            'split_ratio': sc['split_ratio'],
            'split_name': sc['split_name'],
            'optimizer': opt,
            'learning_rate': sc['lr'],
            'epochs': sc['epochs']
        })
        scenario_id += 1

print(f"✓ Created {len(scenarios)} training scenarios")
print("\nScenario Summary:")
for i, s in enumerate(scenarios[:3], 1):
    print(f"  Scenario {s['id']}: {s['split_name']} split, {s['optimizer']}, LR={s['learning_rate']}, Epochs={s['epochs']}")
print(f"  ... and {len(scenarios)-3} more scenarios")

# ======================================================================
# CODE CELL 6
# ======================================================================

# ------------------ 5. BUILD FEATURE EXTRACTION MODEL ------------------

def build_feature_extraction_model(num_classes, img_size, optimizer_name, learning_rate, dropout_rate=0.5):
    """
    Build MobileNetV3Large model for feature extraction (frozen base).
    
    Args:
        num_classes: Number of output classes
        img_size: Input image size
        optimizer_name: 'Adam' or 'SGD'
        learning_rate: Learning rate for optimizer
        dropout_rate: Dropout rate before output layer
    
    Returns:
        Compiled Keras model
    """
    # Load pre-trained MobileNetV3Large without top layers
    base_model = MobileNetV3Large(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    
    # Freeze all layers in base model (feature extraction)
    for layer in base_model.layers:
        layer.trainable = False
    
    # Build custom classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(config.DENSE_UNITS, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Select optimizer
    if optimizer_name == 'Adam':
        opt = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'SGD':
        opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Compile model
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

print("✓ Model builder function ready")

# ======================================================================
# CODE CELL 7
# ======================================================================

# ------------------ 6. TRAINING LOOP FOR ALL SCENARIOS ------------------

results = []
best_val_acc = float("-inf")
best_model_path = None
best_scenario_id = None

print("=" * 70)
print(f"STARTING TRAINING FOR {len(scenarios)} SCENARIOS")
print("=" * 70)

for scenario in scenarios:
    print(f"\n{'='*70}")
    print(f"SCENARIO {scenario['id']}/{len(scenarios)}")
    print(f"  Split: {scenario['split_name']}")
    print(f"  Optimizer: {scenario['optimizer']}")
    print(f"  Learning Rate: {scenario['learning_rate']}")
    print(f"  Epochs: {scenario['epochs']}")
    print(f"{'='*70}")
    
    try:
        # Create data generators for this split ratio
        train_gen, val_gen, class_names = create_data_generators(
            config.DATASET_DIR,
            train_ratio=scenario['split_ratio'],
            img_size=config.IMG_SIZE_MOBILE,
            batch_size=config.BATCH_SIZE
        )
        
        train_samples = estimate_samples(train_gen, config.BATCH_SIZE)
        val_samples = estimate_samples(val_gen, config.BATCH_SIZE)
        if train_samples is not None:
            print(f"  Training samples: {train_samples}")
        else:
            print("  Training samples: unknown")
        if val_samples is not None:
            print(f"  Validation samples: {val_samples}")
        else:
            print("  Validation samples: unknown")
        
        # Build model
        model = build_feature_extraction_model(
            num_classes=config.NUM_CLASSES,
            img_size=config.IMG_SIZE_MOBILE,
            optimizer_name=scenario['optimizer'],
            learning_rate=scenario['learning_rate'],
            dropout_rate=config.DROPOUT_RATE
        )
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train model
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=scenario['epochs'],
            callbacks=[early_stop],
            verbose=1
        )
        
        # Extract final metrics
        final_train_acc = history.history['accuracy'][-1]
        final_train_loss = history.history['loss'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        # Store results
        result = {
            'scenario_id': scenario['id'],
            'split_ratio': scenario['split_name'],
            'optimizer': scenario['optimizer'],
            'learning_rate': scenario['learning_rate'],
            'epochs': scenario['epochs'],
            'train_accuracy': final_train_acc,
            'train_loss': final_train_loss,
            'val_accuracy': final_val_acc,
            'val_loss': final_val_loss,
            'history': history.history
        }
        
        # Calculate precision, recall, F1 score on validation set
        # Make predictions on validation set
        y_pred = model.predict(val_gen, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Recreate validation dataset to extract labels (predict consumed the dataset)
        val_ds_for_labels = tf.keras.utils.image_dataset_from_directory(
            config.DATASET_DIR,
            validation_split=1.0 - scenario['split_ratio'],
            subset='validation',
            seed=SEED,
            image_size=(config.IMG_SIZE_MOBILE, config.IMG_SIZE_MOBILE),
            batch_size=config.BATCH_SIZE,
            shuffle=False
        )
        
        # Extract true labels
        y_true_batches = []
        for x, y in val_ds_for_labels:
            y_true_batches.append(y.numpy())
        y_true = np.concatenate(y_true_batches, axis=0)
        
        # Convert to class indices if one-hot encoded
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
        
        # Ensure we match the prediction length
        y_true = y_true[:len(y_pred_classes)]
        
        # Calculate metrics (weighted average for multi-class)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_classes, average='weighted', zero_division=0
        )
        
        # Add metrics to result dictionary
        result['precision'] = float(precision)
        result['recall'] = float(recall)
        result['f1_score'] = float(f1)
        results.append(result)
        
        print(f"\n  ✓ Scenario {scenario['id']} completed!")
        print(f"    Final Train Accuracy: {final_train_acc:.4f}")
        print(f"    Final Val Accuracy: {final_val_acc:.4f}")
        print("     Precision: {precision:.4f}")
        print("     Recall: {recall:.4f}")
        print("     f1_score: {f1:.4f}")
        
        # Save model for this scenario in models directory
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        scenario_model_path = os.path.join(
            config.MODELS_DIR, 
            f"model_scenario_{scenario['id']:02d}_{scenario['split_name'].replace(':', '-')}_{scenario['optimizer']}_lr{scenario['learning_rate']}.keras"
        )
        model.save(scenario_model_path)
        print(f"    ✓ Model saved: {scenario_model_path}")
        
        # Save best model as training progresses
        if final_val_acc > best_val_acc:
            best_val_acc = final_val_acc
            best_scenario_id = scenario['id']
            best_model_path = os.path.join(config.MODELS_DIR, 'best_model.keras')
            model.save(best_model_path)
            print(f"    ✓ New best model saved: {best_model_path}")
        
        # Clear memory
        del model
        del train_gen
        del val_gen
        tf.keras.backend.clear_session()
        
    except Exception as e:
        print(f"\n  ✗ Scenario {scenario['id']} failed: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        results.append({
            'scenario_id': scenario['id'],
            'split_ratio': scenario['split_name'],
            'optimizer': scenario['optimizer'],
            'learning_rate': scenario['learning_rate'],
            'epochs': scenario['epochs'],
            'error': str(e),
            'traceback': traceback.format_exc()
        })

print(f"\n{'='*70}")
print(f"TRAINING COMPLETED: {len([r for r in results if 'val_accuracy' in r])}/{len(scenarios)} scenarios successful")
print(f"{'='*70}")

# ======================================================================
# CODE CELL 8
# ======================================================================

# ------------------ 7. PREPARE RESULTS DATAFRAME ------------------

# Filter successful results
successful_results = [r for r in results if 'val_accuracy' in r]

if len(successful_results) == 0:
    print("⚠ No successful training runs to analyze!")
else:
    # Create DataFrame
    df_results = pd.DataFrame([{
        'Scenario_ID': r['scenario_id'],
        'Split_Ratio': r['split_ratio'],
        'Optimizer': r['optimizer'],
        'Learning_Rate': r['learning_rate'],
        'Epochs': r['epochs'],
        'Train_Accuracy': r['train_accuracy'],
        'Val_Accuracy': r['val_accuracy'],
        'Train_Loss': r['train_loss'],
        'Val_Loss': r['val_loss'],
        'Precision': r.get('precision', 0.0),
        'Recall': r.get('recall', 0.0),
        'F1_Score': r.get('f1_score', 0.0)
    } for r in successful_results])
    
    # Sort by validation accuracy
    df_results = df_results.sort_values('Val_Accuracy', ascending=False).reset_index(drop=True)
    
    # Save to CSV
    results_csv = os.path.join(config.OUTPUT_DIR, 'scenario_results.csv')
    df_results.to_csv(results_csv, index=False)
    
    # Save detailed history to JSON
    history_json = os.path.join(config.OUTPUT_DIR, 'training_history.json')
    with open(history_json, 'w') as f:
        json.dump(successful_results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved:")
    print(f"  - CSV: {results_csv}")
    print(f"  - JSON: {history_json}")
    
    # Display top 10
    print(f"\n{'='*70}")
    print("TOP 10 SCENARIOS BY VALIDATION ACCURACY")
    print(f"{'='*70}")
    print(df_results.head(10).to_string(index=False))
    print(f"{'='*70}")

# ======================================================================
# CODE CELL 9
# ======================================================================

# ------------------ 8. VISUALIZATION: TOP 10 BAR CHART ------------------

if len(successful_results) > 0:
    # Get top 10 scenarios
    top_10 = df_results.head(10).copy()
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Create labels with all info
    top_10['Label'] = top_10.apply(
        lambda x: f"S{x['Scenario_ID']}: {x['Split_Ratio']}, {x['Optimizer']}, LR={x['Learning_Rate']}", 
        axis=1
    )
    
    # Create horizontal bar chart
    bars = plt.barh(range(len(top_10)), top_10['Val_Accuracy'], color='steelblue', alpha=0.8)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, top_10['Val_Accuracy'])):
        plt.text(val + 0.005, i, f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
    
    plt.yticks(range(len(top_10)), top_10['Label'])
    plt.xlabel('Validation Accuracy', fontsize=12, fontweight='bold')
    plt.title('Top 10 Scenarios by Validation Accuracy', fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(config.OUTPUT_DIR, 'top_10_scenarios.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Top 10 bar chart saved: {fig_path}")
    # plt.show() # Removed - saving to file instead
else:
    print("⚠ No results to visualize")

# ======================================================================
# CODE CELL 10
# ======================================================================

# ------------------ 9. VISUALIZATION: TOP 3 TRAINING CURVES ------------------

if len(successful_results) >= 3:
    top_3_ids = df_results.head(3)['Scenario_ID'].tolist()
    top_3_results = [r for r in successful_results if r['scenario_id'] in top_3_ids]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, result in enumerate(top_3_results):
        ax = axes[idx]
        history = result['history']
        epochs_range = range(1, len(history['accuracy']) + 1)
        
        # Plot accuracy
        ax.plot(epochs_range, history['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
        ax.plot(epochs_range, history['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
        
        ax.set_title(
            f"Scenario {result['scenario_id']}: {result['split_ratio']}\n"
            f"{result['optimizer']}, LR={result['learning_rate']}, Val Acc={result['val_accuracy']:.4f}",
            fontsize=11, fontweight='bold'
        )
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(config.OUTPUT_DIR, 'top_3_training_curves.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Top 3 training curves saved: {fig_path}")
    # plt.show() # Removed - saving to file instead
else:
    print("⚠ Not enough results to plot top 3")

# ======================================================================
# CODE CELL 11
# ======================================================================

# ------------------ 10. VISUALIZATION: CONFUSION MATRIX FOR BEST SCENARIO ------------------

if len(successful_results) > 0:
    # Get best scenario
    best_scenario_id = df_results.iloc[0]['Scenario_ID']
    best_result = [r for r in successful_results if r['scenario_id'] == best_scenario_id][0]
    
    print(f"\n{'='*70}")
    print(f"EVALUATING BEST SCENARIO: {best_scenario_id}")
    print(f"  Split: {best_result['split_ratio']}")
    print(f"  Optimizer: {best_result['optimizer']}")
    print(f"  Learning Rate: {best_result['learning_rate']}")
    print(f"  Val Accuracy: {best_result['val_accuracy']:.4f}")
    print(f"{'='*70}")
    
    if isinstance(best_result['split_ratio'], str):
        split_num = float(best_result['split_ratio'].split(':')[0]) / 100.0
    else:
        split_num = best_result['split_ratio']
    
    # Recreate data generators for best scenario
    train_gen, val_gen, class_names = create_data_generators(
        config.DATASET_DIR,
        train_ratio=split_num,
        img_size=config.IMG_SIZE_MOBILE,
        batch_size=config.BATCH_SIZE
    )
    
    # Load best model if available; otherwise retrain and save
    if best_model_path and os.path.exists(best_model_path):
        best_model = tf.keras.models.load_model(best_model_path)
        print(f"\n✓ Loaded best model from: {best_model_path}")
    else:
        best_model = build_feature_extraction_model(
            num_classes=config.NUM_CLASSES,
            img_size=config.IMG_SIZE_MOBILE,
            optimizer_name=best_result['optimizer'],
            learning_rate=best_result['learning_rate'],
            dropout_rate=config.DROPOUT_RATE
        )
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=0
        )
        
        best_model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=best_result['epochs'],
            callbacks=[early_stop],
            verbose=0
        )
        
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        best_model_path = os.path.join(config.MODELS_DIR, 'best_model.keras')
        best_model.save(best_model_path)
        print(f"\n✓ Best model saved: {best_model_path}")
    
    # Generate predictions
    y_pred = best_model.predict(val_gen, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Recreate validation dataset to extract labels (predict consumed the dataset)
    val_ds_for_labels = tf.keras.utils.image_dataset_from_directory(
        config.DATASET_DIR,
        validation_split=1.0 - split_num,
        subset='validation',
        seed=SEED,
        image_size=(config.IMG_SIZE_MOBILE, config.IMG_SIZE_MOBILE),
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    
    # Extract true labels
    y_true_batches = []
    for x, y in val_ds_for_labels:
        y_true_batches.append(y.numpy())
    y_true = np.concatenate(y_true_batches, axis=0)
    
    # Convert to class indices if one-hot encoded
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    # Ensure we match the prediction length
    y_true = y_true[:len(y_pred_classes)]
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - Best Scenario {best_scenario_id}\n'
              f'{best_result["split_ratio"]}, {best_result["optimizer"]}, LR={best_result["learning_rate"]}',
              fontsize=13, fontweight='bold', pad=15)
    plt.ylabel('True Label', fontsize=11, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(config.OUTPUT_DIR, 'best_scenario_confusion_matrix.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved: {fig_path}")
    # plt.show() # Removed - saving to file instead
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    
    # Cleanup
    del best_model
    tf.keras.backend.clear_session()
    
else:
    print("⚠ No results to evaluate")

# ======================================================================
# CODE CELL 12
# ======================================================================

# ------------------ 11. FINAL SUMMARY ------------------

if len(successful_results) > 0:
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"Total Scenarios: {len(scenarios)}")
    print(f"Successful Runs: {len(successful_results)}")
    print(f"Failed Runs: {len(scenarios) - len(successful_results)}")
    print(f"\nBest Scenario: {df_results.iloc[0]['Scenario_ID']}")
    print(f"  Configuration: {df_results.iloc[0]['Split_Ratio']}, "
          f"{df_results.iloc[0]['Optimizer']}, LR={df_results.iloc[0]['Learning_Rate']}")
    print(f"  Validation Accuracy: {df_results.iloc[0]['Val_Accuracy']:.4f}")
    print(f"\nWorst Scenario: {df_results.iloc[-1]['Scenario_ID']}")
    print(f"  Configuration: {df_results.iloc[-1]['Split_Ratio']}, "
          f"{df_results.iloc[-1]['Optimizer']}, LR={df_results.iloc[-1]['Learning_Rate']}")
    print(f"  Validation Accuracy: {df_results.iloc[-1]['Val_Accuracy']:.4f}")
    
    # Performance comparison by optimizer
    print(f"\n{'='*70}")
    print("PERFORMANCE BY OPTIMIZER")
    print(f"{'='*70}")
    for opt in ['Adam', 'SGD']:
        opt_results = df_results[df_results['Optimizer'] == opt]
        if len(opt_results) > 0:
            print(f"{opt}:")
            print(f"  Mean Val Accuracy: {opt_results['Val_Accuracy'].mean():.4f}")
            print(f"  Best Val Accuracy: {opt_results['Val_Accuracy'].max():.4f}")
            print(f"  Worst Val Accuracy: {opt_results['Val_Accuracy'].min():.4f}")
    
    # Performance comparison by split ratio
    print(f"\n{'='*70}")
    print("PERFORMANCE BY SPLIT RATIO")
    print(f"{'='*70}")
    for split in ['90:10', '80:20', '70:30']:
        split_results = df_results[df_results['Split_Ratio'] == split]
        if len(split_results) > 0:
            print(f"{split}:")
            print(f"  Mean Val Accuracy: {split_results['Val_Accuracy'].mean():.4f}")
            print(f"  Best Val Accuracy: {split_results['Val_Accuracy'].max():.4f}")
    
    print(f"\n{'='*70}")
    print("All outputs saved to:", config.OUTPUT_DIR)
    print(f"{'='*70}")
    print("\n✓ Multi-scenario transfer learning experiment completed successfully!")
else:
    print("\n⚠ No successful results to summarize")
