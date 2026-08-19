"""
Customer Complaint Classification - Training Script
Fine-tunes DistilBERT for customer complaint classification using MLflow for experiment tracking.
"""

import os
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Constants
MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "hblim/customer-complaints"
MAX_LENGTH = 128
LABEL_MAPPING = {0: "billing", 1: "delivery", 2: "product"}
NUM_LABELS = 3

def load_and_explore_data():
    """Load dataset and explore its structure."""
    print("\n" + "="*50)
    print("LOADING DATASET")
    print("="*50)
    
    dataset = load_dataset(DATASET_NAME)
    
    # Explore dataset
    print(f"\nDataset splits: {dataset.keys()}")
    print(f"Train size: {len(dataset['train'])}")
    print(f"Validation size: {len(dataset['validation'])}")
    print(f"Test size: {len(dataset['test'])}")
    
    # Check label distribution
    train_labels = dataset['train']['label']
    unique, counts = np.unique(train_labels, return_counts=True)
    print("\nTraining set label distribution:")
    for label, count in zip(unique, counts):
        label_name = LABEL_MAPPING[label]
        print(f"  {label_name}: {count} ({count/len(train_labels)*100:.1f}%)")
    
    # Show sample complaints
    print("\nSample complaints from training set:")
    for i in range(3):
        text = dataset['train']['text'][i]
        label = dataset['train']['label'][i]
        print(f"  Example {i+1}: {text[:100]}...")
        print(f"  Label: {LABEL_MAPPING[label]}")
        print()
    
    return dataset

def tokenize_function(examples, tokenizer):
    """Tokenize the text column."""
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors=None
    )

def compute_metrics(eval_pred):
    """Compute evaluation metrics for the Trainer."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_confusion_matrix(labels, predictions, save_path='artifacts/confusion_matrix.png'):
    """Generate and save confusion matrix plot."""
    os.makedirs('artifacts', exist_ok=True)
    
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=list(LABEL_MAPPING.values()),
        yticklabels=list(LABEL_MAPPING.values())
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def train():
    """Main training function with MLflow tracking."""
    
    # Create MLflow experiment
    experiment_name = "customer-complaint-classifier"
    mlflow.set_experiment(experiment_name)
    
    # Load dataset
    dataset = load_and_explore_data()
    
    # Load tokenizer
    print("\n" + "="*50)
    print("INITIALIZING TOKENIZER AND MODEL")
    print("="*50)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Tokenize dataset - keep labels by not removing them
    print("\nTokenizing dataset...")
    
    def tokenize_with_labels(examples):
        """Tokenize and keep labels."""
        tokenized = tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
        )
        # Keep the label
        tokenized['label'] = examples['label']
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_with_labels,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    # Load model with ignore_mismatched_sizes to handle warnings
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True
    )
    
    # Training parameters
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        save_total_limit=2,
        seed=42,
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"distilbert-finetune-{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        
        # Log parameters
        params = {
            "base_model": MODEL_NAME,
            "dataset_name": DATASET_NAME,
            "learning_rate": training_args.learning_rate,
            "epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size,
            "max_length": MAX_LENGTH,
            "weight_decay": training_args.weight_decay,
            "optimizer": "AdamW",
            "seed": 42,
            "num_train_samples": len(dataset['train']),
            "num_val_samples": len(dataset['validation']),
        }
        
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        # Train the model
        print("\n" + "="*50)
        print("STARTING TRAINING")
        print("="*50)
        print("Training on CPU... This may take a few minutes.")
        trainer.train()
        
        # Evaluate on validation set
        print("\n" + "="*50)
        print("EVALUATING MODEL")
        print("="*50)
        eval_results = trainer.evaluate()
        
        # Get predictions for confusion matrix
        predictions = trainer.predict(tokenized_dataset['validation'])
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids
        
        # Calculate additional metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted'
        )
        
        # Log metrics
        metrics = {
            "train_loss": eval_results.get('train_loss', None),
            "eval_loss": eval_results.get('eval_loss', None),
            "eval_accuracy": eval_results.get('eval_accuracy', accuracy),
            "eval_precision": eval_results.get('eval_precision', precision),
            "eval_recall": eval_results.get('eval_recall', recall),
            "eval_f1": eval_results.get('eval_f1', f1),
        }
        
        for key, value in metrics.items():
            if value is not None:
                mlflow.log_metric(key, value)
        
        print("\nValidation Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        # Generate and log confusion matrix
        cm_path = plot_confusion_matrix(true_labels, pred_labels)
        mlflow.log_artifact(cm_path)
        
        # Save and log the model
        print("\n" + "="*50)
        print("SAVING MODEL")
        print("="*50)
        
        # Save model locally
        model_path = "./artifacts/final_model"
        os.makedirs(model_path, exist_ok=True)
        tokenizer.save_pretrained(model_path)
        trainer.save_model(model_path)
        
        # Log model to MLflow
        mlflow.log_artifacts(model_path, artifact_path="model")
        
        # Log evaluation results as CSV
        eval_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1'],
            'Value': [accuracy, precision, recall, f1]
        })
        eval_csv_path = 'artifacts/evaluation_results.csv'
        os.makedirs('artifacts', exist_ok=True)
        eval_df.to_csv(eval_csv_path, index=False)
        mlflow.log_artifact(eval_csv_path)
        
        # Log training configuration
        config_path = 'artifacts/training_config.txt'
        with open(config_path, 'w') as f:
            f.write("Training Configuration\n")
            f.write("="*30 + "\n")
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
            f.write("\nMetrics\n")
            f.write("="*30 + "\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        mlflow.log_artifact(config_path)
        
        print(f"\n Training complete! MLflow run ID: {run.info.run_id}")
        print(f"   Run name: {run.info.run_name}")
        print("\nTo view MLflow UI, run:")
        print("  mlflow ui")
        
        return trainer, tokenizer, metrics

if __name__ == "__main__":
    train()