"""
Customer Complaint Classification - Evaluation Script
Loads the trained model and evaluates on test set.
"""

import os
import json
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Constants
MODEL_PATH = "./artifacts/final_model"
DATASET_NAME = "hblim/customer-complaints"
LABEL_MAPPING = {0: "billing", 1: "delivery", 2: "product"}
MAX_LENGTH = 128

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def evaluate_on_test():
    """Evaluate the trained model on the test set."""
    
    print("\n" + "="*50)
    print("EVALUATING ON TEST SET")
    print("="*50)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        print("Please run train.py first to train the model.")
        return None
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    
    # Force CPU
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    
    # Load dataset
    dataset = load_dataset(DATASET_NAME)
    
    # Get test set
    test_data = dataset['test']
    
    # Make predictions
    print("Making predictions on test set...")
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for i, item in enumerate(test_data):
            # Tokenize the text
            inputs = tokenizer(
                item['text'],
                padding='max_length',
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # Make prediction
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.append(prediction.item())
            all_labels.append(item['label'])
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    true_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted'
    )
    
    # Get per-class metrics
    per_class_metrics = precision_recall_fscore_support(
        true_labels, predictions, average=None, labels=[0, 1, 2]
    )
    
    # Print results
    print("\n" + "="*50)
    print("TEST SET RESULTS")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    
    print("\nPer-class performance:")
    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<8}")
    print("-" * 50)
    for i in range(3):
        class_name = LABEL_MAPPING[i]
        p = per_class_metrics[0][i]
        r = per_class_metrics[1][i]
        f = per_class_metrics[2][i]
        support = per_class_metrics[3][i]
        print(f"{class_name:<12} {p:<10.4f} {r:<10.4f} {f:<10.4f} {support:<8}")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(true_labels, predictions, target_names=list(LABEL_MAPPING.values())))
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=list(LABEL_MAPPING.values()),
        yticklabels=list(LABEL_MAPPING.values())
    )
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    os.makedirs('artifacts', exist_ok=True)
    plt.savefig('artifacts/test_confusion_matrix.png')
    plt.close()
    
    # Prepare results with proper type conversion
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'per_class': {
            LABEL_MAPPING[i]: {
                'precision': float(per_class_metrics[0][i]),
                'recall': float(per_class_metrics[1][i]),
                'f1': float(per_class_metrics[2][i]),
                'support': int(per_class_metrics[3][i])
            }
            for i in range(3)
        }
    }
    
    # Convert any remaining numpy types
    results = convert_to_serializable(results)
    
    # Save to file
    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to artifacts/test_results.json")
    print(f"✅ Confusion matrix saved to artifacts/test_confusion_matrix.png")
    
    return results

if __name__ == "__main__":
    evaluate_on_test()