"""
Customer Complaint Classification - Prediction Script
Loads the trained model and makes predictions on new complaints.
"""

import sys
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Constants
MODEL_PATH = "./artifacts/final_model"
LABEL_MAPPING = {0: "billing", 1: "delivery", 2: "product"}

def load_model():
    """Load the trained model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    
    # Force CPU
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def predict(text, model, tokenizer, device):
    """
    Make a prediction for a single complaint text.
    
    Args:
        text: Customer complaint text
        model: Trained model
        tokenizer: Tokenizer
        device: CPU device
    
    Returns:
        label: Predicted label (billing, delivery, or product)
        confidence: Confidence score
        probabilities: All class probabilities
    """
    # Tokenize input
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1)
    
    # Get results
    pred_label = prediction.item()
    label_name = LABEL_MAPPING[pred_label]
    confidence = probabilities[0][pred_label].item()
    
    # Get all class probabilities
    all_probs = {
        LABEL_MAPPING[i]: probabilities[0][i].item()
        for i in range(len(LABEL_MAPPING))
    }
    
    return label_name, confidence, all_probs

def main():
    parser = argparse.ArgumentParser(
        description='Classify customer complaints into billing, delivery, or product.'
    )
    parser.add_argument('text', type=str, nargs='*', 
                       help='Complaint text to classify. Enclose in quotes.')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, tokenizer, device = load_model()
    print("Model loaded successfully!\n")
    
    if args.interactive:
        # Interactive mode
        print("Customer Complaint Classifier")
        print("="*40)
        print("Enter complaints to classify (type 'quit' or 'exit' to stop):")
        print()
        
        while True:
            text = input("> ")
            if text.lower() in ['quit', 'exit', 'q']:
                break
            if not text.strip():
                continue
            
            label, confidence, probs = predict(text, model, tokenizer, device)
            
            print(f"\nPrediction: {label}")
            print(f"Confidence: {confidence:.4f}")
            print("Probabilities:")
            for cls, prob in probs.items():
                print(f"  {cls}: {prob:.4f}")
            print()
    
    elif args.text:
        # Single prediction mode
        text = ' '.join(args.text)
        label, confidence, probs = predict(text, model, tokenizer, device)
        
        print(f"Text: {text}")
        print("-" * 50)
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.4f}")
        print("\nProbabilities:")
        for cls, prob in probs.items():
            print(f"  {cls}: {prob:.4f}")
    
    else:
        # No arguments - show examples
        print("Usage Examples:")
        print("-" * 50)
        examples = [
            ("My package has still not arrived", "delivery"),
            ("I was charged twice for the same order", "billing"),
            ("The product stopped working after two days", "product"),
        ]
        
        for text, expected in examples:
            label, confidence, _ = predict(text, model, tokenizer, device)
            print(f"Text: {text}")
            print(f"Expected: {expected}")
            print(f"Prediction: {label} (confidence: {confidence:.4f})")
            print()
        
        print("\nTo make a prediction:")
        print('  python predict.py "Your complaint text here"')
        print("\nFor interactive mode:")
        print("  python predict.py --interactive")

if __name__ == "__main__":
    main()