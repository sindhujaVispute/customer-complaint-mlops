"""
Unit tests for the customer complaint classification pipeline.
"""

import unittest
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class TestPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.dataset_name = "hblim/customer-complaints"
        cls.model_name = "distilbert-base-uncased"
        cls.label_mapping = {0: "billing", 1: "delivery", 2: "product"}
        cls.max_length = 128
        
    def test_dataset_loading(self):
        """Test that dataset loads correctly."""
        dataset = load_dataset(self.dataset_name)
        
        self.assertIn('train', dataset)
        self.assertIn('validation', dataset)
        self.assertIn('test', dataset)
        
        self.assertGreater(len(dataset['train']), 0)
        self.assertGreater(len(dataset['validation']), 0)
        self.assertGreater(len(dataset['test']), 0)
    
    def test_label_mapping(self):
        """Test that label mapping is correct."""
        dataset = load_dataset(self.dataset_name)
        
        # Check that labels are in the expected range
        labels = dataset['train']['label']
        unique_labels = set(labels)
        
        for label in unique_labels:
            self.assertIn(label, self.label_mapping)
            self.assertIsInstance(self.label_mapping[label], str)
        
        # Check label distribution
        self.assertEqual(min(labels), 0)
        self.assertEqual(max(labels), 2)
    
    def test_tokenizer(self):
        """Test that tokenizer works correctly."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        text = "My package has not arrived"
        
        tokenized = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Check tokenized output
        self.assertIn('input_ids', tokenized)
        self.assertIn('attention_mask', tokenized)
        self.assertEqual(tokenized['input_ids'].shape[1], self.max_length)
        self.assertEqual(tokenized['attention_mask'].shape[1], self.max_length)
        
        # Check that tokenizer works on a batch
        texts = ["Complaint 1", "Complaint 2"]
        batch_tokenized = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        self.assertEqual(batch_tokenized['input_ids'].shape[0], 2)
    
    def test_model_configuration(self):
        """Test that model is configured correctly."""
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3
        )
        
        # Check that model is on CPU
        self.assertEqual(next(model.parameters()).device.type, 'cpu')
        
        # Check number of labels
        self.assertEqual(model.config.num_labels, 3)
        
        # Test forward pass
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        text = "Test complaint"
        inputs = tokenizer(text, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        self.assertEqual(logits.shape[1], 3)
    
    def test_inference(self):
        """Test basic inference with a sample complaint."""
        try:
            # Try to load trained model
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            model_path = "./artifacts/final_model"
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.eval()
            
            # Test prediction
            text = "My package has not arrived"
            inputs = tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=-1)
            
            self.assertIn(prediction.item(), [0, 1, 2])
            
        except (FileNotFoundError, OSError):
            # Skip if model not trained yet
            self.skipTest("Model not found. Run train.py first.")

def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], exit=False)

if __name__ == "__main__":
    unittest.main()