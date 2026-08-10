"""
FastAPI application for customer complaint classification.
"""

import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn
import warnings
warnings.filterwarnings('ignore')

# Constants
MODEL_PATH = os.getenv("MODEL_PATH", "./artifacts/final_model")
LABEL_MAPPING = {0: "billing", 1: "delivery", 2: "product"}

# Initialize FastAPI app
app = FastAPI(
    title="Customer Complaint Classification API",
    description="Classify customer complaints into billing, delivery, or product",
    version="1.0.0"
)

# Load model at startup
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cpu")
model.to(device)
model.eval()
print("Model loaded successfully!")

class ComplaintRequest(BaseModel):
    """Request model for complaint classification."""
    text: str

class ComplaintResponse(BaseModel):
    """Response model for complaint classification."""
    text: str
    label: str
    confidence: float
    probabilities: dict

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Customer Complaint Classification API",
        "endpoints": {
            "/predict": "POST - Classify a complaint",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=ComplaintResponse)
async def predict(request: ComplaintRequest):
    """
    Classify a customer complaint.
    
    Args:
        request: Complaint text
        
    Returns:
        Prediction result with label and confidence
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Tokenize
    inputs = tokenizer(
        request.text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1)
    
    # Get results
    label = LABEL_MAPPING[prediction.item()]
    confidence = probabilities[0][prediction.item()].item()
    
    # Get all probabilities
    all_probs = {
        LABEL_MAPPING[i]: probabilities[0][i].item()
        for i in range(len(LABEL_MAPPING))
    }
    
    return ComplaintResponse(
        text=request.text,
        label=label,
        confidence=confidence,
        probabilities=all_probs
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)