# Customer Complaint Classification MLOps Pipeline

A complete MLOps project that fine-tunes DistilBERT to classify customer complaints into billing, delivery, or product categories. The project demonstrates experiment tracking, model versioning, and reproducible ML workflows using MLflow.

## Problem Statement

Automatically classify customer complaints into three categories:
- **Billing**: Issues related to payments, charges, invoices
- **Delivery**: Issues related to shipping, delivery delays, missing packages
- **Product**: Issues related to product quality, functionality, defects

## Technologies

- **Python 3.8+**
- **Hugging Face Transformers**: Model fine-tuning
- **Hugging Face Datasets**: Dataset management
- **DistilBERT**: Lightweight transformer model
- **PyTorch**: Deep learning framework
- **MLflow**: Experiment tracking and model registry
- **Scikit-learn**: Metrics and evaluation
- **Matplotlib/Seaborn**: Visualization
- **GitHub Actions**: CI/CD


## Hardware Requirements

- **CPU only** (No GPU required)
- 8GB+ RAM recommended
- 2GB+ free disk space

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/customer-complaint-mlops.git
cd customer-complaint-mlops
