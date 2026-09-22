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
## Technical Architecture

## Architecture
flowchart TB
    subgraph Data["Data Layer"]
        DS["Hugging Face Dataset<br/>hblim/customer-complaints<br/>1261 train / 210 val / 211 test"]
    end
    subgraph ML["ML Pipeline"]
        TK["Tokenizer<br/>DistilBERT"]
        MD["Model<br/>DistilBERT + 3-class head"]
        TR["Trainer<br/>Hugging Face"]
        EV["Evaluation<br/>scikit-learn"]
    end
    subgraph MLOps["MLOps Layer"]
        MF["MLflow Tracking"]
        MR["MLflow Model Registry"]
    end
    subgraph App["Application Layer"]
        T["train.py"]
        E["evaluate.py"]
        P["predict.py"]
        A["api.py / FastAPI"]
        TS["test_pipeline.py"]
    end
    subgraph Deploy["Deployment Layer"]
        DK["Dockerfile"]
        DKA["Dockerfile.api"]
        GHCR["GHCR"]
        GHA["GitHub Actions"]
    end
    DS --> TK
    TK --> MD
    MD --> TR
    TR --> EV
    EV --> MF
    TR --> MF
    MF --> MR
    T --> TR
    E --> MD
    P --> MD
    A --> MD
    TS --> T
    T --> DK
    A --> DKA
    DK --> GHCR
    DKA --> GHCR
    GHA --> GHCR

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/customer-complaint-mlops.git
cd customer-complaint-mlops
