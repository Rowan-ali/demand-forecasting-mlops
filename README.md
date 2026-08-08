# E-commerce Demand Forecasting — MLOps Deployment

An end-to-end demand forecasting project that combines data analysis, machine learning, Spark optimization, model serialization, API deployment, monitoring, and an interactive dashboard.

The project was developed as a multi-phase data/ML engineering project and finalized with an MLOps deployment layer.

---

## Project Overview

The goal is to forecast product demand using business and operational features such as:

- Price
- Discount
- Stock availability
- Marketing effect
- Seasonal effect
- Public holiday

The final system exposes the trained model through a **FastAPI REST API** and provides a **Streamlit dashboard** for interactive predictions and monitoring.

### High-level pipeline

```text
Raw Demand Data
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering & Model Comparison
      ↓
Spark Processing & Optimization
      ↓
Gradient Boosting Model
      ↓
Model Serialization (Joblib)
      ↓
FastAPI Prediction Service
      ↓
Monitoring / Logging / Drift Checks / Retraining
      ↓
Streamlit Dashboard
```

---

## Dataset

The project uses the **Product Demand Forecasting Dataset** from Kaggle.

- Source: Kaggle
- Dataset: Product Demand Forecasting Dataset
- Records: 35,000
- Time period: January 2019 – December 2021
- License listed on the Kaggle dataset page: MIT
- Dataset type: synthetic/generated data

Source:

https://www.kaggle.com/datasets/chavindudulaj/product-demand-forecasting-dataset

The dataset is included in this repository under:

```text
data/demand_forecasting_data.csv
```

Please retain the source attribution when reusing the dataset.

---

## Dataset Features

The original dataset contains 13 columns:

| Feature | Description |
|---|---|
| `Date` | Observation date |
| `Product_ID` | Product identifier |
| `Base_Sales` | Base sales level |
| `Marketing_Campaign` | Marketing campaign type |
| `Marketing_Effect` | Marketing impact factor |
| `Seasonal_Trend` | Seasonal category |
| `Seasonal_Effect` | Seasonal multiplier |
| `Price` | Product price |
| `Discount` | Discount value |
| `Competitor_Price` | Competitor price |
| `Stock_Availability` | Available inventory |
| `Public_Holiday` | Holiday indicator |
| `Demand` | Target demand |

---

## Machine Learning

Several approaches were evaluated during the earlier project phases, including:

- Linear Regression
- Polynomial Regression
- Random Forest
- Gradient Boosting
- SARIMAX
- Prophet

The overall model comparison identified **Gradient Boosting** as the strongest machine-learning approach.

### Best model from the model-comparison phase

**Gradient Boosting — Lag + Seasonal features**

| Metric | Result |
|---|---:|
| R² (Train) | 0.7276 |
| R² (Test) | 0.7208 |
| RMSE | 18,672.48 |
| MAE | 14,261.55 |
| MAPE | 38.66% |

The model showed a small train/test R² gap (~0.0068), indicating relatively stable generalization in the evaluated setup.

---

## Production Model

For the MLOps handoff, the model was retrained on the optimized data and serialized with Joblib.

The production artifact is:

```text
models/gradient_boosting_demand_model.pkl
```

The corresponding feature order is stored in:

```text
models/feature_columns.txt
```

Production inference uses these six features in this exact order:

```text
Price
Discount
Stock_Availability
Marketing_Effect
Seasonal_Effect
Public_Holiday
```

### Final serialized-model training metrics

The model handoff notebook reports:

- R²: **0.7291**
- RMSE: **17,602.82**
- MAE: **13,342.69**

These metrics refer to the final serialized model trained for the Phase 5 handoff and should not be confused with the earlier model-comparison benchmark above.

---

## Spark Optimization

The earlier Big Data phase used Apache Spark to investigate partitioning and optimize processing.

The project started with:

- 35,000 records
- 200 Spark partitions
- Approximately 175 records per partition

The optimization reduced unnecessary partition overhead and applied techniques including:

- Partition coalescing
- Shuffle configuration tuning
- Adaptive Query Execution (AQE)
- Caching
- Performance benchmarking

The notebook reports an average performance improvement of **6.46×** in the final benchmark, with reported operation-level gains of:

| Operation | Reported Speedup |
|---|---:|
| Aggregation | 9.18× |
| Filter + Aggregation | 6.82× |
| Window Operation | 3.39× |
| Average | **6.46×** |

The exact benchmark results depend on the execution environment and should be interpreted as measurements from the project notebook rather than universal Spark performance guarantees.

---

# MLOps Layer

The MLOps service is implemented in:

```text
src/mlops_deployment.py
```

It provides a FastAPI-based prediction service with model lifecycle and monitoring functionality.

### Main capabilities

- Model loading with Joblib
- Feature-order validation
- Demand prediction endpoint
- Health checks
- Prediction logging
- Model information endpoint
- Metrics endpoint
- Drift/performance monitoring logic
- Retraining workflow
- Scheduled background tasks
- Report and log access

### Request validation

The API validates:

- Product ID format
- Non-negative price
- Discount range
- Non-negative stock availability
- Marketing effect
- Seasonal effect
- Public-holiday indicator

---

## Streamlit Dashboard

The interactive frontend is implemented in:

```text
src/dashboard.py
```

The dashboard communicates with the FastAPI backend and supports:

- Inputting product information
- Requesting demand predictions
- Viewing prediction history
- Viewing analytics
- Checking API health
- Accessing monitoring information

---

## API Endpoints

The FastAPI service includes endpoints for functionality such as:

- `GET /health`
- `POST /predict`
- `GET /metrics`
- Model information / monitoring endpoints
- Retraining and feedback-related endpoints

For the complete automatically generated API documentation, start the server and open:

```text
http://localhost:8000/docs
```

FastAPI provides the interactive Swagger UI there.

---

# Project Structure

```text
demand-forecasting-mlops/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── data/
│   └── demand_forecasting_data.csv
│
├── models/
│   ├── gradient_boosting_demand_model.pkl
│   └── feature_columns.txt
│
├── notebooks/
│   └── DDA_FINAL_PROJECT_TILL_PHASE_4.ipynb
│
├── src/
│   ├── mlops_deployment.py
│   └── dashboard.py
│
├── tests/
│   └── test_client.py
│

```

Runtime-generated files are intentionally excluded from version control. The API creates the required runtime directories automatically when it starts.

---

# Installation

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd demand-forecasting-mlops
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the API

From the repository root:

```bash
uvicorn src.mlops_deployment:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Interactive documentation:

```text
http://localhost:8000/docs
```

---

# Running the Dashboard

In a second terminal, with the virtual environment activated:

```bash
streamlit run src/dashboard.py
```

The dashboard will normally open at the local Streamlit address shown in the terminal.

Make sure the FastAPI backend is running before using the dashboard.

---

# Testing the API

With the API running:

```bash
python tests/test_client.py
```

The test client covers:

- Health check
- Single prediction
- Metrics retrieval
- Load testing

---

# Example Prediction Input

A prediction request contains the product identifier and the six model features:

```json
{
  "Product_ID": "P001",
  "Price": 45.50,
  "Discount": 10.0,
  "Stock_Availability": 500,
  "Marketing_Effect": 1.5,
  "Seasonal_Effect": 1.2,
  "Public_Holiday": 0
}
```

A successful response contains the predicted demand, timestamp, model version, and confidence interval fields.

---

# Monitoring & MLOps Concepts

The project demonstrates several practical MLOps concepts:

### Model lifecycle

```text
Train
  ↓
Serialize
  ↓
Load
  ↓
Serve
  ↓
Monitor
  ↓
Retrain
  ↓
Version
```

### Monitoring

The deployment layer records prediction information that can be used for:

- Prediction monitoring
- Model performance checks
- Drift detection
- Auditing
- Retraining decisions

### Retraining

The service includes a retraining workflow designed to update the model using newly available data and save the updated artifact.

For a real production system, retraining would normally be connected to a proper data pipeline, model registry, validation gate, and CI/CD workflow.

---

# Important Project Notes

### Model artifact paths

The current deployment code was originally written to expect the model and feature file at the project root.

In this cleaned repository they are stored under:

```text
models/
```

Before running the API, update the model and feature paths in:

```text
src/mlops_deployment.py
```

to:

```python
MODEL_PATH = "models/gradient_boosting_demand_model.pkl"
FEATURES_PATH = "models/feature_columns.txt"
```

For a more portable implementation, these paths can later be resolved relative to the project directory rather than the current working directory.

### Dashboard API URL

The current dashboard uses:

```text
http://localhost:8000
```

for the backend.

This is correct for local development. For cloud deployment, make the API URL configurable through an environment variable rather than hard-coding localhost.

---

# Future Improvements

Possible next steps include:

- Docker containerization
- CI/CD with GitHub Actions
- MLflow experiment/model tracking
- A proper model registry
- Automated data validation
- Automated model validation before deployment
- Cloud deployment
- Prometheus/Grafana monitoring
- Configurable API URLs
- Automated scheduled retraining pipeline
- Unit and integration test coverage
- Model explainability with SHAP

---

# Team

This project was developed collaboratively as a three-member team.

Add the three contributors' GitHub/LinkedIn profiles here:

- **[Team Member 1]**
- **[Team Member 2]**
- **[Team Member 3]**

---

# License

### Project code

Add the license that applies to your team's original project code here after confirming the academic/project requirements.

### Dataset

The dataset is attributed to its Kaggle source and is listed there under the MIT license:

https://www.kaggle.com/datasets/chavindudulaj/product-demand-forecasting-dataset

---

## Acknowledgements

- Kaggle dataset source: Product Demand Forecasting Dataset
- Scikit-learn for machine learning
- FastAPI for model serving
- Streamlit for the dashboard
- Apache Spark for distributed data processing
- MLflow and supporting MLOps tooling used in the deployment implementation
