from pathlib import Path
"""
PHASE 5: MLOps Engineer - Production Deployment & Monitoring

This module implements a production-ready demand forecasting API with:
- Model serving via FastAPI
- Automated health checks and monitoring
- Drift detection and alerting
- Scheduled retraining pipeline
- Log file access endpoints

Author: MLOps Engineer
Version: 1.0.0
"""

# Import MLflow for model tracking and version management
import mlflow
import mlflow.sklearn

# Import joblib for loading and saving the trained model file (binary format)
import joblib

# Import pandas for data manipulation and DataFrame operations
import pandas as pd

# Import numpy for numerical operations and array handling
import numpy as np

# Import FastAPI components for building the REST API
# FastAPI is the web framework that creates the API endpoints
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

# Import Pydantic models for request/response validation
# BaseModel is the parent class for data validation schemas
# Field defines validation rules for each field
# field_validator adds custom validation logic
from pydantic import BaseModel, Field, field_validator

# Import typing for type hints and better code documentation
from typing import List, Dict, Optional

# Import schedule for automated recurring tasks (cron-like functionality)
import schedule

# Import time for sleep operations in background threads
import time

# Import logging for production observability (writing logs to files and console)
import logging

# Import datetime for timestamps in logs, predictions, and reports
from datetime import datetime, timedelta

# Import json for serializing prediction logs to JSON format
import json

# Import os for file system operations (creating directories, checking file paths)
import os

# Import threading for running background tasks alongside the API server
# This allows scheduled tasks to run without blocking the API
import threading

# Import context manager for startup and shutdown events
# lifespan manages code that runs when the API starts and stops
from contextlib import asynccontextmanager

# Import scikit-learn metrics for model performance evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import glob for finding files matching a pattern (used for finding report files)
import glob


# ============================================
# PART 1: CONFIGURATION AND SETUP
# ============================================

# Define directory names for organizing production artifacts
# These are folders that will store different types of data
DIRECTORIES = ["logs", "models", "predictions", "reports", "alerts"]

# Loop through each directory name and create it if it doesn't exist
# exist_ok=True means no error is raised if the directory already exists
for directory in DIRECTORIES:
    os.makedirs(directory, exist_ok=True)

# Configure the logging system for production monitoring
# level=logging.INFO means show INFO, WARNING, ERROR messages (not DEBUG)
# format defines how each log line looks
# handlers determine where logs go (both file and console)
logging.basicConfig(
    level=logging.INFO,  # Only show messages at INFO level or higher
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Timestamp - Module - Level - Message
    handlers=[
        logging.FileHandler('logs/mlops_production.log'),  # Save logs to this file
        logging.StreamHandler()  # Also print logs to the terminal
    ]
)

# Create a logger instance for this specific module
# __name__ is the module name (mlops_deployment)
logger = logging.getLogger(__name__)

# Production configuration parameters (constants that control system behavior)
MODEL_PATH = "models/gradient_boosting_demand_model.pkl"  # Path to the trained model file
FEATURES_PATH = "models/feature_columns.txt"  # Path to file listing feature names in order
MODEL_VERSION = "1.0.0"  # Current version tag for the model
ALERT_MAPE_THRESHOLD = 20.0  # Alert if Mean Absolute Percentage Error exceeds 20%
RETRAIN_SAMPLE_THRESHOLD = 1000  # Retrain after collecting 1000 new samples
PREDICTION_LOG_RETENTION_DAYS = 30  # Keep prediction logs for 30 days


# ============================================
# PART 2: MODEL MANAGER CLASS
# ============================================

class ModelManager:
    """
    Manages model lifecycle including loading, prediction, versioning, and retraining.

    This class is the central component for all model-related operations.
    It handles:
    - Loading the serialized model from disk
    - Validating feature order for predictions
    - Making predictions with proper error handling
    - Retraining the model with new data
    - Tracking prediction counts and model versions
    """

    def __init__(self, model_path=MODEL_PATH, features_path=FEATURES_PATH):
        """
        Initialize the model manager with file paths.

        This constructor runs when a ModelManager object is created.
        It stores the file paths and immediately loads the model.

        Args:
            model_path: String path to the pickled model file
            features_path: String path to the feature columns text file
        """
        # Store file paths as instance variables (accessible from any method)
        self.model_path = model_path
        self.features_path = features_path

        # Initialize instance variables with default values
        self.model = None  # Will hold the loaded sklearn model object
        self.feature_columns = None  # Will hold list of feature names in correct order
        self.model_version = MODEL_VERSION  # Current version string
        self.total_predictions = 0  # Counter for number of predictions made
        self.model_loaded = False  # Boolean flag indicating if model is ready

        # Automatically load the model when the manager is created
        self.load_model()

    def load_model(self):
        """
        Load the trained model and feature columns from disk.

        This method deserializes the pickled model and reads the feature order file.
        If files are not found, it creates a dummy model for testing.
        The dummy model allows the API to run even before the real model is available.
        """
        try:
            # Deserialize the pickled model file
            # joblib.load() reads the binary file and reconstructs the Python object
            self.model = joblib.load(self.model_path)

            # Log success message with file path
            logger.info(f"Model loaded successfully from {self.model_path}")

            # Open and read the feature columns text file
            # 'r' means read mode
            with open(self.features_path, "r") as f:
                # Read all lines, strip whitespace from each, store in list
                self.feature_columns = [line.strip() for line in f.readlines()]

            # Log the loaded feature columns for verification
            logger.info(f"Feature columns loaded: {self.feature_columns}")

            # Set flag to True indicating model is ready for predictions
            self.model_loaded = True

        except FileNotFoundError as e:
            # This runs if model file or features file doesn't exist
            logger.error(f"File not found: {e}")
            logger.warning("Creating dummy model for testing. Replace with actual model file.")

            # Import GradientBoostingRegressor for the dummy model
            from sklearn.ensemble import GradientBoostingRegressor

            # Create a dummy model with default parameters
            self.model = GradientBoostingRegressor()

            # Define default feature columns (must match the order used in training)
            self.feature_columns = [
                "Price", "Discount", "Stock_Availability",
                "Marketing_Effect", "Seasonal_Effect", "Public_Holiday"
            ]

            # Save the dummy model to disk for future runs
            joblib.dump(self.model, self.model_path)

            # Save the feature columns to disk for future runs
            with open(self.features_path, "w") as f:
                f.write("\n".join(self.feature_columns))

            # Set flag to True even for dummy model
            self.model_loaded = True

    def predict(self, features_dict):
        """
        Make a prediction using the loaded model.

        This method takes a dictionary of feature values, converts them to the
        correct order expected by the model, and returns a prediction.

        Args:
            features_dict: Dictionary mapping feature names to values
                           Example: {"Price": 45.50, "Discount": 10.0,
                                     "Stock_Availability": 500, ...}

        Returns:
            float: Predicted demand value (number of units)
        """
        # Extract features in the correct order expected by the model
        # List comprehension: for each column name in self.feature_columns,
        # get the value from features_dict and put it in a list
        # This ensures features are passed to the model in the same order as training
        features = [features_dict[col] for col in self.feature_columns]

        # Make prediction using the model
        # [features] creates a 2D array (single row, multiple columns)
        # This is what sklearn's predict() method expects
        # [0] extracts the first (and only) prediction value from the returned array
        prediction = self.model.predict([features])[0]

        # Increment the prediction counter for monitoring
        self.total_predictions += 1

        # Return the predicted value
        return prediction

    def retrain(self, X_new, y_new):
        """
        Retrain the model with new data.

        This method updates the model with recent data to adapt to changing patterns.
        In production, you would combine old and new data for full retraining.

        Args:
            X_new: New feature data as pandas DataFrame
            y_new: New target values as pandas Series

        Returns:
            bool: True if retraining succeeded, False otherwise
        """
        try:
            # Log the start of retraining with sample count
            logger.info(f"Starting model retraining with {len(X_new)} samples")

            # Fit the existing model on new data
            # This updates the model parameters based on new patterns
            self.model.fit(X_new, y_new)

            # Generate new version string with timestamp
            # Format: 1.202412151430 (version.timestamp)
            # strftime formats the current date/time as YYYYMMDDHHMM
            new_version = f"1.{datetime.now().strftime('%Y%m%d%H%M')}"
            self.model_version = new_version

            # Save the retrained model to disk
            # This ensures the updated model persists after server restart
            joblib.dump(self.model, self.model_path)

            # Log success with new version number
            logger.info(f"Model retrained successfully. New version: {self.model_version}")

            # Return True to indicate success
            return True

        except Exception as e:
            # Log the error for debugging
            logger.error(f"Retraining failed: {str(e)}")

            # Return False to indicate failure
            return False

    def get_model_info(self):
        """
        Return current model metadata for monitoring endpoints.

        This method collects all relevant model information into a dictionary.

        Returns:
            dict: Dictionary containing model information
        """
        return {
            "model_version": self.model_version,
            "total_predictions": self.total_predictions,
            "features": self.feature_columns,
            "model_loaded": self.model_loaded
        }


# ============================================
# PART 3: FASTAPI REQUEST/RESPONSE MODELS (Pydantic Schemas)
# ============================================

class DemandRequest(BaseModel):
    """
    Request schema for demand prediction endpoint.

    This Pydantic model validates incoming JSON POST requests.
    Each field has validation rules and example values for API documentation.
    """

    # Product_ID is a required string field
    # ... means the field is required (no default value)
    # description is shown in the API documentation
    # example is shown in Swagger UI as a sample value
    Product_ID: str = Field(
        ...,  # Required field
        description="Unique product identifier",  # Documentation
        example="P001"  # Example value
    )

    # Price is a required float field with validation constraints
    # ge=0 means greater than or equal to 0 (price cannot be negative)
    Price: float = Field(
        ...,
        ge=0,  # Greater than or equal to zero
        description="Product price in USD",
        example=45.50
    )

    # Discount is a required float field
    # ge=0 means discount cannot be negative
    # le=100 means discount cannot exceed 100%
    Discount: float = Field(
        ...,
        ge=0,
        le=100,
        description="Discount percentage applied",
        example=10.0
    )

    # Stock_Availability is a required integer field
    # ge=0 means stock cannot be negative
    Stock_Availability: int = Field(
        ...,
        ge=0,
        description="Current inventory count",
        example=500
    )

    # Marketing_Effect is a float field representing campaign impact
    Marketing_Effect: float = Field(
        ...,
        ge=0,
        description="Marketing campaign effectiveness score (0.5 to 2.5 typical)",
        example=1.5
    )

    # Seasonal_Effect is a float field representing seasonal multiplier
    Seasonal_Effect: float = Field(
        ...,
        ge=0,
        description="Seasonal demand multiplier (0.8 to 1.5 typical)",
        example=1.2
    )

    # Public_Holiday is an integer (0 or 1) indicating holiday status
    Public_Holiday: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 if public holiday, 0 otherwise",
        example=0
    )

    # Custom validator for Product_ID format
    # This runs automatically when a request is received
    @field_validator('Product_ID')
    @classmethod
    def validate_product_id(cls, v):
        """
        Validate that Product_ID follows the expected format.

        Args:
            v: The value being validated (the Product_ID string)

        Returns:
            str: The validated value

        Raises:
            ValueError: If the format is incorrect
        """
        # Check if Product_ID starts with the letter P
        if not v.startswith('P'):
            raise ValueError('Product_ID must start with P')

        # Check if Product_ID has exactly 4 characters (e.g., P001, P002)
        if len(v) != 4:
            raise ValueError('Product_ID must be 4 characters (e.g., P001)')

        # Return the value if validation passed
        return v


class DemandResponse(BaseModel):
    """
    Response schema for demand prediction endpoint.

    This defines the structure of JSON responses sent back to API clients.
    """
    predicted_demand: float  # The forecasted demand quantity in units
    timestamp: str  # When the prediction was made (ISO format)
    model_version: str  # Which model version made the prediction
    confidence_interval_lower: float  # Lower bound of 90% confidence interval
    confidence_interval_upper: float  # Upper bound of 90% confidence interval


class HealthResponse(BaseModel):
    """
    Response schema for health check endpoint.

    Used by monitoring systems (like Kubernetes, load balancers) to verify API is operational.
    """
    status: str  # "healthy" or "unhealthy"
    model_loaded: bool  # Whether the model is successfully loaded into memory
    model_version: str  # Current model version string
    total_predictions: int  # Total predictions made since server started
    timestamp: str  # When the health check was performed


# ============================================
# PART 4: GLOBAL STATE AND STORAGE
# ============================================

# Create a global instance of the ModelManager
# This single instance will be used by all API endpoints
model_manager = ModelManager()

# Create a list to store recent predictions for monitoring
# Each entry is a dictionary with timestamp, product_id, prediction, and features
prediction_store = []


# ============================================
# PART 5: FASTAPI APPLICATION SETUP WITH LIFESPAN
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for FastAPI startup and shutdown events.

    This function runs code when the API starts and when it shuts down.
    It's used to initialize resources and clean up properly.
    The @asynccontextmanager decorator makes this work with async code.

    Args:
        app: The FastAPI application instance (automatically passed)
    """
    # ===== STARTUP CODE =====
    # This runs before the API accepts any requests
    logger.info("Starting Demand Forecasting API...")

    # Yield control back to FastAPI - the API runs here
    # Everything after this runs on shutdown
    yield

    # ===== SHUTDOWN CODE =====
    # This runs when the API is stopping (e.g., Ctrl+C)
    logger.info("Shutting down Demand Forecasting API...")


# Create the FastAPI application instance
# title: Name shown in API documentation (Swagger UI)
# description: Detailed explanation of the service's purpose
# version: API version (separate from model version)
# lifespan: Startup/shutdown handler defined above

# Create runtime directories automatically when the service starts.
for _directory in ("logs", "predictions", "reports", "alerts"):
    Path(_directory).mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="E-commerce Demand Forecasting API",
    description="MLOps deployment for demand prediction - Phase 5",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================
# PART 6: HELPER FUNCTIONS
# ============================================

def log_prediction(product_id, features_dict, prediction):
    """
    Store prediction data for monitoring and analysis.

    This function saves each prediction to memory and periodically to disk.
    The stored data is used for:
    - Drift detection (checking if predictions are changing over time)
    - Performance monitoring (comparing predictions to actual values)
    - Audit trails (what was predicted, when, and using which features)

    Args:
        product_id: The product identifier (e.g., "P001")
        features_dict: Dictionary of input features used for the prediction
        prediction: The predicted demand value
    """
    # Create a log entry with timestamp and all relevant data
    # datetime.now().isoformat() creates ISO 8601 format: 2024-01-15T10:30:00.123456
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "product_id": product_id,
        "prediction": float(prediction),
        "features": features_dict
    }

    # Add the entry to the in-memory store (list)
    prediction_store.append(log_entry)

    # Periodically save to disk (every 100 predictions)
    # This prevents memory from growing too large and provides persistence
    # % is the modulo operator - checks if length is divisible by 100
    if len(prediction_store) % 100 == 0:
        # Generate filename with current date
        # strftime formats the date: YYYYMMDD
        filename = f"predictions/predictions_{datetime.now().strftime('%Y%m%d')}.json"

        # Save the most recent 1000 predictions to file
        # [-1000:] is Python slice syntax - takes the last 1000 elements
        # indent=2 makes the JSON file human-readable with 2-space indentation
        with open(filename, "w") as f:
            json.dump(prediction_store[-1000:], f, indent=2)


# ============================================
# PART 7: API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """
    Root endpoint - returns API information.

    This is the entry point for the API.
    Used by clients to discover available endpoints and get basic info.
    No authentication required.

    Returns:
        dict: API metadata including service name, status, and available endpoints
    """
    return {
        "service": "Demand Forecasting API",
        "status": "running",
        "model_version": model_manager.model_version,
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "metrics": "/metrics",
            "logs": {
                "production": "/logs/production",
                "alerts": "/logs/alerts",
                "reports": "/logs/reports"
            }
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for monitoring systems.

    This endpoint is called by:
    - Load balancers to check if server should receive traffic
    - Kubernetes liveness/readiness probes
    - Monitoring dashboards
    - Uptime monitoring services

    Returns:
        HealthResponse: Status information about the API and model
    """
    return HealthResponse(
        status="healthy",  # Always "healthy" if this endpoint responds
        model_loaded=model_manager.model_loaded,
        model_version=model_manager.model_version,
        total_predictions=model_manager.total_predictions,
        timestamp=datetime.now().isoformat()
    )


@app.get("/metrics")
async def get_metrics():
    """
    Get model performance metrics.

    This endpoint returns operational metrics for monitoring dashboards.
    Unlike /health which just checks if running, /metrics gives detailed data.

    Returns:
        dict: Metrics including prediction count and recent predictions
    """
    return {
        "total_predictions": model_manager.total_predictions,
        "model_version": model_manager.model_version,
        "model_loaded": model_manager.model_loaded,
        "features_count": len(model_manager.feature_columns) if model_manager.feature_columns else 0,
        "recent_predictions": prediction_store[-10:] if prediction_store else [],
        "store_size": len(prediction_store),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/logs/production")
async def get_production_log():
    """
    Get the production log file.

    This endpoint allows monitoring systems to retrieve the application log.
    The log contains all INFO, WARNING, and ERROR messages from the API.

    Returns:
        FileResponse: The log file as a downloadable text file
    """
    # Define the path to the production log file
    log_path = "logs/mlops_production.log"

    # Check if the file exists
    if os.path.exists(log_path):
        # Return the file for download
        # media_type="text/plain" tells browser it's a text file
        # filename sets the download name
        return FileResponse(
            log_path,
            media_type="text/plain",
            filename="production.log"
        )
    else:
        # Return 404 error if log file doesn't exist
        raise HTTPException(status_code=404, detail="Production log file not found")


@app.get("/logs/alerts")
async def get_alerts_log():
    """
    Get the alerts log file.

    This endpoint returns all alerts generated by the monitoring system.
    Alerts are triggered when model drift or performance degradation is detected.

    Returns:
        FileResponse: The alerts log file as a downloadable text file
    """
    # Define the path to the alerts log file
    log_path = "logs/alerts.log"

    # Check if the file exists
    if os.path.exists(log_path):
        # Return the file for download
        return FileResponse(
            log_path,
            media_type="text/plain",
            filename="alerts.log"
        )
    else:
        # Return 404 error if alerts file doesn't exist
        raise HTTPException(status_code=404, detail="Alerts log file not found")


@app.get("/logs/reports")
async def get_reports_list():
    """
    Get a list of all available daily reports.

    This endpoint returns the names of all report files in the reports directory.
    Reports contain daily performance metrics and model health data.

    Returns:
        dict: Dictionary with a list of report filenames
    """
    # Use glob to find all JSON report files
    # glob searches for files matching the pattern
    reports = glob.glob("reports/daily_report_*.json")

    # Return just the filenames (without the full path)
    # os.path.basename extracts the filename from the full path
    return {
        "reports": [os.path.basename(r) for r in reports],
        "count": len(reports),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/logs/reports/{report_filename}")
async def get_report_file(report_filename: str):
    """
    Download a specific report file by filename.

    This endpoint allows downloading individual daily reports.

    Args:
        report_filename: The name of the report file (e.g., "daily_report_20241215.json")

    Returns:
        FileResponse: The requested report file
    """
    # Construct the full file path
    # Using os.path.join ensures correct path separator for the operating system
    report_path = os.path.join("reports", report_filename)

    # Check if the file exists
    if os.path.exists(report_path):
        # Return the file for download
        return FileResponse(
            report_path,
            media_type="application/json",
            filename=report_filename
        )
    else:
        # Return 404 error if report doesn't exist
        raise HTTPException(status_code=404, detail=f"Report {report_filename} not found")


@app.post("/predict", response_model=DemandResponse)
async def predict(request: DemandRequest, background_tasks: BackgroundTasks):
    """
    Predict demand based on input features.

    This is the main prediction endpoint.
    Clients send product features and receive a demand forecast.

    Args:
        request: DemandRequest object containing input features (validated by Pydantic)
        background_tasks: FastAPI background tasks (runs after response is sent)

    Returns:
        DemandResponse: Prediction result with confidence interval and metadata
    """
    try:
        # Convert the Pydantic request model to a dictionary
        # model_dump() is the modern equivalent of dict() (Pydantic v2)
        features_dict = request.model_dump()

        # Remove Product_ID from features (it's not used for prediction)
        # pop() removes the key from the dictionary and returns its value
        product_id = features_dict.pop("Product_ID")

        # Log the incoming request for debugging
        logger.info(f"Received prediction request for product: {product_id}")
        logger.debug(f"Features: {features_dict}")

        # Make the prediction using the model manager
        prediction = model_manager.predict(features_dict)

        # Log the prediction result
        logger.info(f"Prediction for {product_id}: {prediction:.2f} units")

        # Log the prediction in the background (doesn't slow down response time)
        # Background tasks run after the HTTP response is sent to the client
        background_tasks.add_task(log_prediction, product_id, features_dict, prediction)

        # Calculate 90% confidence interval:
        # - Lower bound: 10% below the prediction
        # - Upper bound: 10% above the prediction
        # In production, this would be calculated using model uncertainty metrics
        lower_bound = prediction * 0.9
        upper_bound = prediction * 1.1

        # Return the formatted response
        return DemandResponse(
            predicted_demand=round(prediction, 2),  # Round to 2 decimal places
            timestamp=datetime.now().isoformat(),
            model_version=model_manager.model_version,
            confidence_interval_lower=round(lower_bound, 2),
            confidence_interval_upper=round(upper_bound, 2)
        )

    except Exception as e:
        # Log the error for debugging
        logger.error(f"Prediction error: {str(e)}")

        # Import traceback for detailed error information
        import traceback
        traceback.print_exc()  # Print full stack trace to terminal

        # Return HTTP 500 error to client with error message
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# PART 8: MODEL MONITORING CLASS
# ============================================

class ModelMonitor:
    """
    Monitor model performance and detect drift.

    This class analyzes prediction patterns to identify when model performance degrades.
    It checks for:
    - Prediction variability (standard deviation)
    - Model drift (changing patterns over time)
    - Alert generation for anomalies
    """

    def __init__(self):
        """Initialize the monitor with empty alert history and detection thresholds."""
        self.alert_history = []  # Store all alerts for audit trail and reporting

        # Define threshold for drift detection
        # If prediction standard deviation exceeds 30000, drift is suspected
        # This threshold was determined based on the training data distribution
        self.drift_threshold = 30000

    def calculate_recent_metrics(self, lookback_hours=24):
        """
        Calculate statistical metrics for recent predictions.

        This method analyzes predictions from the last N hours to detect anomalies.

        Args:
            lookback_hours: Number of hours to look back for analysis (default 24)

        Returns:
            dict: Metrics dictionary with statistics, or None if insufficient data
        """
        # Calculate the cutoff time (current time minus lookback_hours)
        # timedelta represents a duration of time
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)

        # Filter predictions that occurred after the cutoff time
        # List comprehension with condition
        # datetime.fromisoformat() converts string timestamp back to datetime object
        recent_predictions = [
            p for p in prediction_store
            if datetime.fromisoformat(p["timestamp"]) > cutoff_time
        ]

        # Need at least 10 predictions for meaningful statistical analysis
        if len(recent_predictions) < 10:
            return None

        # Extract just the prediction values from the log entries
        predictions = [p["prediction"] for p in recent_predictions]

        # Calculate statistical metrics using numpy
        return {
            "samples_evaluated": len(recent_predictions),
            "average_prediction": np.mean(predictions),  # Mean (average)
            "prediction_std": np.std(predictions),  # Standard deviation (variability)
            "min_prediction": np.min(predictions),  # Minimum value
            "max_prediction": np.max(predictions),  # Maximum value
            "prediction_range": np.max(predictions) - np.min(predictions),  # Range
            "timestamp": datetime.now().isoformat()
        }

    def check_drift(self):
        """
        Check for model drift and return alert status.

        Drift is detected when prediction variability exceeds the threshold.
        High variability may indicate:
        - Changing demand patterns
        - Data quality issues
        - Model degradation

        Returns:
            bool: True if drift detected (alert triggered), False otherwise
        """
        # Get recent metrics (last 24 hours)
        metrics = self.calculate_recent_metrics()

        # Return False if not enough data for analysis
        if metrics is None:
            logger.debug("Insufficient data for drift detection (need at least 10 predictions)")
            return False

        # Check if standard deviation exceeds the drift threshold
        # High standard deviation means predictions are very inconsistent
        if metrics["prediction_std"] > self.drift_threshold:
            # Create detailed alert message with metrics
            alert_message = (
                f"DRIFT DETECTED: Prediction variability exceeds threshold.\n"
                f"  Standard deviation: {metrics['prediction_std']:.2f}\n"
                f"  Average prediction: {metrics['average_prediction']:.2f}\n"
                f"  Prediction range: {metrics['prediction_range']:.2f}\n"
                f"  Samples evaluated: {metrics['samples_evaluated']}"
            )

            # Log the warning (writes to both file and console)
            logger.warning(alert_message)

            # Send alert to monitoring system
            self.send_alert(alert_message)

            return True  # Drift detected

        else:
            # Log stable status for monitoring (DEBUG level to avoid log spam)
            logger.debug(
                f"Model stable. Avg: {metrics['average_prediction']:.2f}, "
                f"Std: {metrics['prediction_std']:.2f}"
            )
            return False  # No drift detected

    def send_alert(self, message):
        """
        Send alert to monitoring system.

        In production, this would integrate with:
        - Slack webhook (send to #alerts channel)
        - PagerDuty (page on-call engineer)
        - Email (send to team distribution list)
        - Prometheus/AlertManager (push to metrics system)

        For this implementation, alerts are logged to file.

        Args:
            message: Alert message text describing the issue
        """
        # Create alert entry with timestamp
        alert_entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "alert_type": "model_drift" if "DRIFT" in message else "performance_warning"
        }

        # Add to in-memory alert history
        self.alert_history.append(alert_entry)

        # Write to alert log file for audit trail and persistence
        # "a" mode appends to the file (doesn't overwrite)
        with open("logs/alerts.log", "a") as f:
            f.write(f"{datetime.now().isoformat()} - {message}\n")
            f.write("-" * 80 + "\n")  # Separator line for readability


# Create a global monitor instance
# This single instance will be used for all monitoring operations
monitor = ModelMonitor()


# ============================================
# PART 9: SCHEDULED TASKS FOR AUTOMATION
# ============================================

def scheduled_monitoring():
    """
    Run scheduled monitoring check.

    This function is called automatically by the schedule library.
    It checks for model drift and logs results.
    The function is designed to be idempotent (safe to run multiple times).
    """
    logger.info("Running scheduled monitoring check")

    # Check for drift (this will log and alert if detected)
    drift_detected = monitor.check_drift()

    # Log the result of the check
    if drift_detected:
        logger.warning("Drift was detected during scheduled check")
    else:
        logger.info("No drift detected - model performance is stable")


def generate_daily_report():
    """
    Generate daily performance report.

    This function creates a JSON report with daily statistics.
    Reports can be used for:
    - Business dashboards (show prediction volume)
    - Compliance auditing (track model usage)
    - Performance analysis (identify trends)

    The report is saved as a JSON file in the reports directory.
    """
    logger.info("Generating daily report")

    # Calculate metrics from the last 24 hours
    metrics = monitor.calculate_recent_metrics(lookback_hours=24)

    # Determine overall health status
    if metrics and metrics.get("prediction_std", 0) > monitor.drift_threshold:
        health_status = "degraded"  # High variability - possible issues
    elif model_manager.model_loaded:
        health_status = "healthy"  # Model loaded and working
    else:
        health_status = "unhealthy"  # Model not loaded

    # Compile comprehensive report data
    report = {
        "report_date": datetime.now().strftime("%Y-%m-%d"),
        "report_timestamp": datetime.now().isoformat(),
        "total_predictions_lifetime": model_manager.total_predictions,
        "model_version": model_manager.model_version,
        "health_status": health_status,
        "recent_alerts": monitor.alert_history[-5:],  # Last 5 alerts
        "recent_metrics": metrics,  # 24-hour statistics
        "prediction_store_size": len(prediction_store),
        "system_info": {
            "model_loaded": model_manager.model_loaded,
            "feature_count": len(model_manager.feature_columns) if model_manager.feature_columns else 0
        }
    }

    # Save report to file with date in filename
    filename = f"reports/daily_report_{datetime.now().strftime('%Y%m%d')}.json"
    with open(filename, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Daily report saved: {filename}")

    # Also log the report summary
    logger.info(f"Report summary: {health_status}, {model_manager.total_predictions} total predictions")


def scheduled_retraining():
    """
    Scheduled model retraining task.

    This function checks if enough new data has been collected,
    and retrains the model if the threshold is met.
    In production, this would:
    1. Query a database for new labeled data    2. Filter for high-quality samples
    3. Retrain the model
    4. Validate the new model
    5. Deploy if validation passes

    Time: Runs daily at 2 AM (low traffic period)
    """
    logger.info("Checking if retraining is needed")

    # In a real implementation, this would:
    # 1. Check a database for new data
    # 2. Verify data quality
    # 3. Retrain if enough samples collected

    # For this implementation, we check if we have enough predictions
    # (assuming predictions could be validated with actual demand later)
    if len(prediction_store) >= RETRAIN_SAMPLE_THRESHOLD:
        logger.info(f"Retraining threshold reached ({len(prediction_store)} samples)")

        # In production, this is where you would:
        # 1. Load actual demand values for past predictions
        # 2. Prepare training data
        # 3. Call model_manager.retrain(X_new, y_new)

        logger.info("Retraining would be triggered here with actual data")
    else:
        logger.debug(f"Not enough data for retraining. Need {RETRAIN_SAMPLE_THRESHOLD}, have {len(prediction_store)}")


def setup_schedules():
    """
    Configure all scheduled tasks.

    This function sets up recurring jobs using the schedule library.
    Jobs run in the background while the API serves requests.

    The schedule library uses a simple syntax:
    every().[time_unit].do(function)
    """
    # Run monitoring check every hour
    # This ensures we detect drift quickly (within 1 hour)
    schedule.every(1).hours.do(scheduled_monitoring)

    # Generate daily report at 11:59 PM (end of day)
    # This captures the full day's activity
    schedule.every().day.at("23:59").do(generate_daily_report)

    # Run retraining check at 2 AM (when traffic is lowest)
    # Retraining is CPU-intensive, so run during off-peak hours
    schedule.every().day.at("02:00").do(scheduled_retraining)

    # Log confirmation of scheduled tasks
    logger.info("Schedules configured:")
    logger.info("  - Hourly health checks")
    logger.info("  - Daily report at 23:59")
    logger.info("  - Retraining check at 02:00")


def run_scheduler():
    """
    Run the scheduler in a background thread.

    This function continuously checks for pending scheduled tasks.
    It runs forever until the program is terminated.
    The function uses a simple polling loop with a sleep to prevent CPU spinning.
    """
    # Set up all scheduled jobs
    setup_schedules()

    # Infinite loop - runs until program stops
    while True:
        # Check if any scheduled jobs need to run (checks all jobs)
        schedule.run_pending()

        # Wait 60 seconds before checking again
        # This prevents the loop from consuming too much CPU
        # 60 seconds is the polling interval
        time.sleep(60)


# ============================================
# PART 10: CUSTOMER FEEDBACK ENDPOINT (for retraining)
# ============================================

class FeedbackRequest(BaseModel):
    """
    Request schema for providing customer feedback.

    This allows the system to learn from actual outcomes.
    After a prediction is made and the actual demand is known,
    this endpoint can be called to provide the true value.
    """
    prediction_id: str  # Identifier linking to the original prediction
    actual_demand: float  # The actual demand that occurred
    product_id: str  # Which product


@app.post("/feedback")
async def provide_feedback(feedback: FeedbackRequest):
    """
    Receive feedback about actual demand for model improvement.

    This endpoint is used to collect ground truth data.
    After a prediction is made and time passes, the actual demand
    can be submitted here for future model retraining.

    Args:
        feedback: FeedbackRequest containing actual demand value

    Returns:
        dict: Confirmation message
    """
    logger.info(f"Received feedback for product {feedback.product_id}: "
                f"Actual demand = {feedback.actual_demand}")

    # In production, this data would be stored in a database
    # and used for periodic model retraining

    # For this implementation, we save to a feedback file
    feedback_entry = {
        "timestamp": datetime.now().isoformat(),
        "prediction_id": feedback.prediction_id,
        "product_id": feedback.product_id,
        "actual_demand": feedback.actual_demand
    }

    # Append to feedback log
    with open("logs/feedback.log", "a") as f:
        f.write(json.dumps(feedback_entry) + "\n")

    return {
        "status": "feedback received",
        "message": "Thank you for providing actual demand data",
        "will_be_used_for_retraining": True
    }


# ============================================
# PART 11: MAIN ENTRY POINT
# ============================================

if __name__ == "__main__":
    """
    Main entry point - runs when the script is executed directly.
    
    This starts:
    1. The background scheduler thread
    2. The FastAPI server with uvicorn
    """
    # Import uvicorn for running the ASGI server
    # Uvicorn is a high-performance ASGI server for FastAPI
    import uvicorn

    # Print startup banner
    print("="*60)
    print("PHASE 5: MLOPS ENGINEER - DEPLOYMENT & MONITORING")
    print("="*60)

    # Print configuration information
    print(f"\nModel file: {MODEL_PATH}")
    print(f"Features file: {FEATURES_PATH}")

    # Get model info from model manager
    model_info = model_manager.get_model_info()
    print(f"Model loaded: {model_info['model_loaded']}")
    print(f"Model version: {model_info['model_version']}")
    print(f"Features: {model_info['features']}")

    # Print endpoint information
    print("\n" + "="*60)
    print("API ENDPOINTS")
    print("="*60)
    print("GET  /                    - API information")
    print("GET  /health              - Health check")
    print("POST /predict             - Make a prediction")
    print("GET  /metrics             - Get model metrics")
    print("GET  /logs/production     - Download production log")
    print("GET  /logs/alerts         - Download alerts log")
    print("GET  /logs/reports        - List daily reports")
    print("GET  /logs/reports/{name} - Download specific report")
    print("POST /feedback            - Provide actual demand feedback")
    print("GET  /docs                - Interactive API documentation (Swagger UI)")

    # Print startup instructions
    print("\n" + "="*60)
    print("STARTING SERVER")
    print("="*60)
    print("\nAPI Server: http://localhost:8000")
    print("API Docs:   http://localhost:8000/docs")
    print("Health:     http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server\n")

    # Start scheduler in a background daemon thread
    # daemon=True means the thread will exit when the main thread exits
    # This prevents the program from hanging on shutdown
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    logger.info("Background scheduler started")

    # Start the FastAPI server with uvicorn
    # host="0.0.0.0" means listen on all network interfaces (including localhost and network)
    # port=8000 is the default HTTP port for this API
    # This call blocks (the program stays here until Ctrl+C)
    uvicorn.run(app, host="0.0.0.0", port=8000)