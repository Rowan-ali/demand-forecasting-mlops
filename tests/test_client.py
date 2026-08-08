"""
Test Client for Demand Forecasting API

This script tests all API endpoints including:
- Health check
- Prediction requests
- Load testing
- Metrics gathering

Run with: python test_client.py
"""

# Import for making HTTP requests to the API
import requests

# Import for generating random test data
import random

# Import for timing operations
import time

# Import for JSON serialization
import json

# API base URL - points to the local running server
API_URL = "http://localhost:8000"


def test_health():
    """
    Test the health check endpoint.

    This function verifies that the API is responsive
    and returns correct health status.
    """
    # Send GET request to /health endpoint
    # timeout=5 means wait max 5 seconds for response
    response = requests.get(f"{API_URL}/health", timeout=5)

    # Parse JSON response
    data = response.json()

    # Print results
    print(f"Health check response:")
    print(f"  Status: {data['status']}")
    print(f"  Model loaded: {data['model_loaded']}")
    print(f"  Model version: {data['model_version']}")
    print(f"  Total predictions: {data['total_predictions']}")

    # Return True if health check passed
    return data['status'] == 'healthy'


def test_prediction():
    """
    Test the prediction endpoint with random data.

    This function generates random input features,
    sends a prediction request, and prints the result.

    Returns:
        dict: The prediction response data
    """
    # Generate random feature values for testing
    # Each value is chosen to match realistic ranges
    payload = {
        # Random product ID from P001 to P005
        "Product_ID": f"P00{random.randint(1, 5)}",

        # Price between 30 and 100 USD
        "Price": round(random.uniform(30, 100), 2),

        # Discount between 0 and 30 percent
        "Discount": round(random.uniform(0, 30), 2),

        # Stock availability between 50 and 1000 units
        "Stock_Availability": random.randint(50, 1000),

        # Marketing effect between 0.5 and 2.5
        "Marketing_Effect": round(random.uniform(0.5, 2.5), 2),

        # Seasonal effect between 0.8 and 1.5
        "Seasonal_Effect": round(random.uniform(0.8, 1.5), 2),

        # Public holiday: 0 or 1 (10% chance of holiday)
        "Public_Holiday": 1 if random.random() < 0.1 else 0
    }

    # Send POST request to /predict endpoint
    # json=payload automatically sets Content-Type: application/json
    response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)

    # Parse JSON response
    result = response.json()

    # Print results with product ID and prediction
    print(f"Prediction for {payload['Product_ID']}: {result['predicted_demand']:.0f} units")
    print(
        f"  Confidence interval: [{result['confidence_interval_lower']:.0f}, {result['confidence_interval_upper']:.0f}]")
    print(f"  Model version: {result['model_version']}")

    # Return the result for further use
    return result


def test_metrics():
    """
    Test the metrics endpoint.

    This function retrieves operational metrics
    from the API and prints them.
    """
    # Send GET request to /metrics endpoint
    response = requests.get(f"{API_URL}/metrics", timeout=5)

    # Parse JSON response
    data = response.json()

    # Print metrics
    print(f"Metrics:")
    print(f"  Total predictions: {data['total_predictions']}")
    print(f"  Model version: {data['model_version']}")
    print(f"  Recent predictions: {len(data['recent_predictions'])}")

    # Return the data
    return data


def run_load_test(n_requests=50):
    """
    Run a load test with multiple requests.

    This function sends many sequential requests to test
    API performance and stability.

    Args:
        n_requests: Number of requests to send (default 50)
    """
    print(f"\nRunning load test: {n_requests} requests")

    # Record start time
    start_time = time.time()

    # Send requests in a loop
    for i in range(n_requests):
        # Make a prediction
        test_prediction()

        # Print progress every 10 requests
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1} requests")

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Calculate throughput (requests per second)
    requests_per_second = n_requests / elapsed_time

    # Print load test results
    print(f"\nLoad test results:")
    print(f"  Total time: {elapsed_time:.2f} seconds")
    print(f"  Requests per second: {requests_per_second:.2f}")

    # Get final metrics
    test_metrics()


def run_continuous_test(duration_minutes=5, interval_seconds=30):
    """
    Run a continuous test for a specified duration.

    This function sends prediction requests at regular intervals
    to simulate real usage patterns.

    Args:
        duration_minutes: How long to run the test (default 5 minutes)
        interval_seconds: Time between requests (default 30 seconds)
    """
    # Calculate end time
    end_time = time.time() + (duration_minutes * 60)

    print(f"\nRunning continuous test for {duration_minutes} minutes")
    print(f"Request every {interval_seconds} seconds")

    # Counter for requests sent
    request_count = 0

    # Loop until end time is reached
    while time.time() < end_time:
        # Send a prediction request
        test_prediction()
        request_count += 1

        # Wait for the specified interval
        time.sleep(interval_seconds)

    print(f"\nContinuous test completed")
    print(f"  Total requests: {request_count}")

    # Get final metrics
    test_metrics()


# ============================================
# MAIN EXECUTION - CHOOSE TEST TYPE
# ============================================

if __name__ == "__main__":
    # Print test header
    print("=" * 60)
    print("DEMAND FORECASTING API TESTER")
    print("=" * 60)

    # First check if API is healthy
    print("\nChecking API health...")
    if test_health():
        print("API is healthy. Running tests...")
    else:
        print("API is not healthy. Please start the API server first.")
        print("Run: python mlops_deployment.py")
        exit(1)

    # Run prediction test
    print("\n" + "-" * 40)
    print("SINGLE PREDICTION TEST")
    print("-" * 40)
    test_prediction()

    # Run load test with 20 requests
    print("\n" + "-" * 40)
    run_load_test(20)

    # Print summary
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 60)