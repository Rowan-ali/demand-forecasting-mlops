"""
dashboard.py - Streamlit Frontend for Demand Forecasting API

This module provides a web-based user interface for the demand forecasting system.
Users can:
- Enter product features through a form
- Get demand predictions
- View prediction history and analytics
- Monitor API status
- Access system logs and reports

Run with: streamlit run dashboard.py

Author: MLOps Engineer
Version: 1.0.0
"""

# ============================================
# IMPORT STATEMENTS
# ============================================

# Import streamlit for creating the web interface
# Streamlit is a framework that turns Python scripts into web apps
import streamlit as st

# Import requests for calling the FastAPI backend
# This allows the frontend to communicate with the API
import requests

# Import pandas for data manipulation and creating dataframes
# Used for displaying history tables and analytics
import pandas as pd

# Import datetime for timestamps in predictions and history
from datetime import datetime

# Import plotly for creating interactive charts
# plotly.express provides high-level chart creation
import plotly.express as px

# Import plotly.graph_objects for more advanced chart customization
import plotly.graph_objects as go

# Import json for parsing and displaying JSON data
import json

# Import glob for finding files matching patterns (used for finding reports)
import glob

# Import os for file system operations (checking if files exist)
import os

# Import numpy for numerical operations (calculating statistics)
import numpy as np


# ============================================
# PAGE CONFIGURATION (Must be first Streamlit command)
# ============================================

# Configure the web page settings
# This must be the first Streamlit command in the script
st.set_page_config(
    page_title="Demand Forecasting System",  # Title shown in browser tab
    page_icon="",  # Icon shown in browser tab (empty for default)
    layout="wide",  # Use wide layout (full width of browser)
    initial_sidebar_state="expanded"  # Sidebar starts expanded (open)
)


# ============================================
# HELPER FUNCTIONS
# ============================================

def convert_arabic_to_english(text):
    """
    Convert Arabic (Eastern) numerals to English (Western) numerals.

    This function handles the localization issue where numbers appear as
    Arabic digits on systems with Arabic locale settings.

    Arabic numerals: ٠ ١ ٢ ٣ ٤ ٥ ٦ ٧ ٨ ٩
    English numerals: 0 1 2 3 4 5 6 7 8 9

    Args:
        text: String that may contain Arabic numerals

    Returns:
        String with Arabic numerals replaced by English numerals
    """
    # Define mapping from Arabic to English numerals
    # The keys are Arabic Unicode characters, values are English digits
    arabic_to_english = {
        '٠': '0',  # Arabic zero
        '١': '1',  # Arabic one
        '٢': '2',  # Arabic two
        '٣': '3',  # Arabic three
        '٤': '4',  # Arabic four
        '٥': '5',  # Arabic five
        '٦': '6',  # Arabic six
        '٧': '7',  # Arabic seven
        '٨': '8',  # Arabic eight
        '٩': '9'   # Arabic nine
    }

    # Replace each Arabic numeral with its English equivalent
    for arabic, english in arabic_to_english.items():
        text = text.replace(arabic, english)

    return text


def safe_format_number(value, format_string="{:.2f}"):
    """
    Safely format a number, handling potential Arabic numeral conversion.

    Args:
        value: The number to format
        format_string: The format specifier (default 2 decimal places)

    Returns:
        Formatted string with English numerals
    """
    # Format the number using the format string
    formatted = format_string.format(value)

    # Convert any Arabic numerals to English
    formatted = convert_arabic_to_english(formatted)

    return formatted


def check_api_health():
    """
    Check if the backend API is running and healthy.

    This function is called to verify connectivity to the FastAPI backend.

    Returns:
        tuple: (is_healthy, health_data) where is_healthy is boolean
               and health_data contains the API response
    """
    try:
        # Send GET request to health endpoint
        # timeout=5 means wait max 5 seconds for response
        response = requests.get("http://localhost:8000/health", timeout=5)

        # Check if response status is 200 OK
        if response.status_code == 200:
            # Parse JSON response
            health_data = response.json()
            return True, health_data
        else:
            # Non-200 response means API is unhealthy
            return False, {"status": f"Error {response.status_code}"}

    except requests.exceptions.ConnectionError:
        # Connection refused - API server is not running
        return False, {"status": "Cannot connect - Server not running"}

    except requests.exceptions.Timeout:
        # Request timed out - API is slow or unresponsive
        return False, {"status": "Timeout - Server not responding"}


def get_prediction(product_id, price, discount, stock, marketing, seasonal, holiday):
    """
    Send a prediction request to the backend API.

    This function constructs the JSON payload and makes the POST request.

    Args:
        product_id: Product identifier string
        price: Product price in USD
        discount: Discount percentage
        stock: Stock availability count
        marketing: Marketing effect score
        seasonal: Seasonal effect multiplier
        holiday: 1 for public holiday, 0 otherwise

    Returns:
        tuple: (success, result_or_error)
    """
    # Construct the payload matching the API's expected schema
    payload = {
        "Product_ID": product_id,
        "Price": price,
        "Discount": discount,
        "Stock_Availability": stock,
        "Marketing_Effect": marketing,
        "Seasonal_Effect": seasonal,
        "Public_Holiday": holiday
    }

    try:
        # Send POST request to prediction endpoint
        # json=payload automatically sets Content-Type: application/json
        response = requests.post(
            "http://localhost:8000/predict",
            json=payload,
            timeout=10  # 10 second timeout for prediction
        )

        # Check if request was successful
        if response.status_code == 200:
            # Parse and return the prediction result
            result = response.json()
            return True, result
        else:
            # Return error message from API
            return False, f"API Error: {response.status_code}"

    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to API. Make sure the server is running."

    except requests.exceptions.Timeout:
        return False, "Request timed out. API may be overloaded."

    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


# ============================================
# SIDEBAR NAVIGATION
# ============================================

# Add title to the sidebar
st.sidebar.title("Navigation")

# Create radio button for page selection
# This creates a menu where only one option can be selected
# The selected value is stored in the 'page' variable
page = st.sidebar.radio(
    "Go to",  # Label for the radio group
    [       # List of options (pages)
        "Prediction",      # Main prediction form
        "Analytics",       # Charts and statistics
        "Monitoring",      # Logs and alerts viewer
        "API Status",      # Backend health and endpoints
        "Documentation"    # User guide and API docs
    ]
)

# Add a separator line in the sidebar
st.sidebar.markdown("---")

# Add version information to the sidebar footer
st.sidebar.caption("Demand Forecasting System v1.0")
st.sidebar.caption("Phase 5 - MLOps Deployment")
st.sidebar.caption("Powered by Gradient Boosting")


# ============================================
# PAGE 1: PREDICTION (Main Page)
# ============================================

if page == "Prediction":
    """
    Main prediction page - where users enter features and get predictions.
    """

    # Display page title
    st.title("Demand Forecasting")

    # Display subtitle with description
    st.markdown("Enter product details to get demand prediction")
    st.markdown("---")  # Horizontal line separator

    # Create two equal-width columns for layout
    # This splits the page into left (input) and right (output) sections
    col1, col2 = st.columns([1, 1])  # 1:1 ratio

    # ===== LEFT COLUMN: INPUT FORM =====
    with col1:
        st.subheader("Product Information")
        st.markdown("Enter the following product details:")

        # Text input for Product ID
        # help parameter adds a tooltip
        product_id = st.text_input(
            "Product ID",
            value="P001",
            help="Example: P001, P002, P003, P004, P005"
        )

        # Number input for Price
        # min_value=0 means negative prices not allowed
        # step=5.0 means up/down buttons change by 5
        price = st.number_input(
            "Price (USD)",
            min_value=0.0,
            max_value=500.0,
            value=45.50,
            step=5.0,
            format="%.2f"  # Always show 2 decimal places
        )

        # Slider for Discount (more intuitive than number input)
        # Sliders are good for bounded ranges like percentages
        discount = st.slider(
            "Discount (%)",
            min_value=0.0,
            max_value=50.0,
            value=10.0,
            step=1.0,
            help="Discount percentage applied to the product"
        )

        # Number input for Stock Availability
        stock_availability = st.number_input(
            "Stock Availability",
            min_value=0,
            max_value=2000,
            value=500,
            step=50,
            help="Current inventory count (units available)"
        )

        # Slider for Marketing Effect
        marketing_effect = st.slider(
            "Marketing Effect",
            min_value=0.0,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Marketing campaign effectiveness score (higher = more effective)"
        )

        # Slider for Seasonal Effect
        seasonal_effect = st.slider(
            "Seasonal Effect",
            min_value=0.5,
            max_value=2.0,
            value=1.2,
            step=0.1,
            help="Seasonal demand multiplier (>1 means peak season)"
        )

        # Dropdown for Public Holiday
        # format_func converts the stored value (0/1) to display text
        public_holiday = st.selectbox(
            "Public Holiday",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Is this a public holiday?"
        )

        # Add a separator before the prediction button
        st.markdown("---")

        # Prediction button
        # use_container_width makes the button span the column width
        # type="primary" makes it stand out (blue color)
        predict_button = st.button(
            "Predict Demand",
            use_container_width=True,
            type="primary"
        )

    # ===== RIGHT COLUMN: RESULTS =====
    with col2:
        st.subheader("Prediction Result")

        # Check if the predict button was clicked
        if predict_button:
            # Show a loading spinner while waiting for API response
            with st.spinner("Calling prediction API..."):
                # Make the API call
                success, result = get_prediction(
                    product_id, price, discount, stock_availability,
                    marketing_effect, seasonal_effect, public_holiday
                )

            # Handle successful prediction
            if success:
                # Extract prediction from result
                prediction = result['predicted_demand']

                # Create a styled card using HTML
                # unsafe_allow_html=True allows HTML rendering
                st.markdown(
                    f"""
                    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
                        <h3 style="text-align: center; color: #0066cc;">Predicted Demand</h3>
                        <h1 style="text-align: center; color: #0066cc;">{safe_format_number(prediction, '{:,.0f}')}</h1>
                        <p style="text-align: center;">units</p>
                        <hr>
                        <p><strong>Confidence Interval (90%):</strong></p>
                        <p>{safe_format_number(result['confidence_interval_lower'], '{:,.0f}')} - {safe_format_number(result['confidence_interval_upper'], '{:,.0f}')} units</p>
                        <hr>
                        <p><strong>Model Version:</strong> {result['model_version']}</p>
                        <p><strong>Timestamp:</strong> {result['timestamp'][:19]}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Store prediction in session state for history
                # session_state persists across page reruns
                if 'history' not in st.session_state:
                    st.session_state.history = []

                # Add current prediction to history
                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "product_id": product_id,
                    "price": price,
                    "discount": discount,
                    "stock": stock_availability,
                    "marketing": marketing_effect,
                    "seasonal": seasonal_effect,
                    "holiday": "Yes" if public_holiday == 1 else "No",
                    "predicted_demand": prediction
                })

                # Show success message
                st.success("Prediction completed successfully!")

            else:
                # Show error message
                st.error(f"Prediction failed: {result}")

        else:
            # Show placeholder when no prediction has been made yet
            st.info("Click 'Predict Demand' to see results")

        # ===== INPUT SUMMARY (always visible) =====
        st.markdown("---")
        st.subheader("Input Summary")

        # Display the current input values in a formatted way
        summary_data = {
            "Product ID": product_id,
            "Price": safe_format_number(price, "${:.2f}"),
            "Discount": safe_format_number(discount, "{:.1f}%"),
            "Stock": safe_format_number(stock_availability, "{:d}"),
            "Marketing": safe_format_number(marketing_effect, "{:.2f}"),
            "Seasonal": safe_format_number(seasonal_effect, "{:.2f}"),
            "Holiday": "Yes" if public_holiday == 1 else "No"
        }

        # Display each key-value pair
        for key, value in summary_data.items():
            st.write(f"**{key}:** {value}")

    # ===== HISTORY SECTION (full width at bottom) =====
    st.markdown("---")
    st.subheader("Recent Predictions")

    # Check if there are any predictions in history
    if 'history' in st.session_state and st.session_state.history:
        # Convert history to DataFrame for better display
        history_df = pd.DataFrame(st.session_state.history)

        # Display as interactive table
        st.dataframe(history_df, use_container_width=True)

        # Add clear history button
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()  # Refresh the page
    else:
        st.info("No predictions yet. Click 'Predict Demand' to see results.")


# ============================================
# PAGE 2: ANALYTICS
# ============================================

elif page == "Analytics":
    """
    Analytics page - shows charts and statistics from prediction history.
    """

    st.title("Analytics Dashboard")
    st.markdown("View prediction trends and statistics")
    st.markdown("---")

    # Check if there is history data
    if 'history' in st.session_state and len(st.session_state.history) > 0:
        # Convert history to DataFrame
        history_df = pd.DataFrame(st.session_state.history)

        # ===== KEY METRICS ROW =====
        # Create 3 columns for metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            # Total predictions made
            st.metric("Total Predictions", len(history_df))

        with col2:
            # Average demand across all predictions
            avg_demand = history_df['predicted_demand'].mean()
            st.metric("Average Demand", safe_format_number(avg_demand, "{:,.0f}"))

        with col3:
            # Maximum demand predicted
            max_demand = history_df['predicted_demand'].max()
            st.metric("Max Demand", safe_format_number(max_demand, "{:,.0f}"))

        st.markdown("---")

        # ===== CHARTS ROW =====
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Demand by Product")
            # Calculate average demand per product
            product_demand = history_df.groupby('product_id')['predicted_demand'].mean().reset_index()

            # Create bar chart using plotly
            fig = px.bar(
                product_demand,
                x='product_id',
                y='predicted_demand',
                title="Average Demand by Product",
                color='product_id',  # Different color for each product
                labels={'predicted_demand': 'Average Demand (units)'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Demand Trend Over Time")

            # Sort by timestamp to show trend
            history_df_sorted = history_df.sort_values('timestamp')

            # Create line chart
            fig = px.line(
                history_df_sorted,
                x='timestamp',
                y='predicted_demand',
                title="Prediction Trend Over Time",
                markers=True,  # Show markers at each data point
                labels={'predicted_demand': 'Demand (units)', 'timestamp': 'Time'}
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ===== DISCOUNT IMPACT CHART =====
        st.subheader("Impact of Discount on Demand")

        # Create scatter plot showing relationship between discount and demand
        fig = px.scatter(
            history_df,
            x='discount',
            y='predicted_demand',
            title="Discount vs Predicted Demand",
            trendline="ols",  # Add linear regression trend line
            labels={'discount': 'Discount (%)', 'predicted_demand': 'Predicted Demand (units)'},
            color='product_id'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ===== FULL DATA TABLE =====
        st.subheader("Full Prediction History")
        st.dataframe(history_df, use_container_width=True)

        # ===== STATISTICAL SUMMARY =====
        st.subheader("Statistical Summary")

        # Calculate additional statistics
        stats = {
            "Mean Demand": f"{history_df['predicted_demand'].mean():,.0f}",
            "Median Demand": f"{history_df['predicted_demand'].median():,.0f}",
            "Standard Deviation": f"{history_df['predicted_demand'].std():,.0f}",
            "Min Demand": f"{history_df['predicted_demand'].min():,.0f}",
            "Max Demand": f"{history_df['predicted_demand'].max():,.0f}",
            "Total Predictions": len(history_df),
            "Unique Products": history_df['product_id'].nunique()
        }

        # Display statistics in columns
        stat_cols = st.columns(4)
        for i, (stat_name, stat_value) in enumerate(stats.items()):
            col_index = i % 4
            with stat_cols[col_index]:
                st.metric(stat_name, stat_value)

    else:
        # Show message when no data available
        st.info("No prediction data available. Go to 'Prediction' page and make some predictions first.")
        st.markdown("""
        ### How to get started:
        1. Go to the **Prediction** page using the sidebar
        2. Enter product details
        3. Click "Predict Demand"
        4. Return here to see analytics
        """)


# ============================================
# PAGE 3: MONITORING (Logs and Alerts)
# ============================================

elif page == "Monitoring":
    """
    Monitoring page - view system logs, alerts, and reports.
    """

    st.title("System Monitoring")
    st.markdown("View logs, alerts, and system health")
    st.markdown("---")

    # ===== API HEALTH STATUS =====
    st.subheader("API Health Status")

    # Check if backend API is healthy
    is_healthy, health_data = check_api_health()

    if is_healthy:
        st.success(f"API Status: {health_data['status']}")

        # Display health metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Loaded", "Yes" if health_data['model_loaded'] else "No")
        with col2:
            st.metric("Model Version", health_data['model_version'])
        with col3:
            st.metric("Total Predictions", health_data['total_predictions'])
    else:
        st.error(f"API Status: {health_data['status']}")
        st.info("Make sure the backend server is running: uvicorn mlops_deployment:app --reload")

    st.markdown("---")

    # ===== LOG VIEWER =====
    st.subheader("Log Viewer")

    # Create tabs for different log types
    log_tab1, log_tab2, log_tab3 = st.tabs(["Production Log", "Alerts Log", "Feedback Log"])

    with log_tab1:
        st.write("Production Log - API activity and system events")

        # Check if production log exists
        if os.path.exists("logs/mlops_production.log"):
            # Add button to refresh log
            if st.button("Refresh Production Log"):
                st.rerun()

            # Read and display the last 100 lines of the log
            with open("logs/mlops_production.log", "r") as f:
                lines = f.readlines()
                last_lines = lines[-100:] if len(lines) > 100 else lines
                st.text_area("Log Content", "".join(last_lines), height=400)
        else:
            st.warning("Production log file not found. Make sure the API server has been started.")

    with log_tab2:
        st.write("Alerts Log - Drift detection and performance alerts")

        if os.path.exists("logs/alerts.log"):
            if st.button("Refresh Alerts Log"):
                st.rerun()

            with open("logs/alerts.log", "r") as f:
                content = f.read()
                if content:
                    st.text_area("Alert Content", content, height=400)
                else:
                    st.info("No alerts recorded yet")
        else:
            st.info("Alerts log file not found. No alerts have been generated yet.")

    with log_tab3:
        st.write("Feedback Log - Customer feedback for model improvement")

        if os.path.exists("logs/feedback.log"):
            if st.button("Refresh Feedback Log"):
                st.rerun()

            with open("logs/feedback.log", "r") as f:
                lines = f.readlines()
                if lines:
                    # Parse JSON lines and display as table
                    feedback_data = []
                    for line in lines[-50:]:  # Last 50 entries
                        try:
                            feedback_data.append(json.loads(line.strip()))
                        except:
                            pass

                    if feedback_data:
                        feedback_df = pd.DataFrame(feedback_data)
                        st.dataframe(feedback_df, use_container_width=True)
                    else:
                        st.text_area("Feedback Content", "".join(lines[-100:]), height=400)
                else:
                    st.info("No feedback recorded yet")
        else:
            st.info("Feedback log file not found. Use the /feedback endpoint to provide feedback.")

    st.markdown("---")

    # ===== DAILY REPORTS =====
    st.subheader("Daily Reports")

    # Find all report files in the reports directory
    if os.path.exists("reports"):
        reports = glob.glob("reports/daily_report_*.json")

        if reports:
            # Sort reports by name (which includes date) - newest first
            reports.sort(reverse=True)

            # Let user select a report
            selected_report = st.selectbox("Select Report", reports)

            if selected_report:
                with open(selected_report, "r") as f:
                    report_data = json.load(f)

                # Display report data in an expandable section
                with st.expander("Report Details", expanded=True):
                    st.json(report_data)

                # Display key metrics from the report
                st.subheader("Report Summary")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Health Status", report_data.get('health_status', 'N/A'))
                with col2:
                    st.metric("Total Predictions", report_data.get('total_predictions_lifetime', 'N/A'))
                with col3:
                    st.metric("Model Version", report_data.get('model_version', 'N/A'))
        else:
            st.info("No reports generated yet. Reports are generated daily at 23:59.")
    else:
        st.info("Reports directory not found")


# ============================================
# PAGE 4: API STATUS
# ============================================

elif page == "API Status":
    """
    API Status page - shows backend endpoints and their status.
    """

    st.title("API Status")
    st.markdown("Check the health of the backend API and view available endpoints")
    st.markdown("---")

    # ===== BACKEND CONNECTION =====
    st.subheader("Backend Connection")

    # Check API health
    is_healthy, health_data = check_api_health()

    if is_healthy:
        st.success("API Server is running")

        # Display health information
        st.write(f"**Status:** {health_data['status']}")
        st.write(f"**Model Loaded:** {health_data['model_loaded']}")
        st.write(f"**Model Version:** {health_data['model_version']}")
        st.write(f"**Total Predictions:** {health_data['total_predictions']}")

        st.markdown("---")

        # ===== AVAILABLE ENDPOINTS =====
        st.subheader("Available API Endpoints")

        # Define all available endpoints with descriptions
        endpoints = {
            "Root API": {
                "url": "http://localhost:8000/",
                "method": "GET",
                "description": "API information and available endpoints"
            },
            "Health Check": {
                "url": "http://localhost:8000/health",
                "method": "GET",
                "description": "Check if API is healthy"
            },
            "Make Prediction": {
                "url": "http://localhost:8000/predict",
                "method": "POST",
                "description": "Get demand forecast for a product"
            },
            "Get Metrics": {
                "url": "http://localhost:8000/metrics",
                "method": "GET",
                "description": "Get model performance metrics"
            },
            "Production Log": {
                "url": "http://localhost:8000/logs/production",
                "method": "GET",
                "description": "Download production log file"
            },
            "Alerts Log": {
                "url": "http://localhost:8000/logs/alerts",
                "method": "GET",
                "description": "Download alerts log file"
            },
            "List Reports": {
                "url": "http://localhost:8000/logs/reports",
                "method": "GET",
                "description": "Get list of daily reports"
            },
            "API Documentation": {
                "url": "http://localhost:8000/docs",
                "method": "GET",
                "description": "Interactive Swagger UI documentation"
            }
        }

        # Display endpoints in a table format
        endpoints_df = pd.DataFrame([
            {
                "Endpoint": name,
                "Method": info["method"],
                "URL": info["url"],
                "Description": info["description"]
            }
            for name, info in endpoints.items()
        ])

        st.dataframe(endpoints_df, use_container_width=True)

        st.markdown("---")

        # ===== QUICK TEST BUTTONS =====
        st.subheader("Quick API Tests")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Test Health Endpoint", use_container_width=True):
                try:
                    response = requests.get("http://localhost:8000/health", timeout=5)
                    st.success(f"Status Code: {response.status_code}")
                    st.json(response.json())
                except Exception as e:
                    st.error(f"Error: {e}")

        with col2:
            if st.button("Test Metrics Endpoint", use_container_width=True):
                try:
                    response = requests.get("http://localhost:8000/metrics", timeout=5)
                    st.success(f"Status Code: {response.status_code}")
                    st.json(response.json())
                except Exception as e:
                    st.error(f"Error: {e}")

    else:
        st.error("Cannot connect to API Server")
        st.markdown(f"""
        ### Troubleshooting:
        
        1. **Start the backend server:**
        ```bash
        cd D:\\Phase 5 MLOps
        uvicorn mlops_deployment:app --reload    **Feature Importance (what drives demand):**
    1. **Stock Availability** - Most important factor
    2. **Marketing Effect** - Second most important
    3. **Price** - Moderate impact
    4. **Discount** - Moderate impact
    5. **Seasonal Effect** - Lower impact
    6. **Public Holiday** - Lowest impact
    """)

    st.markdown("---")

elif page == "Documentation":
    """
    Documentation page - user guide and system documentation.
    """

    st.title("Documentation")
    st.markdown("How to use the Demand Forecasting System")
    st.markdown("---")

    # ===== ABOUT SECTION =====
    st.subheader("About")
    st.write("""
    This system forecasts product demand using a **Gradient Boosting** machine learning model.
    The model was trained on historical sales data and considers multiple factors that influence demand.

    **Key Features:**
    - Real-time demand predictions via REST API
    - Web-based user interface for easy access
    - Automatic model monitoring and drift detection
    - Daily performance reports
    - Prediction history and analytics
    """)

    st.markdown("---")

    # ===== INPUT FEATURES SECTION =====
    st.subheader("Input Features")

    features_df = pd.DataFrame({
        "Feature": ["Price", "Discount", "Stock Availability", "Marketing Effect", "Seasonal Effect", "Public Holiday"],
        "Description": [
            "Product price in US Dollars",
            "Discount percentage applied to the product",
            "Current inventory count in units",
            "Marketing campaign effectiveness score (higher = more effective)",
            "Seasonal demand multiplier (>1 means peak season)",
            "1 if the day is a public holiday, 0 otherwise"
        ],
        "Typical Range": ["$30 - $100", "0% - 30%", "50 - 1000", "0.5 - 2.5", "0.8 - 1.5", "0 or 1"],
        "Impact on Demand": [
            "Negative (higher price = lower demand)",
            "Positive (higher discount = higher demand)",
            "Positive (more stock = more sales)",
            "Positive (effective marketing increases demand)",
            "Varies by product and time of year",
            "Variable (depends on product type)"
        ]
    })

    st.dataframe(features_df, use_container_width=True)

    st.markdown("---")

    # ===== OUTPUT SECTION =====
    st.subheader("Output")
    st.write("""
    | Output Field | Description |
    |--------------|-------------|
    | **Predicted Demand** | Forecasted number of units to be sold |
    | **Confidence Interval (90%)** | Range where actual demand is likely to fall (90% probability) |
    | **Model Version** | Which model version made the prediction |
    | **Timestamp** | When the prediction was generated |
    """)

    st.markdown("---")

    # ===== API ENDPOINTS SECTION =====
    st.subheader("API Endpoints")

    st.code("""
    GET  /                           - API information
    GET  /health                     - Health check endpoint
    POST /predict                    - Make a demand prediction
    GET  /metrics                    - Get model performance metrics
    GET  /logs/production            - Download production log
    GET  /logs/alerts                - Download alerts log
    GET  /logs/reports               - List daily reports
    GET  /logs/reports/{filename}    - Download specific report
    POST /feedback                   - Provide actual demand feedback
    GET  /docs                       - Interactive API documentation
    """, language="bash")

    st.markdown("---")

    # ===== EXAMPLE API CALL SECTION =====
    st.subheader("Example API Call")

    st.write("**Using curl (Command Line):**")
    st.code("""
    curl -X POST "http://localhost:8000/predict" \\
      -H "Content-Type: application/json" \\
      -d '{
        "Product_ID": "P001",
        "Price": 45.50,
        "Discount": 10.0,
        "Stock_Availability": 500,
        "Marketing_Effect": 1.5,
        "Seasonal_Effect": 1.2,
        "Public_Holiday": 0
      }'
    """, language="bash")

    st.write("**Using Python (requests library):**")
    st.code("""
    import requests

    payload = {
        "Product_ID": "P001",
        "Price": 45.50,
        "Discount": 10.0,
        "Stock_Availability": 500,
        "Marketing_Effect": 1.5,
        "Seasonal_Effect": 1.2,
        "Public_Holiday": 0
    }

    response = requests.post("http://localhost:8000/predict", json=payload)
    prediction = response.json()
    print(f"Predicted Demand: {prediction['predicted_demand']} units")
    """, language="python")

    st.markdown("---")

    # ===== MODEL PERFORMANCE SECTION =====
    st.subheader("Model Performance")

    st.write("""
    The Gradient Boosting model was evaluated on historical data with the following metrics:

    | Metric | Value | Interpretation |
    |--------|-------|----------------|
    | **R-squared (R²)** | 0.7208 | Model explains 72% of demand variance |
    | **RMSE** | 18,672 units | Average prediction error magnitude |
    | **MAE** | 14,261 units | Average absolute error |
    | **MAPE** | 38.66% | Average percentage error |

    **Feature Importance (what drives demand):**
    1. **Stock Availability** - Most important factor
    2. **Marketing Effect** - Second most important
    3. **Price** - Moderate impact
    4. **Discount** - Moderate impact
    5. **Seasonal Effect** - Lower impact
    6. **Public Holiday** - Lowest impact
    """)

    st.markdown("---")