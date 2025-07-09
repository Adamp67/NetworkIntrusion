# streamlit_app.py

import streamlit as st  # Streamlit for web app
import pandas as pd  # Data manipulation
import numpy as np  # Numerical operations
import joblib  # Load serialized models
import pathlib  # For file paths
import time  # Sleep for simulation
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns  # Advanced plotting
from sklearn.preprocessing import StandardScaler  # Feature scaling

# ---------------------------------------------
# App Configuration
# ---------------------------------------------
st.set_page_config(page_title="Network Intrusion Detection", layout="wide")  # Set Streamlit layout
st.title("Network Intrusion Detection System (NIDS)")  # Page title
st.markdown("""  # Intro text
This web app simulates a real-time intrusion detection system powered by machine learning.  
Models were trained using cybersecurity datasets to distinguish normal vs. malicious network traffic.
""")

# ---------------------------------------------
# Constants
# ---------------------------------------------
MODEL_DIR = pathlib.Path(__file__).parent / "Models"  # Directory where models and assets are stored
DATA_SAMPLE_SIZE = 1000  # Max records for live simulation
SLEEP_TIME = 0.4  # Delay between each live iteration

# ---------------------------------------------
# Load models and objects
# ---------------------------------------------
@st.cache_resource  # Cache for better performance
def load_model_and_scaler(name):
    try:
        model_path = MODEL_DIR / f"{name.lower().replace(' ', '_')}_model.pkl" #model file path
        scaler_path = MODEL_DIR / "scaler.pkl"           #scaler file path
        feature_path = MODEL_DIR / "training_features.pkl" # feature list path
        model = joblib.load(model_path)  # Load model
        scaler = joblib.load(scaler_path)  # Load scaler
        features = joblib.load(feature_path)  # Load column template
        return model, scaler, features  # Return all
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")  # Show error message
        return None, None, None  # Return empty on failure

# ---------------------------------------------
# Preprocessing function
# ---------------------------------------------
def preprocess_input(df, scaler, training_cols):
    df = df.drop(columns=['session_id', 'attack_detected'], errors='ignore')  # Drop unused columns
    cat_cols = ['protocol_type', 'encryption_used', 'browser_type']  # Define categorical cols
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)  # Ensure string type

    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)  # One-hot encode categorical columns

    for col in training_cols:  # Ensure all training columns exist
        if col not in df_encoded:
            df_encoded[col] = 0  # Add missing column with default value
    df_encoded = df_encoded[training_cols]  # Reorder to match training

    X_scaled = scaler.transform(df_encoded)  # Scale features
    return X_scaled  # Return scaled data

# ---------------------------------------------
# Sidebar Controls
# ---------------------------------------------
with st.sidebar:  # Sidebar UI
    st.header("Control Panel")  # Sidebar header
    model_name = st.selectbox("Choose Detection Model", ["Random Forest", "Logistic Regression", "KNN"])  # Model choice
    enable_simulation = st.checkbox("Enable Live Traffic Simulation (RF only)", value=True if model_name=="Random Forest" else False)  # Toggle live mode
    st.markdown("Upload network traffic CSV or use demo data.")  # Info message

# ---------------------------------------------
# Load Data
# ---------------------------------------------
st.subheader("1. Upload or Load Data")  # Section title
uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])  # File uploader
if uploaded_file:
    df = pd.read_csv(uploaded_file)  # Read uploaded file
else:
    demo_path = MODEL_DIR / "cybersecurity_intrusion_data.csv"
    if demo_path.exists():
        df = pd.read_csv(demo_path)  # Load demo data
        st.info("✅ Using demo data from Models folder.")  # Notify user
    else:
        st.error("❌ No data found. Please upload a CSV.")  # Stop if nothing available
        st.stop()

st.dataframe(df.head(), use_container_width=True)  # Show data preview
st.write(f"**Records:** {len(df):,}")  # Show number of rows

# ---------------------------------------------
# Load model and process data
# ---------------------------------------------
st.subheader("2. Load & Prepare Model")  # Section title
model, scaler, training_cols = load_model_and_scaler(model_name)  # Load model + tools
if not model:
    st.stop()  # Stop if model fails

X_scaled = preprocess_input(df.copy(), scaler, training_cols)  # Preprocess the data

# ---------------------------------------------
# Live Simulation Mode
# ---------------------------------------------
if enable_simulation and model_name == "Random Forest":  # Only allow for Random Forest
    st.subheader("3. Live Traffic Simulation")  # Section header

    placeholder = st.empty()  # Container for dynamic output
    progress = st.progress(0)  # Progress bar
    total = 0  # Initialize counter
    attacks = 0  # Initialize counter

    for i in range(min(DATA_SAMPLE_SIZE, len(df))):  # Loop through records
        sample = df.iloc[:i+1].copy()  # Select batch
        X_sample = preprocess_input(sample, scaler, training_cols)  # Preprocess
        preds = model.predict(X_sample)  # Predict class
        probs = model.predict_proba(X_sample)[:, 1]  # Get probability

        total = len(preds)  # Update total
        attacks = int(preds.sum())  # Count attacks

        with placeholder.container():  # Update UI
            col1, col2 = st.columns(2)  # Two metrics
            col1.metric("Total Connections", total)  # Show total
            col2.metric("Detected Attacks", attacks, delta=f"{(attacks/total):.1%}")  # Show attack count
            fig, ax = plt.subplots()  # Bar plot
            ax.bar(["Normal", "Attack"], [total - attacks, attacks], color=["green", "red"])  # Plot
            st.pyplot(fig)  # Display

        progress.progress(total / DATA_SAMPLE_SIZE)  # Update progress
        time.sleep(SLEEP_TIME)  # Simulate delay

    st.success("✅ Live simulation complete.")  # End message

# ---------------------------------------------
# Static Detection Button
# ---------------------------------------------
else:
    st.subheader("3. Run Batch Detection")  # Section title
    if st.button("Run Detection Now"):  # Trigger
        preds = model.predict(X_scaled)  # Predict labels
        probs = model.predict_proba(X_scaled)[:, 1]  # Predict probabilities

        st.success("✅ Detection complete.")  # Notify user
        col1, col2 = st.columns(2)  # Metric columns
        col1.metric("Total Connections", len(preds))  # Show total
        col2.metric("Detected Attacks", int(preds.sum()), delta=f"{(preds.mean()):.2%}")  # Show attacks

        if model_name == "Random Forest" and hasattr(model, "feature_importances_"):  # Show feature importance
            importances = model.feature_importances_  # Get importance
            top_features = pd.DataFrame({  # Create DataFrame
                'Feature': training_cols,
                'Importance': importances
            }).sort_values(by="Importance", ascending=False).head(10)  # Top 10 features

            st.subheader("Key Features (Random Forest)")  # Section title
            sns.barplot(data=top_features, x="Importance", y="Feature", palette="viridis")  # Plot
            st.pyplot(plt)  # Display plot

# ---------------------------------------------
# Compare Models
# ---------------------------------------------
st.subheader("4. Compare Models")  # Section title
try:
    summary_df = pd.read_csv(MODEL_DIR / "Model_Comparison_Summary.csv")  # Load summary
    metric = st.selectbox("Select metric", ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"])  # Metric selector
    sns.barplot(data=summary_df, x="Model", y=metric, palette="rocket")  # Plot bar chart
    plt.title(f"Model Comparison by {metric}")  # Add title
    st.pyplot(plt)  # Show plot
except:
    st.warning("Model comparison data not found.")  # Handle errors

# ---------------------------------------------
# Learn More
# ---------------------------------------------
with st.expander("ℹ️ Learn About NIDS"):  # Educational section
    st.markdown("""
    A **Network Intrusion Detection System (NIDS)** is a security tool used to monitor computer networks and identify unusual or suspicious activity.

    In simpler terms, it works like a security camera for your internet traffic watching for patterns that may indicate hacking attempts, malware, or unauthorized access.

    This system uses **machine learning** models trained on real-world attack data to recognize signs of cyber intrusions. It analyses many features such as:

    - The number of failed login attempts
    - The reputation of the IP address
    - How long a session lasts
    - Whether suspicious encryption is being used
    - The type of browser or protocol being used
    - And many other behavioral indicators

    Our tool allows users to upload network activity logs and instantly check if the data shows signs of malicious behavior, making cybersecurity more accessible for everyone.
    """)

st.markdown("---")  # Divider
st.markdown("📚 Developed by Adam Patel at the University of Westminster – 2025")  # Footer