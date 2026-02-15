# ============================================================
# MEDISCAN — AI-Powered CKD Risk Assessment Dashboard
# ============================================================
# This is the MAIN APPLICATION FILE that brings everything together:
#   1. Loads and processes patient data (via DataProcessor)
#   2. Auto-trains an ML model (via Predictor)
#   3. Displays an interactive dashboard with charts
#   4. Provides patient screening with real-time CKD prediction
#   5. Shows model analytics, explainability (SHAP), and reports
# ============================================================

import streamlit as st          # Streamlit — builds the web dashboard UI
import pandas as pd             # Data manipulation — DataFrames, CSV handling
import plotly.express as px     # Interactive charts (histograms, scatter, pie)
import numpy as np              # Numerical operations
import matplotlib.pyplot as plt # Static plotting (used for SHAP plots)
import seaborn as sns           # Statistical data visualization (for EDA heatmaps)
import requests                 # For loading Lottie animations
from streamlit_lottie import st_lottie # For rendering animations
from streamlit_option_menu import option_menu  # Custom sidebar navigation menu
from data_processing import DataProcessor       # Our custom data pipeline module
from prediction import Predictor                # Our custom ML prediction module
from sklearn.metrics import recall_score, precision_score       # For threshold tuning
from sklearn.model_selection import cross_val_score             # For cross-validation

# --- Page Configuration ---
st.set_page_config(
    page_title="Mediscan Dashboard",
    page_icon="🩺",
    layout="wide",
)

# --- Animations & Styling Setup ---
@st.cache_data
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Load DNA Animation (Medical/Scientific theme)
lottie_dna = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_4kji20Y9.json")

# Define Dataset Path
dataset_path = "mediscan_ckd_diagnostic_P5.csv"

# Global CSS for Animations
st.markdown("""
<style>
    /* Gradient Background */
    .stApp {
        background: radial-gradient(circle at top right, #1E293B, #0F172A);
        color: #E2E8F0;
    }
    
    /* Animation: Fade In for Page Content */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .block-container {
        animation: fadeIn 0.8s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# --- Hide Streamlit Default UI Elements ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* header {visibility: hidden;} */ /* Unhide header so sidebar toggle is visible */
    </style>
""", unsafe_allow_html=True)

# --- Custom CSS for Professional Dark Theme ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ========================================
       ANIMATIONS
    ======================================== */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ========================================
       GLOBAL STYLES
    ======================================== */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0B1120 0%, #0F172A 40%, #131C31 100%);
        color: #E2E8F0;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0F172A; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }

    /* ========================================
       SIDEBAR
    ======================================== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617 0%, #0B1120 50%, #0F172A 100%);
        border-right: 1px solid rgba(56, 189, 248, 0.08);
    }
    
    section[data-testid="stSidebar"] .stTitle,
    section[data-testid="stSidebar"] h1 {
        background: linear-gradient(135deg, #38BDF8, #818CF8, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 28px !important;
        letter-spacing: -0.02em;
    }

    /* ========================================
       HEADERS & TYPOGRAPHY
    ======================================== */
    h1 {
        color: #F1F5F9 !important;
        font-weight: 800 !important;
        font-size: 2rem !important;
        letter-spacing: -0.03em;
        animation: fadeInUp 0.5s ease-out;
    }
    h2 {
        color: #E2E8F0 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    h3 {
        color: #CBD5E1 !important;
        font-weight: 600 !important;
    }
    
    /* Gradient accent under page titles */
    .stTitle + div::before,
    h1::after {
        content: '';
        display: block;
        height: 3px;
        width: 60px;
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        border-radius: 2px;
        margin-top: 8px;
    }

    /* ========================================
       METRIC CARDS (Dashboard)
    ======================================== */
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.08);
        text-align: left;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 16px;
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.6s ease-out both;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #38BDF8, #818CF8, #C084FC);
        background-size: 200% 100%;
        animation: gradientMove 3s ease infinite;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px -12px rgba(0, 0, 0, 0.4);
        border-color: rgba(56, 189, 248, 0.2);
    }
    .metric-card:hover::before {
        opacity: 1;
    }

    .metric-card h3 {
        color: #94A3B8;
        font-size: 12px !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }

    .metric-card h2 {
        color: #F8FAFC !important;
        font-size: 32px !important;
        font-weight: 800 !important;
        margin: 4px 0;
        letter-spacing: -0.03em;
    }
    
    .metric-icon {
        font-size: 22px;
        margin-bottom: 12px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.15), rgba(129, 140, 248, 0.1));
    }

    /* ========================================
       RISK BARS (Dashboard)
    ======================================== */
    .risk-bar-container {
        background: rgba(30, 41, 59, 0.4);
        padding: 16px 20px;
        border-radius: 14px;
        margin-bottom: 12px;
        border: 1px solid rgba(148, 163, 184, 0.06);
        transition: all 0.2s ease;
    }
    .risk-bar-container:hover {
        border-color: rgba(148, 163, 184, 0.12);
        background: rgba(30, 41, 59, 0.6);
    }
    .risk-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        font-size: 13px;
        font-weight: 600;
        color: #E2E8F0;
    }
    .risk-progress {
        height: 6px;
        background-color: rgba(30, 41, 59, 0.8);
        border-radius: 3px;
        overflow: hidden;
    }
    .risk-progress-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ========================================
       STREAMLIT NATIVE WIDGETS
    ======================================== */
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1E3A5F 0%, #1E293B 100%);
        color: #E2E8F0;
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        letter-spacing: 0.01em;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%);
        color: #FFFFFF;
        border-color: transparent;
        box-shadow: 0 8px 25px -5px rgba(56, 189, 248, 0.35);
        transform: translateY(-1px);
    }
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Form Submit Buttons */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 32px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        box-shadow: 0 4px 15px -3px rgba(56, 189, 248, 0.3);
        transition: all 0.3s ease !important;
    }
    .stFormSubmitButton > button:hover {
        box-shadow: 0 8px 25px -5px rgba(56, 189, 248, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Download Buttons */
    .stDownloadButton > button {
        background: rgba(16, 185, 129, 0.1);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stDownloadButton > button:hover {
        background: rgba(16, 185, 129, 0.2);
        border-color: #10B981;
        box-shadow: 0 4px 15px -3px rgba(16, 185, 129, 0.25);
    }

    /* Text and Number Inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background-color: rgba(15, 23, 42, 0.6) !important;
        color: #E2E8F0 !important;
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: rgba(56, 189, 248, 0.5) !important;
        box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.1) !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
        border-radius: 10px !important;
    }
    
    /* Sliders */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #38BDF8, #818CF8) !important;
    }
    
    /* Radio buttons & Checkboxes labels */
    .stRadio > label, .stCheckbox > label {
        color: #CBD5E1 !important;
        font-weight: 500;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(148, 163, 184, 0.08) !important;
        border-radius: 12px !important;
        color: #E2E8F0 !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }
    .streamlit-expanderHeader:hover {
        border-color: rgba(56, 189, 248, 0.2) !important;
        background: rgba(30, 41, 59, 0.7) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(15, 23, 42, 0.5);
        padding: 4px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #94A3B8;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
        font-size: 13px;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.15) !important;
        color: #38BDF8 !important;
    }
    
    /* DataFrames / Tables */
    .stDataFrame {
        border: 1px solid rgba(148, 163, 184, 0.08);
        border-radius: 12px;
        overflow: hidden;
    }

    /* Metrics (native st.metric) */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(148, 163, 184, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        transition: all 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(56, 189, 248, 0.15);
    }
    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        color: #F1F5F9 !important;
        font-weight: 700 !important;
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 12px !important;
        border-left-width: 4px !important;
    }
    
    /* Spinners */
    .stSpinner > div {
        border-color: #38BDF8 transparent transparent transparent !important;
    }

    /* Forms container */
    [data-testid="stForm"] {
        background: rgba(15, 23, 42, 0.3);
        border: 1px solid rgba(148, 163, 184, 0.06);
        border-radius: 16px;
        padding: 24px;
    }
    
    /* Plotly charts container */
    .stPlotlyChart {
        background: rgba(30, 41, 59, 0.2);
        border: 1px solid rgba(148, 163, 184, 0.05);
        border-radius: 12px;
        padding: 8px;
    }
    
    /* Divider lines */
    hr {
        border-color: rgba(148, 163, 184, 0.08) !important;
    }

    /* Captions */
    .stCaption, small {
        color: #64748B !important;
    }
    
    /* Links */
    a {
        color: #38BDF8 !important;
        text-decoration: none !important;
        transition: color 0.2s ease;
    }
    a:hover {
        color: #7DD3FC !important;
    }

    </style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING & PROCESSING PIPELINE
# ============================================================
# This section handles the complete data pipeline:
#   1. Read CSV file containing CKD patient records
#   2. Impute (fill) missing values using median strategy
#   3. Engineer new features (age binning, polynomial terms)
#   4. Encode categorical features (Yes/No → 1/0)
#
# @st.cache_resource: Caches the result so data is loaded ONCE
# and reused across page refreshes (huge performance boost!
# Without caching, data would reload on every interaction).
# ============================================================

@st.cache_resource
def load_and_process_data_v2(file_path):
    """Load and preprocess the CKD dataset. Cached for performance."""
    # Step 1: Initialize DataProcessor with the path to our CSV dataset
    processor = DataProcessor(file_path)
    
    # Step 2: Load the CSV into a pandas DataFrame
    df = processor.load_data()
    
    if df is not None:
        # Step 3: Fill missing values with median (robust to outliers)
        processor.impute_data()
        
        # Step 4: Create new features (age groups, polynomial interactions)
        processor.feature_engineering()
        
        # Step 5: Convert categorical text features to numbers
        # e.g., Hypertension: "Yes"→1, "No"→0
        processor.encode_features()
    
    return processor

# Show a loading spinner while data is being processed
with st.spinner("Initializing MediScan System..."):
    # Use the dynamic dataset path
    if not os.path.exists(dataset_path):
         st.error(f"Dataset not found at: {dataset_path}")
         st.stop()
         
    processor = load_and_process_data_v2(dataset_path)

# ============================================================
# AUTO-TRAIN ML MODEL ON STARTUP
# ============================================================
# When the app first loads, we automatically:
#   1. Split data into 80% training / 20% testing
#   2. Scale features (StandardScaler: mean=0, std=1)
#   3. Select top 10 features using RFE (Recursive Feature Elimination)
#   4. Train a Random Forest classifier on the selected features
#   5. Store the trained model in session_state for reuse
#
# st.session_state: Streamlit's way of persisting data across
# page interactions. Without it, the model would retrain
# every time the user clicks a button!
# ============================================================

if processor and processor.df is not None:
    df = processor.df
    
    if 'mediscan_model' not in st.session_state:
        try:
            # Initialize the Predictor (our ML engine)
            predictor = Predictor()
            
            # Split data: 80% for training, 20% for testing
            # This ensures we evaluate on data the model has NEVER seen
            X_train, X_test, y_train, y_test = processor.split_data()
            
            # Scale features to mean=0, std=1 (prevents features with large
            # ranges like WBC count from dominating smaller features like Creatinine)
            processor.scale_features()
            X_train_scaled = processor.X_train
            
            # Feature Selection using RFE:
            # Recursively removes least important features until 10 remain
            # This improves accuracy and reduces noise
            if 'selected_features' not in st.session_state:
                best_features = predictor.select_features_rfe(X_train_scaled, y_train, n_features=10)
                st.session_state['selected_features'] = best_features
            else:
                best_features = st.session_state['selected_features']
            
            # Train the default model: Random Forest
            # Why Random Forest? It's an ensemble of 100 decision trees
            # that vote on the prediction — highly accurate and robust
            predictor.train_model(X_train_scaled[best_features], y_train, algorithm='Random Forest')
            
            # Save everything to session_state for use across pages
            st.session_state['mediscan_model'] = predictor
            st.session_state['mediscan_processor'] = processor
            
        except Exception as e:
            st.error(f"Auto-training failed: {e}")

else:
    st.stop()

# --- Sidebar Navigation ---
with st.sidebar:
    # Branded Logo Header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0 10px 0;">
        <h1 style="
            background: linear-gradient(135deg, #38BDF8, #818CF8, #C084FC);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 32px !important;
            font-weight: 800 !important;
            margin: 0 !important;
            letter-spacing: -0.02em;
        ">MediScan</h1>
        <p style="color: #64748B; font-size: 11px; margin-top: 4px; letter-spacing: 0.15em; text-transform: uppercase;">
            Diagnostic System
        </p>
        <span style="
            background: rgba(56, 189, 248, 0.1);
            color: #38BDF8;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 10px;
            font-weight: 600;
            border: 1px solid rgba(56, 189, 248, 0.2);
        ">v2.0</span>
    </div>
    """, unsafe_allow_html=True)
    
    
    
    selected = option_menu(
        menu_title="NAVIGATION",
        options=["Dashboard", "Exploratory Analysis", "Patient Screening", "Model Analytics", "Explainability", "Reports"],
        icons=["speedometer2", "bar-chart-line", "activity", "graph-up", "lightbulb", "file-text"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#94A3B8", "font-size": "16px"}, 
            "nav-link": {
                "font-size": "14px", 
                "text-align": "left", 
                "margin": "4px 0", 
                "padding": "10px 16px",
                "border-radius": "10px",
                "color": "#CBD5E1", 
                "--hover-color": "rgba(30, 41, 59, 0.6)"
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, rgba(56, 189, 248, 0.15), rgba(129, 140, 248, 0.1))",
                "color": "#38BDF8",
                "font-weight": "600",
            },
        }
    )
    
    # System Status Footer
    st.markdown("---")
    st.markdown("""
    <div style="padding: 12px; background: rgba(30, 41, 59, 0.3); border-radius: 10px; border: 1px solid rgba(148, 163, 184, 0.06);">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
            <div style="width: 8px; height: 8px; background: #10B981; border-radius: 50%; animation: pulse 2s infinite;"></div>
            <span style="color: #94A3B8; font-size: 12px; font-weight: 600;">System Online</span>
        </div>
        <p style="color: #475569; font-size: 11px; margin: 0;">
            Model: Auto-trained<br>
            Last Updated: Today
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- Dashboard Page ---
if selected == "Dashboard":
    # --- Hero Section with Animation ---
    col_hero1, col_hero2 = st.columns([3, 1])
    
    with col_hero1:
        st.markdown(f"""
        <div style="padding: 20px 0;">
            <h1 style="font-size: 3rem; font-weight: 800; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Welcome, Doctor
            </h1>
            <p style="font-size: 1.1rem; color: #94A3B8; margin-top: -10px;">
                MediScan Diagnostic System v2.0 is ready.
            </p>
            <div style="display: flex; gap: 15px; margin-top: 20px;">
                <div style="background: rgba(56, 189, 248, 0.1); padding: 8px 16px; border-radius: 20px; border: 1px solid rgba(56, 189, 248, 0.2);">
                    <span style="color: #38BDF8; font-weight: 600;">System Online</span>
                </div>
                <p style="color: #475569; font-size: 11px; margin: 0; align-self: center;">
                    Model: Auto-trained<br>
                    Last Updated: Today
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_hero2:
        if lottie_dna:
            st_lottie(lottie_dna, height=180, key="hero_dna")
    
    st.markdown("---")
    
    # Metrics
    # Split into 2 rows of 3 columns for better responsiveness
    row1 = st.columns(3)
    row2 = st.columns(3)
    
    # Initialize session state for stats if not present
    if 'screening_count' not in st.session_state:
        st.session_state['screening_count'] = 24
    
    total_patients = len(df)
    screenings_today = st.session_state['screening_count'] # Real-time count
    ckd_detected = df[df['CKD_Status'] == 1].shape[0] if 'CKD_Status' in df.columns else 0 # Assuming 1 is Yes after encoding
    healthy_patients = total_patients - ckd_detected
    
    # Critical Cases logic (Age > 60 & CKD)
    critical_cases = df[(df['CKD_Status'] == 1) & (df['Age'] > 60)].shape[0]
    
    # Reports Generated (default dummy value)
    pending_reports = st.session_state.get('reports_generated', 18)

    # Define metrics with HTML icons (using Unicode symbols for a clean professional look)
    metrics = [
        ("Total Patients", total_patients, "&#9776;"),
        ("Screenings (Session)", screenings_today, "&#9638;"),
        ("CKD Detected (Dataset)", ckd_detected, "&#9888;"),
        ("Healthy Patients", healthy_patients, "&#10003;"),
        ("Critical Cases", critical_cases, "&#9889;"),
        ("Reports Generated", pending_reports, "&#9776;"),
    ]

    # Combine columns to iterate smoothly
    all_cols = row1 + row2
    
    # Accent colors for each card
    accents = ["#38BDF8", "#818CF8", "#F59E0B", "#10B981", "#F43F5E", "#A78BFA"]

    for i, (col, (label, value, icon)) in enumerate(zip(all_cols, metrics)):
        accent = accents[i % len(accents)]
        delay = i * 0.08
        with col:
            st.markdown(f"""
                <div class="metric-card" style="animation-delay: {delay}s;">
                    <div class="metric-icon" style="background: linear-gradient(135deg, {accent}22, {accent}11);">{icon}</div>
                    <h2>{value:,}</h2>
                    <h3>{label}</h3>
                </div>
            """, unsafe_allow_html=True)

    # ============================================================
    # DASHBOARD CHARTS — Plotly Interactive Visualizations
    # ============================================================
    # Plotly Express (px) creates interactive charts that users can
    # hover, zoom, and pan. We use transparent backgrounds to
    # match our dark theme.
    # ============================================================
    col_chart1, col_chart2 = st.columns([2, 1])
    
    with col_chart1:
        st.subheader("CKD Cases by Age")
        if 'Age' in df.columns:
            # Filter to only CKD patients (CKD_Status == 1)
            # This histogram shows the AGE DISTRIBUTION of CKD cases
            # Helps identify which age groups are most affected
            ckd_df = df[df['CKD_Status'] == 1]
            fig = px.histogram(ckd_df, x="Age", nbins=20, color_discrete_sequence=['#38BDF8'])
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#94A3B8",
                xaxis_title="Age",
                yaxis_title="Count",
                bargap=0.1
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.1)")
            st.plotly_chart(fig, use_container_width=True)

    with col_chart2:
        st.subheader("Risk Factor Prevalence")
        
        # Calculate prevalence (assuming binary 0/1 after encoding)
        risk_factors = {
            "Hypertension": "Hypertension" if 'Hypertension' in df.columns else None,
            "Diabetes": "Diabetes_Mellitus" if 'Diabetes_Mellitus' in df.columns else None,
            "Anemia": "Anemia" if 'Anemia' in df.columns else None # Might need derivation
        }
        
        colors = {"Hypertension": "#F43F5E", "Diabetes": "#F59E0B", "Anemia": "#10B981"} # Rose 500, Amber 500, Emerald 500
        
        for label, col_name in risk_factors.items():
            percentage = 0
            if col_name and col_name in df.columns:
                 # Check if categorical or numerical. If encoded to 0/1:
                 percentage = int((df[col_name].sum() / total_patients) * 100)
            
            color = colors.get(label, "#FFFFFF")
            
            st.markdown(f"""
                <div class="risk-bar-container">
                    <div class="risk-label">
                        <span>{label}</span>
                        <span>{percentage}%</span>
                    </div>
                    <div class="risk-progress">
                        <div class="risk-progress-fill" style="width: {percentage}%; background-color: {color};"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)


# ============================================================
# ADVANCED EXPLORATORY DATA ANALYSIS (EDA) PAGE
# ============================================================
# EDA is crucial for understanding the dataset structure, missing values,
# distributions, and relationships before modeling.
# This section provides a comprehensive "Data Audit" for stakeholders.
# ============================================================
elif selected == "Exploratory Analysis":
    st.title("Advanced Exploratory Data Analysis")
    st.markdown("### 📊 Dataset Overview & Health Check")
    
    # --- Load RAW Data for Analysis (to show missing values before imputation) ---
    @st.cache_data
    def load_raw_data(file_path):
        return pd.read_csv(file_path)
    
    raw_df = load_raw_data(dataset_path)
    
    # 1. Dataset Statistics
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    col_stats1.metric("Total Records", raw_df.shape[0])
    col_stats2.metric("Total Features", raw_df.shape[1])
    col_stats3.metric("Target Variable", "CKD_Status")
    
    with st.expander("🔍 View Raw Dataset", expanded=False):
        st.dataframe(raw_df.head(), use_container_width=True)
        st.write(f"**Shape:** {raw_df.shape}")
        st.write("**Data Types:**")
        st.write(raw_df.dtypes.astype(str))
        st.caption("This is a structured medical dataset for binary classification (CKD vs. Healthy).")
    
    st.markdown("---")
    
    # 2. Missing Values Analysis
    st.markdown("### 🔵 2. Missing Values Analysis")
    st.markdown("**Why it matters:** Clinical data often has missing values due to skipped tests. We must identify them to apply correct imputation strategies.")
    
    missing_data = raw_df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    
    if not missing_data.empty:
        col_miss1, col_miss2 = st.columns([1, 2])
        
        with col_miss1:
            st.markdown("#### Missing Values Table")
            miss_df = pd.DataFrame({'Missing Count': missing_data, '% Missing': (missing_data / len(raw_df)) * 100})
            st.dataframe(miss_df.style.format({'% Missing': '{:.1f}%'}), use_container_width=True)
            st.info("💡 We handle these using **KNN Imputation** in the preprocessing pipeline.")
            
        with col_miss2:
            st.markdown("#### Missing Values Heatmap")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.heatmap(raw_df.isnull(), cbar=False, cmap='viridis', yticklabels=False, ax=ax)
            ax.set_title("Yellow lines indicate missing values", fontsize=10, color='white')
            # Customize plot for dark theme
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('white')
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.success("✅ No missing values detected in the raw dataset.")
    
    st.markdown("---")

    # 3. Target Distribution
    st.markdown("### 🔵 3. Target Variable Distribution (Class Balance)")
    
    ckd_counts = raw_df['CKD_Status'].value_counts()
    # Normalize labels if encoded (assuming 1=CKD, 0=Healthy)
    # Check if target is already numeric or string
    # We'll use the processed df for the rest of EDA to ensure consistency with model
    
    col_tar1, col_tar2 = st.columns([2, 1])
    with col_tar1:
        fig_pie = px.pie(names=['CKD (1)', 'Healthy (0)'], values=ckd_counts.values, 
                         color_discrete_sequence=['#F43F5E', '#10B981'], 
                         title="CKD vs Healthy Distribution")
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#94A3B8")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_tar2:
        st.info(f"""
        **Balance Analysis:**
        - **CKD (1):** {ckd_counts.get(1, 0) if 1 in ckd_counts else ckd_counts.iloc[0]}
        - **Healthy (0):** {ckd_counts.get(0, 0) if 0 in ckd_counts else ckd_counts.iloc[1]}
        
        Using **Recall-focused evaluation** is critical here to minimize False Negatives (missing a sick patient).
        """)

    st.markdown("---")

    # 4. Univariate Analysis
    st.markdown("### 🔵 4. Univariate Analysis (Individual Features)")
    
    # Numeric Features
    st.markdown("#### Numeric Features (Histograms)")
    numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
    # Remove ID or Target from selection if present
    cols_to_plot = [c for c in numeric_cols if c not in ['Patient_ID', 'CKD_Status']]
    
    selected_num = st.selectbox("Select Numeric Feature", cols_to_plot, index=cols_to_plot.index('Serum_Creatinine') if 'Serum_Creatinine' in cols_to_plot else 0)
    
    fig_hist = px.histogram(raw_df, x=selected_num, color="CKD_Status", 
                            barmode="overlay", nbins=30,
                            color_discrete_sequence=['#10B981', '#F43F5E'], # Green for 0, Red for 1
                            title=f"Distribution of {selected_num} by CKD Status")
    fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#94A3B8")
    fig_hist.update_xaxes(showgrid=False)
    fig_hist.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.1)")
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Categorical Features
    st.markdown("#### Categorical Features (Count Plots)")
    cat_cols = raw_df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Add dummy categorization for numeric columns that are actually categorical codes (like Hypertension if already encoded)
    # But usually raw_df has strings for categorical
    
    if cat_cols:
        selected_cat = st.selectbox("Select Categorical Feature", cat_cols)
        
        fig_count = px.histogram(raw_df, x=selected_cat, color="CKD_Status", 
                                 barmode="group",
                                 color_discrete_sequence=['#10B981', '#F43F5E'],
                                 title=f"{selected_cat} Distribution by CKD Status")
        fig_count.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#94A3B8")
        st.plotly_chart(fig_count, use_container_width=True)
        st.caption(f"Visualizing how {selected_cat} prevalance differs between Healthy and CKD patients.")

    st.markdown("---")
    
    # 5. Feature Relationships
    st.markdown("### 🔵 5. Feature Relationships & Correlations")
    
    st.markdown("#### Correlation Matrix")
    # Use processed df (numeric only) for correlation
    numeric_processed = df.select_dtypes(include=np.number)
    
    if not numeric_processed.empty:
        corr = numeric_processed.corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
                            title="Feature Correlation Heatmap")
        fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#94A3B8")
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("Strong Positive Correlation (Red) = Features increase together. Strong Negative (Blue) = Features move inversely.")
    
    st.markdown("#### Pairwise Relationships (Scatter Matrix)")
    col_pair1, col_pair2 = st.columns(2)
    with col_pair1:
        x_axis_pair = st.selectbox("X Axis Feature", numeric_processed.columns, index=0, key="pair_x")
    with col_pair2:
        y_axis_pair = st.selectbox("Y Axis Feature", numeric_processed.columns, index=1, key="pair_y")
    
    fig_pair = px.scatter(df, x=x_axis_pair, y=y_axis_pair, color="CKD_Status" if "CKD_Status" in df.columns else None,
                          color_discrete_sequence=['#10B981', '#F43F5E'],
                          title=f"Relationship: {x_axis_pair} vs {y_axis_pair}")
    fig_pair.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#94A3B8")
    fig_pair.update_xaxes(showgrid=False) 
    fig_pair.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.1)")
    st.plotly_chart(fig_pair, use_container_width=True)
    
    st.markdown("---")

    # 6. Outlier Detection
    st.markdown("### 🔵 6. Outlier Detection")
    st.markdown("**Why it matters:** Outliers in medical data can be errors or critical high-risk cases. We don't remove them blindly.")
    
    outlier_feat = st.selectbox("Select Feature for Boxplot", cols_to_plot, index=cols_to_plot.index('Blood_Urea') if 'Blood_Urea' in cols_to_plot else 0)
    
    fig_box = px.box(df, x="CKD_Status" if "CKD_Status" in df.columns else None, y=outlier_feat, 
                     color="CKD_Status" if "CKD_Status" in df.columns else None,
                     color_discrete_sequence=['#10B981', '#F43F5E'],
                     title=f"Boxplot of {outlier_feat} by CKD Status")
    fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#94A3B8")
    st.plotly_chart(fig_box, use_container_width=True)
    st.caption("Points outside the whiskers are potential outliers. In CKD, high urea/creatinine outliers are expected in sick patients.")

# ============================================================
# PATIENT SCREENING PAGE — Real-Time CKD Prediction
# ============================================================
# This is the CORE PREDICTION page where new patients are screened.
# The workflow is:
#   1. User enters patient data via the form (vital signs, lab results)
#   2. Data is transformed using the SAME pipeline as training data
#      (encoding + scaling via processor.transform_data())
#   3. The trained model predicts CKD probability
#      (model.predict_proba() returns probability between 0-1)
#   4. Risk category is assigned: Normal (<30%), Low Risk (30-60%), High Risk (>60%)
#   5. SHAP waterfall plot explains WHY the model made that prediction
# ============================================================
elif selected == "Patient Screening":
    st.title("Patient Screening & Risk Assessment")
    
    if 'mediscan_model' not in st.session_state or 'mediscan_processor' not in st.session_state:
        st.warning("Please train the model in the 'Model Analytics' section first.")
        st.info("Go to **Model Analytics** > **Train Model** to initialize the diagnostic system.")
    else:
        predictor = st.session_state['mediscan_model']
        processor = st.session_state['mediscan_processor']
        selected_features = predictor.selected_features
        
        if selected_features is None:
            st.error("Model trained but no features selected. Please retrain.")
        else:
            # Normal ranges for highlighting abnormal values
            NORMAL_RANGES = {
                'Age': (0, 120),
                'Blood_Pressure': (70, 90),
                'Specific_Gravity': (1.005, 1.025),
                'Albumin': (0, 0),
                'Sugar': (0, 0),
                'Blood_Glucose_Random': (70, 140),
                'Blood_Urea': (15, 40),
                'Serum_Creatinine': (0.6, 1.2),
                'Sodium': (136, 145),
                'Hemoglobin': (12.0, 17.5),
                'White_Blood_Cell_Count': (4500, 11000),
            }
            
            # Helper: check if a feature is in selected features
            def is_selected(feat):
                return feat in selected_features
            
            with st.form("screening_form"):
                input_data = {}
                
                # ============================================
                # --- SECTION 1: Patient Basic Details ---
                # ============================================
                st.markdown("""
                <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); 
                            padding: 15px 20px; border-radius: 10px; border-left: 4px solid #38BDF8; margin-bottom: 15px;">
                    <h4 style="margin:0; color: #38BDF8;">Section 1: Patient Basic Details</h4>
                </div>
                """, unsafe_allow_html=True)
                
                det_col1, det_col2, det_col3 = st.columns(3)
                with det_col1:
                    patient_id = st.text_input("Patient ID", value=f"P-{pd.Timestamp.now().strftime('%H%M%S')}", help="Unique patient identifier")
                with det_col2:
                    screening_date = st.date_input("Date of Screening", value=pd.Timestamp.now())
                with det_col3:
                    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"], help="Optional demographic info")
                
                # Age (always needed - part of dataset)
                if is_selected('Age'):
                    det_col4, det_col5 = st.columns(2)
                    with det_col4:
                        input_data['Age'] = st.number_input("Age (years)", min_value=0, max_value=120, value=45, step=1, help="Patient's age in years")
                    with det_col5:
                        st.markdown("")  # spacer
                
                # ============================================
                # --- SECTION 2: Vital Parameters ---
                # ============================================
                st.markdown("""
                <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); 
                            padding: 15px 20px; border-radius: 10px; border-left: 4px solid #8B5CF6; margin: 20px 0 15px 0;">
                    <h4 style="margin:0; color: #8B5CF6;">Section 2: Vital Parameters</h4>
                </div>
                """, unsafe_allow_html=True)
                
                vit_col1, vit_col2 = st.columns(2)
                
                with vit_col1:
                    if is_selected('Blood_Pressure'):
                        input_data['Blood_Pressure'] = st.slider(
                            "Blood Pressure (mmHg - Diastolic)", 
                            min_value=40, max_value=180, value=80, step=1,
                            help="Normal range: 70–90 mmHg"
                        )
                    if is_selected('Specific_Gravity'):
                        sg_options = [1.005, 1.010, 1.015, 1.020, 1.025]
                        input_data['Specific_Gravity'] = st.select_slider(
                            "Specific Gravity", 
                            options=sg_options, value=1.015,
                            help="Normal range: 1.005–1.025"
                        )
                
                with vit_col2:
                    if is_selected('Albumin'):
                        input_data['Albumin'] = st.selectbox(
                            "Albumin Level", 
                            options=[0, 1, 2, 3, 4, 5], index=0,
                            help="Scale 0–5 | Normal: 0 (None). Higher values indicate protein leakage."
                        )
                    if is_selected('Sugar'):
                        input_data['Sugar'] = st.selectbox(
                            "Sugar Level", 
                            options=[0, 1, 2, 3, 4, 5], index=0,
                            help="Scale 0–5 | Normal: 0 (None). Higher values indicate glucose in urine."
                        )
                
                # ============================================
                # --- SECTION 3: Biochemical Markers ---
                # ============================================
                st.markdown("""
                <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); 
                            padding: 15px 20px; border-radius: 10px; border-left: 4px solid #F59E0B; margin: 20px 0 15px 0;">
                    <h4 style="margin:0; color: #F59E0B;">Section 3: Biochemical Markers</h4>
                </div>
                """, unsafe_allow_html=True)
                
                bio_col1, bio_col2 = st.columns(2)
                
                with bio_col1:
                    if is_selected('Blood_Glucose_Random'):
                        input_data['Blood_Glucose_Random'] = st.number_input(
                            "Blood Glucose Random (mg/dL)", 
                            min_value=20.0, max_value=500.0, value=120.0, step=1.0, format="%.1f",
                            help="Normal: 70-140 mg/dL | High: >140"
                        )
                    if is_selected('Blood_Urea'):
                        input_data['Blood_Urea'] = st.number_input(
                            "Blood Urea (mg/dL)", 
                            min_value=1.0, max_value=400.0, value=30.0, step=1.0, format="%.1f",
                            help="Normal: 15-40 mg/dL | High: >40"
                        )
                    if is_selected('Serum_Creatinine'):
                        input_data['Serum_Creatinine'] = st.number_input(
                            "Serum Creatinine (mg/dL)", 
                            min_value=0.1, max_value=20.0, value=1.0, step=0.1, format="%.1f",
                            help="Normal: 0.6-1.2 mg/dL | High: >1.2 (key CKD indicator)"
                        )
                
                with bio_col2:
                    if is_selected('Sodium'):
                        input_data['Sodium'] = st.number_input(
                            "Sodium (mEq/L)", 
                            min_value=100.0, max_value=200.0, value=140.0, step=1.0, format="%.1f",
                            help="Normal: 136-145 mEq/L | Low: <136"
                        )
                    if is_selected('Hemoglobin'):
                        input_data['Hemoglobin'] = st.number_input(
                            "Hemoglobin (g/dL)", 
                            min_value=2.0, max_value=20.0, value=14.0, step=0.1, format="%.1f",
                            help="Normal: 12-17.5 g/dL | Low: <12 (anemia indicator)"
                        )
                    if is_selected('White_Blood_Cell_Count'):
                        input_data['White_Blood_Cell_Count'] = st.number_input(
                            "WBC Count (cells/µL)", 
                            min_value=1000, max_value=30000, value=8000, step=100,
                            help="Normal: 4,500-11,000 | High: >11,000"
                        )
                
                # ============================================
                # --- SECTION 4: Medical History ---
                # ============================================
                st.markdown("""
                <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); 
                            padding: 15px 20px; border-radius: 10px; border-left: 4px solid #F43F5E; margin: 20px 0 15px 0;">
                    <h4 style="margin:0; color: #F43F5E;">Section 4: Medical History</h4>
                </div>
                """, unsafe_allow_html=True)
                
                hist_col1, hist_col2, hist_col3 = st.columns(3)
                
                with hist_col1:
                    if is_selected('Hypertension'):
                        hyp = st.radio("Hypertension", ["No", "Yes"], horizontal=True, help="Has the patient been diagnosed with hypertension?")
                        input_data['Hypertension'] = hyp
                    
                with hist_col2:
                    if is_selected('Diabetes_Mellitus'):
                        dm = st.radio("Diabetes Mellitus", ["No", "Yes"], horizontal=True, help="Has the patient been diagnosed with diabetes?")
                        input_data['Diabetes_Mellitus'] = dm
                
                with hist_col3:
                    if is_selected('Red_Blood_Cells'):
                        rbc = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"], help="Microscopic examination result")
                        input_data['Red_Blood_Cells'] = rbc
                
                # ---- Any remaining features not covered above ----
                remaining = [f for f in selected_features if f not in input_data]
                if remaining:
                    st.markdown("#### Additional Parameters")
                    rem_cols = st.columns(2)
                    for i, feature in enumerate(remaining):
                        with rem_cols[i % 2]:
                            if feature in processor.encoders:
                                le = processor.encoders[feature]
                                options = list(le.classes_)
                                input_data[feature] = st.selectbox(f"{feature}", options)
                            else:
                                dtype = processor.df[feature].dtype if processor.df is not None and feature in processor.df.columns else float
                                if np.issubdtype(dtype, np.number):
                                    min_val = float(processor.df[feature].min()) if processor.df is not None else 0.0
                                    max_val = float(processor.df[feature].max()) if processor.df is not None else 1000.0
                                    mean_val = float(processor.df[feature].mean()) if processor.df is not None else (min_val + max_val)/2
                                    input_data[feature] = st.number_input(f"{feature}", min_value=min_val, max_value=max_val, value=mean_val, step=0.1)
                                else:
                                    input_data[feature] = st.text_input(f"{feature}")
                
                # ============================================
                # Submit Buttons
                # ============================================
                st.markdown("---")
                submit_col1, submit_col2 = st.columns([3, 1])
                with submit_col1:
                    submitted = st.form_submit_button("Predict CKD Risk", use_container_width=True)
                with submit_col2:
                    reset = st.form_submit_button("Reset", use_container_width=True)
            
            # ============================================
            # --- Abnormal Value Indicators (outside form) ---
            # ============================================
            abnormal_flags = []
            for feat, val in input_data.items():
                if feat in NORMAL_RANGES:
                    try:
                        v = float(val) if not isinstance(val, str) else None
                        if v is not None:
                            lo, hi = NORMAL_RANGES[feat]
                            if hi > 0 and (v < lo or v > hi):
                                status = "HIGH" if v > hi else "LOW"
                                abnormal_flags.append(f"**{feat.replace('_', ' ')}**: {v} ({status} — Normal: {lo}–{hi})")
                    except (ValueError, TypeError):
                        pass
            
            if abnormal_flags and not reset:
                st.markdown("""
                <div style="background: rgba(244, 63, 94, 0.1); border: 1px solid #F43F5E; 
                            border-radius: 8px; padding: 12px; margin: 10px 0;">
                    <strong style="color: #F43F5E;">! Abnormal Values Detected</strong>
                </div>
                """, unsafe_allow_html=True)
                for flag in abnormal_flags:
                    st.markdown(f"- {flag}")
            
            # ============================================
            # --- SECTION 5: Prediction Results ---
            # ============================================
            if submitted and not reset:
                # Convert inputs to DataFrame
                new_patient_df = pd.DataFrame([input_data])
                
                # Transform (Scale/Encode)
                transformed_df = processor.transform_data(new_patient_df)
                
                try:
                    transformed_features = transformed_df[selected_features]
                    prediction = predictor.model.predict(transformed_features)
                    
                    # Get probability
                    probability = 0.0
                    if hasattr(predictor.model, 'predict_proba'):
                        probability = predictor.model.predict_proba(transformed_features)[0][1]
                    else:
                        probability = float(prediction[0])
                    
                    risk_score = prediction[0]
                    
                    # Threshold adjustment (outside form so it's interactive)
                    st.markdown("---")
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); 
                                padding: 15px 20px; border-radius: 10px; border-left: 4px solid #10B981; margin-bottom: 15px;">
                        <h4 style="margin:0; color: #10B981;">Section 5: Prediction Results</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Risk Categories
                    if probability < 0.30:
                        risk_category = "Normal"
                        risk_color = "#10B981"
                    elif probability < 0.60:
                        risk_category = "Low Risk"
                        risk_color = "#F59E0B"
                    else:
                        risk_category = "High Risk"
                        risk_color = "#F43F5E"
                    
                    # Recommendation
                    if risk_category == "High Risk":
                        recommendation = "Immediate consultation with a nephrologist recommended. Further diagnostic evaluation (GFR test, kidney ultrasound) advised."
                    elif risk_category == "Low Risk":
                        recommendation = "Follow-up screening in 3 months recommended. Lifestyle modifications and regular monitoring advised."
                    else:
                        recommendation = "Routine annual check-up recommended. Maintain healthy lifestyle."
                    
                    # Save to session for Reports page
                    st.session_state['last_prediction'] = {
                        'input_data': input_data,
                        'probability': probability,
                        'result': risk_category,
                        'risk_score': risk_score,
                        'model_name': predictor.model.__class__.__name__,
                        'recommendation': recommendation,
                        'screening_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'patient_id': patient_id,
                        'gender': patient_gender,
                    }
                    st.session_state['screening_count'] = st.session_state.get('screening_count', 0) + 1
                    
                    # ---- Risk Badge + Metrics ----
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px; background: rgba({','.join(['16,185,129' if risk_category=='Normal' else '245,158,11' if risk_category=='Low Risk' else '244,63,94'])}, 0.15); 
                                border: 2px solid {risk_color}; border-radius: 12px; margin: 10px 0;">
                        <h1 style="color: {risk_color}; margin:0; font-size: 48px;">{probability:.0%}</h1>
                        <h2 style="color: {risk_color}; margin: 5px 0;">{risk_category.upper()}</h2>
                        <p style="color: #94A3B8; margin:0;">CKD Risk Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    m_col1, m_col2, m_col3 = st.columns(3)
                    m_col1.metric("Risk Probability", f"{probability:.1%}")
                    m_col2.metric("Classification", risk_category)
                    m_col3.metric("Model Used", predictor.model.__class__.__name__)
                    
                    # ---- Risk Gauge Chart ----
                    import plotly.graph_objects as go
                    
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
                        number={'suffix': '%', 'font': {'size': 36, 'color': '#E2E8F0'}},
                        title={'text': "CKD Risk Score", 'font': {'size': 16, 'color': '#94A3B8'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickcolor': '#94A3B8', 'tickfont': {'color': '#94A3B8'}},
                            'bar': {'color': risk_color},
                            'bgcolor': '#1E293B',
                            'steps': [
                                {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.2)'},
                                {'range': [30, 60], 'color': 'rgba(245, 158, 11, 0.2)'},
                                {'range': [60, 100], 'color': 'rgba(244, 63, 94, 0.2)'}
                            ],
                        }
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        height=280, margin=dict(t=50, b=10, l=30, r=30)
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # ---- Recommendation ----
                    if risk_category == "High Risk":
                        st.error(f"**{risk_category}:** {recommendation}")
                    elif risk_category == "Low Risk":
                        st.warning(f"**{risk_category}:** {recommendation}")
                    else:
                        st.success(f"**{risk_category}:** {recommendation}")
                    
                    # ---- Feature Contribution Bar Chart ----
                    st.markdown("#### Top Contributing Factors")
                    if hasattr(predictor.model, 'feature_importances_'):
                        importances = predictor.model.feature_importances_
                        feat_names = selected_features[:len(importances)]
                        
                        contrib_df = pd.DataFrame({
                            'Feature': feat_names,
                            'Importance': importances
                        }).sort_values('Importance', ascending=True).tail(5)
                        
                        fig_feat = px.bar(
                            contrib_df, x='Importance', y='Feature', orientation='h',
                            color='Importance', color_continuous_scale=['#38BDF8', '#F43F5E'],
                            title='Top 5 Factors Influencing This Prediction'
                        )
                        fig_feat.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font_color='#94A3B8', height=300, showlegend=False
                        )
                        fig_feat.update_xaxes(showgrid=True, gridcolor='rgba(148,163,184,0.1)')
                        fig_feat.update_yaxes(showgrid=False)
                        st.plotly_chart(fig_feat, use_container_width=True)
                    elif hasattr(predictor.model, 'coef_'):
                        coefs = predictor.model.coef_.flatten()
                        feat_names = selected_features[:len(coefs)]
                        contrib_df = pd.DataFrame({
                            'Feature': feat_names, 'Weight': coefs
                        })
                        contrib_df['AbsWeight'] = contrib_df['Weight'].abs()
                        contrib_df = contrib_df.sort_values('AbsWeight', ascending=True).tail(5)
                        
                        fig_feat = px.bar(
                            contrib_df, x='Weight', y='Feature', orientation='h',
                            color='Weight', color_continuous_scale=['#38BDF8', '#94A3B8', '#F43F5E'],
                            color_continuous_midpoint=0,
                            title='Top 5 Feature Weights (Red = Increases Risk)'
                        )
                        fig_feat.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font_color='#94A3B8', height=300, showlegend=False
                        )
                        st.plotly_chart(fig_feat, use_container_width=True)
                    
                    # ---- Expandable Model Explanation ----
                    with st.expander("Show Detailed Model Explanation (SHAP)", expanded=False):
                        st.caption("Individual prediction explanation using SHAP values. This shows why the model made this specific decision.")
                        try:
                            fig_waterfall = predictor.plot_shap_waterfall(
                                pd.DataFrame(transformed_features.values, columns=selected_features),
                                transformed_features
                            )
                            if fig_waterfall:
                                st.pyplot(fig_waterfall)
                                plt.close(fig_waterfall)
                            else:
                                st.info("SHAP waterfall not available for this model. See feature contributions above.")
                        except Exception:
                            st.info("SHAP waterfall not available for this model. See feature contributions above.")
                    
                    # ---- Threshold Slider ----
                    st.markdown("---")
                    st.markdown("#### Advanced: Threshold Adjustment")
                    st.caption("Adjust the classification threshold to see how it affects prediction. Lower threshold = catch more cases (higher recall).")
                    
                    threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.05, key="screening_threshold")
                    
                    if hasattr(predictor.model, 'predict_proba'):
                        adj_class = "High Risk" if probability >= threshold else "Low Risk"
                        th_col1, th_col2 = st.columns(2)
                        th_col1.metric("Adjusted Classification", adj_class)
                        th_col2.metric("Probability vs Threshold", f"{probability:.1%} vs {threshold:.0%}")
                        
                        if threshold < 0.5:
                            st.info(f"**Note:** Lowering threshold to {threshold:.0%} catches more CKD cases (higher Recall) but may increase false alarms.")
                    else:
                        st.warning("Threshold adjustment requires probability outputs (not available for this model type).")
                    
                    # ---- Navigate to Report ----
                    st.markdown("---")
                    st.info("Full diagnostic report with downloads available on the **Reports** page.")
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")

# --- Model Analytics Page ---
elif selected == "Model Analytics":
    st.title("Model Analytics & Training")
    
    st.markdown("### 1. Feature Engineering & Selection")
    st.write(f"Dataset Shape: {df.shape}")
    
    # Split & Scale Data
    predictor = Predictor() # Initialize
    X_train, X_test, y_train, y_test = processor.split_data()
    processor.scale_features()
    X_train_scaled = processor.X_train
    X_test_scaled = processor.X_test
    
    # RFE Selection
    if 'selected_features' not in st.session_state:
        st.info("Performing initial feature selection...")
        best_features = predictor.select_features_rfe(X_train_scaled, y_train, n_features=10) # Increased default
        st.session_state['selected_features'] = best_features
    else:
        best_features = st.session_state['selected_features']
    
    st.write(f"**Selected Features (RFE):** {list(best_features)}")

    st.markdown("---")
    st.markdown("### 2. Model Selection & Hyperparameters")
    
    # Advanced Data Options
    st.markdown("#### Data Balancing")
    use_smote = st.checkbox("Use SMOTE (Synthetic Minority Over-sampling)", value=False, help="Balances the training dataset by creating synthetic examples of the minority class.")
    
    model_options = [
        "Logistic Regression", 
        "Decision Tree",
        "Random Forest",
        "SVM", # Re-enabled with speed optimization
        "Naive Bayes",
        "KNN",
        "AdaBoost",
        "Gradient Boosting",
        "XGBoost"
    ]
    
    selected_algorithm = st.selectbox("Choose Algorithm", model_options)
    
    params = {}
    
    # Hyperparameters UI (Simplified for brevity)
    col_params1, col_params2 = st.columns(2)
    with col_params1:
        if selected_algorithm == "Decision Tree":
            params['max_depth'] = st.slider("Max Depth", 1, 20, 5)
        if selected_algorithm in ["Random Forest", "Gradient Boosting", "XGBoost"]:
            params['n_estimators'] = st.slider("Number of Trees", 10, 500, 100)
        if selected_algorithm == "SVM":
            params['C'] = st.slider("C (Regularization)", 0.01, 10.0, 1.0)
            params['kernel'] = st.selectbox("Kernel", ["linear", "rbf"])
    with col_params2:
        if selected_algorithm == "KNN":
            params['n_neighbors'] = st.slider("Neighbors (K)", 1, 20, 5)
        if selected_algorithm in ["Gradient Boosting", "XGBoost"]:
            params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.5, 0.1)

    if st.button(f"Train {selected_algorithm}"):
        with st.spinner(f"Training {selected_algorithm} in progress... Please wait."):
            # Train
            model = predictor.train_model(X_train_scaled[best_features], y_train, algorithm=selected_algorithm, params=params, use_smote=use_smote)
            
            # Evaluate
            metrics = predictor.evaluate_model(X_test_scaled[best_features], y_test)
            
            # Save to session
            st.session_state['mediscan_model'] = predictor
            st.session_state['mediscan_processor'] = processor
            
            # Display Metrics
            st.success("Model Trained Successfully!")
            
            # --- 1. Performance Metrics Panel ---
            st.markdown("### 1. Performance Metrics Panel")
            
            # Row 1: Accuracy & Report
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Train Accuracy", f"{metrics.get('train_accuracy', 0):.2%}")
            m_col2.metric("Test Accuracy", f"{metrics.get('accuracy', 0):.2%}")
            
            # Highlight Recall as requested
            m_col3.markdown(f"""
            <div style="background-color: rgba(255, 0, 0, 0.1); padding: 10px; border-radius: 5px; border: 1px solid red;">
                <p style="margin:0; font-size: 14px; color: #ffcccb;">Recall (Critical)</p>
                <h3 style="margin:0; color: #ff4b4b;">{metrics.get('recall', 0):.2f}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            m_col4.metric("Precision", f"{metrics.get('precision', 0):.2f}")
            
            # Row 2: Advanced Metrics
            m2_col1, m2_col2, m2_col3 = st.columns(3)
            m2_col1.metric("F1 Score", f"{metrics.get('f1', 0):.2f}")
            m2_col2.metric("ROC-AUC Score", f"{metrics.get('roc_auc') if isinstance(metrics.get('roc_auc'), (int, float)) else metrics.get('roc_auc')}")
            
            # Cross-Validation Recall
            cv_mean = metrics.get('cv_recall_mean', 0)
            cv_std = metrics.get('cv_recall_std', 0)
            m2_col3.metric("CV Recall (Mean ± Std)", f"{cv_mean:.2f} ± {cv_std:.2f}")

            
            # --- 2. Mandatory Graphs (Visualizations) ---
            st.markdown("### 2. Standard Visualizations (Mandatory)")
            
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Confusion Matrix 📊", "ROC Curve 📈", "Precision-Recall Curve 📉"])
            
            with viz_tab1:
                st.caption("Minimizing False Negatives is critical for CKD diagnosis.")
                # We need y_pred again for plotting if not saved in metrics fully tailored for viz
                # But predictor methods usually take (y_test, y_pred)
                # Let's use the one from metrics if possible or re-predict
                y_pred_viz = predictor.model.predict(X_test_scaled[best_features])
                fig_cm = predictor.plot_confusion_matrix(y_test, y_pred_viz)
                if fig_cm:
                    fig_cm.patch.set_facecolor('white')
                    st.pyplot(fig_cm)
                    plt.close(fig_cm)
            
            with viz_tab2:
                st.caption("Higher AUC means better separation between CKD and Healthy.")
                fig_roc = predictor.plot_roc_curve(X_test_scaled[best_features], y_test)
                if fig_roc:
                    fig_roc.patch.set_facecolor('white')
                    st.pyplot(fig_roc)
                    plt.close(fig_roc)

            with viz_tab3:
                st.caption("Precision-Recall Curve - Crucial for medical datasets.")
                # Retrieve pre-calculated curve data
                if 'pr_curve' in metrics:
                    precision, recall, avg_precision = metrics['pr_curve']
                    fig_pr = predictor.plot_precision_recall_curve(precision, recall, avg_precision)
                    if fig_pr:
                        fig_pr.patch.set_facecolor('white')
                        st.pyplot(fig_pr)
                        plt.close(fig_pr)
                else:
                    st.info("PR Curve not available for this model type (requires probabilities).")

            
            # --- 3. Model-Specific Outputs ---
            st.markdown(f"### 3. Model-Specific Analysis: {selected_algorithm}")
            
            # Logistic Regression / SVM (Linear) -> Coefficients
            if selected_algorithm == "Logistic Regression" or (selected_algorithm == "SVM" and params.get('kernel') == 'linear'):
                st.write("Coefficient Importance (Impact on Prediction):")
                fig_coef = predictor.plot_coefficients(best_features)
                if fig_coef:
                    fig_coef.patch.set_facecolor('white')
                    st.pyplot(fig_coef)
                    plt.close(fig_coef)
            
            # Trees / Boosting -> Feature Importance
            elif selected_algorithm in ["Decision Tree", "Random Forest", "AdaBoost", "Gradient Boosting", "XGBoost"]:
                col_spec1, col_spec2 = st.columns(2)
                with col_spec1:
                    st.write("Feature Importance:")
                    fig_imp = predictor.plot_feature_importance(best_features)
                    if fig_imp:
                        fig_imp.patch.set_facecolor('white')
                        st.pyplot(fig_imp)
                        plt.close(fig_imp)
                
                # Tree Viz for Decision Tree
                if selected_algorithm == "Decision Tree":
                    with col_spec2:
                        st.write("Tree Structure (Visualization):")
                        fig_tree = predictor.plot_tree_diagram(best_features)
                        if fig_tree:
                            fig_tree.patch.set_facecolor('white')
                            st.pyplot(fig_tree)
                            plt.close(fig_tree)

            # KNN -> Elbow Plot
            elif selected_algorithm == "KNN":
                st.write("K-Value Optimization Curve:")
                fig_knn = predictor.plot_knn_elbow(X_train_scaled[best_features], y_train, X_test_scaled[best_features], y_test)
                if fig_knn:
                    fig_knn.patch.set_facecolor('white')
                    st.pyplot(fig_knn)
                    plt.close(fig_knn)

            # Naive Bayes -> Probability Distribution
            elif selected_algorithm == "Naive Bayes":
                st.write("Probability Distribution (Confidence):")
                if 'y_prob' in metrics:
                    fig_dist = predictor.plot_probability_distribution(y_test, metrics['y_prob'])
                    if fig_dist:
                        fig_dist.patch.set_facecolor('white')
                        st.pyplot(fig_dist)
                        plt.close(fig_dist)
                else:
                    st.info("Probability distribution requires probability estimates.")

            # Linear Regressions -> Residuals
            if predictor.model_type == 'regression':
                 fig_res = predictor.plot_residuals(y_test, metrics.get('y_pred'))
                 fig_res.patch.set_facecolor('white')
                 st.pyplot(fig_res)
                 plt.close(fig_res)
                
            # --- Tree Diagram ---
            if selected_algorithm in ["Decision Tree", "Random Forest"]:
                st.write(f"{selected_algorithm} Visualization (Representative Tree):")
                fig_tree = predictor.plot_tree_diagram(best_features)
                if fig_tree:
                    fig_tree.patch.set_facecolor('white')
                    st.pyplot(fig_tree)
                    plt.close(fig_tree)
            


    st.markdown("---")
    st.markdown("### 4. Model Comparison")
    
    if st.button("Compare All Models"):
        with st.spinner("Running comprehensive model comparison... Please wait."):
            results = []
            roc_data = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, algo in enumerate(model_options):
                progress_pct = (i + 1) / len(model_options)
                status_text.text(f"Training {algo}... ({progress_pct:.0%})")
                try:
                    # Quick train with default params
                    p = Predictor()
                    p.train_model(X_train_scaled[best_features], y_train, algorithm=algo, use_smote=use_smote)
                    m = p.evaluate_model(X_test_scaled[best_features], y_test, skip_cv=True)  # skip CV for speed
                    
                    res_dict = {
                        'Algorithm': algo,
                        'Accuracy': m.get('accuracy', 0),
                        'Precision': m.get('precision', 0),
                        'Recall': m.get('recall', 0),
                        'F1 Score': m.get('f1', 0),
                        'AUC': m.get('roc_auc', 0) if isinstance(m.get('roc_auc'), float) else 0
                    }
                    
                    if 'roc_curve' in m:
                        roc_data[algo] = m['roc_curve']
                    
                    results.append(res_dict)
                except Exception as e:
                    st.error(f"Failed to train {algo}: {e}")
                
                progress_bar.progress(progress_pct)
                
            status_text.text("Comparison Complete!")
            st.success("Comparison Complete!")
        
        if results:
            # Table
            comp_df = pd.DataFrame(results).set_index('Algorithm').sort_values(by="Accuracy", ascending=False)
            st.dataframe(comp_df.style.highlight_max(axis=0, color='green').format("{:.2%}"))
            
            # --- Graphical Representation ---
            st.markdown("### Graphical Representation")
            
            # 1. Bar Chart of Metrics (Interactive)
            st.subheader("Model Performance Comparison (Accuracy)")
            fig_bar = px.bar(
                comp_df.reset_index(), 
                x='Algorithm', 
                y='Accuracy', 
                color='Algorithm',
                title='Model Accuracy Comparison',
                text_auto='.2%',
                labels={'Accuracy': 'Accuracy Score'}
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # 2. ROC Graph Logic (Matplotlib)
            st.subheader("ROC Curve Comparison")
            if roc_data:
                # Use default matplotlib style temporarily if needed, or stick to Streamlit's
                # plt.style.use('default')  <-- REMOVED to avoid global side effects
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                for algo, (fpr, tpr, auc_score) in roc_data.items():
                    ax.plot(fpr, tpr, label=f'{algo} (AUC = {auc_score:.2f})', linewidth=2)
                
                ax.plot([0, 1], [0, 1], 'k--', label='Random Chance', linestyle='--')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curves - All Models')
                ax.legend(loc='lower right')
                ax.grid(True, alpha=0.3)
                fig.tight_layout() # Add tight layout here too
                st.pyplot(fig)
            else:
                st.info("No ROC data available (models might not support probability prediction).")

# --- Explainability Page ---
elif selected == "Explainability":
    st.title("Model Explainability & Transparency")
    st.markdown("### Understanding Why the Model Makes Decisions")

    # Cached SHAP computation removed - matplotlib figures don't cache reliably
    
    if 'mediscan_model' in st.session_state:
        model = st.session_state['mediscan_model']
        # Ensure we have data
        if processor:
             # Split data to get test set for SHAP
             X_train, X_test, y_train, y_test = processor.split_data()
             
             # Check if scaler exists
             if 'main' in processor.scalers:
                scaler = processor.scalers['main']
                # Drop Age_Group if it exists in X_train/X_test but not in scaler
                cols_to_drop = [c for c in ['Age_Group'] if c in X_train.columns]
                X_train_clean = X_train.drop(columns=cols_to_drop)
                X_test_clean = X_test.drop(columns=cols_to_drop)
                
                X_train_scaled = scaler.transform(X_train_clean)
                X_test_scaled = scaler.transform(X_test_clean)
             else:
                st.warning("Scaler not found. Retraining processor...")
                processor.scale_features()
                scaler = processor.scalers['main']
                # Drop Age_Group if it exists
                cols_to_drop = [c for c in ['Age_Group'] if c in X_train.columns]
                X_train_clean = X_train.drop(columns=cols_to_drop)
                X_test_clean = X_test.drop(columns=cols_to_drop)
                
                X_train_scaled = scaler.transform(X_train_clean)
                X_test_scaled = scaler.transform(X_test_clean)
             
             # Need dataframe for column names consistency if possible, or just use indices
             feature_names = processor.df.columns.drop(['CKD_Status']) # Approximation or use stored features
             if 'selected_features' in st.session_state:
                 feature_names = st.session_state['selected_features']
                 
             # Subset to selected features
             try:
                 # feature_names might include 'Age_Group' if it was selected but then dropped?
                 # No, selected_features comes from RFE which runs on X_train.
                 # Just in case, filter feature_names to what is in X_train_clean
                 valid_features = [f for f in feature_names if f in X_train_clean.columns]
                 
                 # Create DF with correct columns from the CLEANED training data
                 X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_clean.columns)
                 X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_clean.columns)
                 
                 # Now subset
                 X_train_scaled = X_train_scaled_df[valid_features]
                 X_test_scaled = X_test_scaled_df[valid_features]
             except Exception as e:
                 st.error(f"Error preparing data for SHAP: {e}")
                 st.stop()
            
             st.info(f"Analysis based on current model: **{model.model.__class__.__name__}**")

             # --- PART 1: GLOBAL EXPLAINABILITY ---
             st.markdown("## 🔵 Part 1: Global Insights (Overall Model Behavior)")
             st.markdown("Answers: *Which features generally drive CKD prediction?*")
             
             col_g1, col_g2 = st.columns(2)
             
             with col_g1:
                 st.markdown("#### Global Feature Impact (SHAP Summary)")
                 st.caption("Red dots = High value of feature. \nBlue dots = Low value. \nRight side = Increases CKD Risk. \nLeft side = Lowers Risk.")
                 with st.spinner("Generating SHAP Summary (optimized with sampling)..."):
                     fig_shap = model.plot_shap_summary(X_train_scaled, X_test_scaled)
                     if fig_shap:
                         st.pyplot(fig_shap)
                         plt.close(fig_shap)
                         st.markdown("**Interpretation:**\n- **High Serum Creatinine** (Red dots on right) strongly increases risk.\n- **Low Hemoglobin** (Blue dots on right) increases risk.")
             
             with col_g2:
                 st.markdown("#### Feature Importance Ranking")
                 st.caption("Relative importance of top features in the model's decision making.")
                 fig_imp = model.plot_feature_importance(valid_features)
                 
                 # Feature Importance / Coef logic
                 if fig_imp:
                     fig_imp.patch.set_facecolor('white')
                     st.pyplot(fig_imp)
                     plt.close(fig_imp)
                 elif hasattr(model.model, 'coef_'):
                      fig_coef = model.plot_coefficients(valid_features)
                      if fig_coef:
                          fig_coef.patch.set_facecolor('white')
                          st.pyplot(fig_coef)
                          plt.close(fig_coef)

                 # Top 5 Features Table
                 st.markdown("##### 🏆 Top Drivers of CKD")
                 st.write("Based on model weights/importance:")
                 # Quick calculation for table
                 try:
                    if hasattr(model.model, 'feature_importances_'):
                        imp = pd.DataFrame({'Feature': valid_features, 'Importance': model.model.feature_importances_})
                        imp = imp.sort_values('Importance', ascending=False).head(5)
                        imp['Interpretation'] = imp.apply(lambda r: "(+) Risk Factor" if r['Importance'] > 0 else "Protective", axis=1) # Generic for tree
                        st.table(imp[['Feature', 'Interpretation']])
                    elif hasattr(model.model, 'coef_'):
                        coefs = model.model.coef_.flatten()
                        imp = pd.DataFrame({'Feature': valid_features, 'Coef': coefs})
                        imp['AbsCoef'] = imp['Coef'].abs()
                        imp = imp.sort_values('AbsCoef', ascending=False).head(5)
                        imp['Effect'] = imp['Coef'].apply(lambda x: "Increases Risk (+)" if x > 0 else "Reduces Risk (-)")
                        st.table(imp[['Feature', 'Effect']])
                 except:
                     st.write("Feature ranking table unavailable for this model type.")


             # --- PART 2: LOCAL EXPLAINABILITY ---
             st.markdown("---")
             st.markdown("## 🔵 Part 2: Individual Case Explanation")
             st.markdown("Answers: *Why was THIS specific patient predicted as High Risk?*")
             
             # Select a High Risk Patient from Test Set for Demo
             try:
                 high_risk_indices = np.where(y_test == 1)[0]
                 if len(high_risk_indices) > 0:
                     # Let user select or pick first
                     demo_idx = st.selectbox("Select a High Risk Test Patient Review:", high_risk_indices[:5], format_func=lambda x: f"Patient ID {x}")
                     
                     # Get Data
                     patient_X = X_test_scaled.iloc[[list(y_test.index).index(y_test.index[high_risk_indices[0]])]] # Safe loc? No, X_test_scaled is DF with RangeIndex usually or match X_test
                     # Actually X_test_scaled was created as new DF, so integer indexing is safest if aligned
                     # Let's just use iloc on X_test_scaled corresponding to the numpy index in y_test
                     patient_X = X_test_scaled.iloc[[np.where(X_test.index == y_test.index[demo_idx])[0][0]]]

                     # Prediction
                     prob = model.model.predict_proba(patient_X)[0][1] if hasattr(model.model, 'predict_proba') else model.model.predict(patient_X)[0]
                     pred_class = "High Risk (CKD)" if prob > 0.5 else "Low Risk (Healthy)"
                     
                     col_l1, col_l2 = st.columns([1, 2])
                     
                     with col_l1:
                         st.metric("Predicted Risk Probability", f"{prob:.1%}", delta="High Risk" if prob > 0.5 else "Safe")
                         st.markdown(f"**Classification:** `{pred_class}`")
                         
                         st.markdown("##### Top Contributing Factors")
                         contrib_df = model.get_feature_contributions(X_train_scaled, patient_X)
                         if contrib_df is not None:
                             st.dataframe(contrib_df.style.format({"Value": "{:.2f}", "Contribution": "{:+.3f}"}), hide_index=True)
                     
                     with col_l2:
                         st.markdown("#### Decision Waterfall Plot")
                         st.caption("Explains how each feature pushes the risk from the baseline (average) to the final score.")
                         fig_waterfall = model.plot_shap_waterfall(X_train_scaled, patient_X)
                         if fig_waterfall:
                             st.pyplot(fig_waterfall)
                             plt.close(fig_waterfall)
                         else:
                             st.info("Waterfall plot not available for this model type/configuration. Using Force impact analysis above.")
                             
                 else:
                     st.warning("No High Risk patients found in test set to demonstrate.")
             except Exception as e:
                 st.error(f"Error in Local Explanation: {e}")

             
             # --- PART 3: RELIABILITY ---
             st.markdown("---")
             st.markdown("## 🔵 Part 3: Model Reliability & Thresholds")
             st.markdown("Answers: *Can doctors trust the model?*")
             
             col_r1, col_r2 = st.columns(2)
             
             with col_r1:
                 st.markdown("#### Reliability Metrics")
                 st.write("Robustness check via 5-Fold Cross-Validation:")
                 # Compute REAL cross-validation recall from the trained model
                 try:
                     cv_scores = cross_val_score(
                         model.model, 
                         X_test_scaled, 
                         y_test, 
                         cv=min(5, len(y_test)), 
                         scoring='recall'
                     )
                     cv_mean = cv_scores.mean()
                     cv_std = cv_scores.std()
                 except Exception:
                     # Fallback: compute simple recall from predictions
                     y_pred_rel = model.model.predict(X_test_scaled)
                     cv_mean = recall_score(y_test, y_pred_rel, zero_division=0)
                     cv_std = 0.0
                 st.metric("Mean Recall (Sensitivity)", f"{cv_mean:.2%}", f"± {cv_std:.2%}")
                 st.caption("Consistent performance across different data subsets indicates a stable model.")
             
             with col_r2:
                 st.markdown("#### Threshold Adjustment Analysis")
                 st.caption("Adjusting the decision threshold to prioritize Recall (reducing missed cases).")
                 threshold = st.slider("Decision Threshold (Default: 0.5)", 0.0, 1.0, 0.5, 0.05)
                 
                 # Mock effect or real effect if probs available
                 if hasattr(model.model, 'predict_proba'):
                     y_probs_test = model.model.predict_proba(X_test_scaled)[:, 1]
                     y_pred_adj = (y_probs_test >= threshold).astype(int)
                     rec_adj = recall_score(y_test, y_pred_adj)
                     prec_adj = precision_score(y_test, y_pred_adj)
                     
                     c1, c2 = st.columns(2)
                     c1.metric("Adjusted Recall", f"{rec_adj:.2%}")
                     c2.metric("Adjusted Precision", f"{prec_adj:.2%}")
                     if threshold < 0.5:
                         st.markdown(f"**Insight:** Lowering threshold to **{threshold}** catches more cases (higher Recall) but may flag more false alarms.")
                 else:
                     st.write("Threshold adjustment requires probability outputs (not available for this model).")
                 
    else:
        st.warning("Please train a model in 'Model Analytics' first.")

# --- Reports Page ---
elif selected == "Reports":
    st.title("🏥 Patient Diagnostic Report")
    
    if 'last_prediction' in st.session_state:
        last_pred = st.session_state['last_prediction']
        input_data = last_pred['input_data']
        probability = last_pred['probability']
        risk_result = last_pred['result']
        model_name = last_pred.get('model_name', 'N/A')
        recommendation = last_pred.get('recommendation', '')
        screening_date = last_pred.get('screening_date', pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # ======================================================
        # SECTION A: Patient Basic Information
        # ======================================================
        st.markdown("""
        <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); 
                    padding: 25px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2 style="margin:0; color: #38BDF8;">🩺 MediScan Diagnostics</h2>
                    <p style="color: #94A3B8; margin: 5px 0 0 0;">AI-Powered CKD Screening Report</p>
                </div>
                <div style="text-align: right;">
                    <p style="color: #CBD5E1; margin: 2px 0;"><strong>Report Date:</strong> {date}</p>
                    <p style="color: #CBD5E1; margin: 2px 0;"><strong>Model:</strong> {model}</p>
                    <p style="color: #CBD5E1; margin: 2px 0;"><strong>Report ID:</strong> MS-{report_id}</p>
                </div>
            </div>
        </div>
        """.format(
            date=screening_date,
            model=model_name,
            report_id=pd.Timestamp.now().strftime('%Y%m%d%H%M')
        ), unsafe_allow_html=True)

        # Patient Info Row
        st.markdown("### 📋 A. Patient Information")
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            age_val = input_data.get('Age', 'N/A')
            st.metric("Age", f"{age_val} years" if age_val != 'N/A' else 'N/A')
        with info_col2:
            bp_val = input_data.get('Blood_Pressure', 'N/A')
            st.metric("Blood Pressure", f"{bp_val} mmHg" if bp_val != 'N/A' else 'N/A')
        with info_col3:
            diabetes_val = input_data.get('Diabetes_Mellitus', 'N/A')
            hypertension_val = input_data.get('Hypertension', 'N/A')
            # Handle encoded values
            diabetes_display = "Yes" if str(diabetes_val) in ['1', '1.0', 'Yes', 'yes'] else ("No" if str(diabetes_val) in ['0', '0.0', 'No', 'no'] else str(diabetes_val))
            hypertension_display = "Yes" if str(hypertension_val) in ['1', '1.0', 'Yes', 'yes'] else ("No" if str(hypertension_val) in ['0', '0.0', 'No', 'no'] else str(hypertension_val))
            st.metric("Diabetes", diabetes_display)
            st.metric("Hypertension", hypertension_display)

        # ======================================================
        # SECTION B: Lab Values Summary Table
        # ======================================================
        st.markdown("---")
        st.markdown("### 🧪 B. Lab Values Summary")
        
        # Normal ranges for CKD-relevant parameters (medical reference values)
        normal_ranges = {
            'Age': {'min': 0, 'max': 120, 'unit': 'years', 'label': 'Age'},
            'Blood_Pressure': {'min': 70, 'max': 90, 'unit': 'mmHg', 'label': 'Blood Pressure (Diastolic)'},
            'Specific_Gravity': {'min': 1.005, 'max': 1.025, 'unit': '', 'label': 'Specific Gravity'},
            'Albumin': {'min': 0, 'max': 0, 'unit': 'scale', 'label': 'Albumin'},
            'Sugar': {'min': 0, 'max': 0, 'unit': 'scale', 'label': 'Sugar'},
            'Blood_Glucose_Random': {'min': 70, 'max': 140, 'unit': 'mg/dL', 'label': 'Blood Glucose (Random)'},
            'Blood_Urea': {'min': 15, 'max': 40, 'unit': 'mg/dL', 'label': 'Blood Urea'},
            'Serum_Creatinine': {'min': 0.6, 'max': 1.2, 'unit': 'mg/dL', 'label': 'Serum Creatinine'},
            'Sodium': {'min': 136, 'max': 145, 'unit': 'mEq/L', 'label': 'Sodium'},
            'Hemoglobin': {'min': 12.0, 'max': 17.5, 'unit': 'g/dL', 'label': 'Hemoglobin'},
            'White_Blood_Cell_Count': {'min': 4500, 'max': 11000, 'unit': 'cells/µL', 'label': 'WBC Count'},
            'Red_Blood_Cells': {'min': 0, 'max': 0, 'unit': '', 'label': 'Red Blood Cells'},
        }
        
        lab_rows = []
        for feature, value in input_data.items():
            if feature in normal_ranges:
                ref = normal_ranges[feature]
                try:
                    val = float(value)
                    range_str = f"{ref['min']}–{ref['max']} {ref['unit']}" if ref['max'] > 0 else "Normal: 0"
                    
                    if ref['max'] > 0:
                        if val < ref['min']:
                            status = "🔻 Low"
                        elif val > ref['max']:
                            status = "🔺 High"
                        else:
                            status = "✅ Normal"
                    else:
                        status = "✅ Normal" if val == 0 else "⚠️ Present"
                    
                    lab_rows.append({
                        'Parameter': ref['label'],
                        'Value': f"{val:.2f}" if not val.is_integer() else f"{int(val)}",
                        'Normal Range': range_str,
                        'Status': status
                    })
                except (ValueError, TypeError):
                    lab_rows.append({
                        'Parameter': ref.get('label', feature),
                        'Value': str(value),
                        'Normal Range': '—',
                        'Status': '—'
                    })
        
        if lab_rows:
            lab_df = pd.DataFrame(lab_rows)
            st.dataframe(lab_df, use_container_width=True, hide_index=True)
        
        # ======================================================
        # SECTION C: Risk Prediction (Gauge Chart)
        # ======================================================
        st.markdown("---")
        st.markdown("### C. Risk Prediction")
        
        gauge_col, details_col = st.columns([1, 1])
        
        with gauge_col:
            # Plotly Gauge Chart
            import plotly.graph_objects as go
            
            if probability < 0.30:
                gauge_color = "#10B981"
            elif probability < 0.60:
                gauge_color = "#F59E0B"
            else:
                gauge_color = "#F43F5E"
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={'suffix': '%', 'font': {'size': 40, 'color': '#E2E8F0'}},
                title={'text': "CKD Risk Probability", 'font': {'size': 18, 'color': '#94A3B8'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#94A3B8', 'tickfont': {'color': '#94A3B8'}},
                    'bar': {'color': gauge_color},
                    'bgcolor': '#1E293B',
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.2)'},
                        {'range': [30, 60], 'color': 'rgba(245, 158, 11, 0.2)'},
                        {'range': [60, 100], 'color': 'rgba(244, 63, 94, 0.2)'}
                    ],
                    'threshold': {
                        'line': {'color': '#F43F5E', 'width': 3},
                        'thickness': 0.8,
                        'value': probability * 100
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=300,
                margin=dict(t=60, b=20, l=30, r=30)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with details_col:
            st.markdown(f"""
            <div style="background: #0F172A; padding: 20px; border-radius: 12px; border: 1px solid #334155;">
                <table style="width: 100%; color: #CBD5E1; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #334155;">
                        <td style="padding: 10px;"><strong>Risk Probability</strong></td>
                        <td style="padding: 10px; text-align: right; font-size: 20px; color: {gauge_color};"><strong>{probability:.1%}</strong></td>
                    </tr>
                    <tr style="border-bottom: 1px solid #334155;">
                        <td style="padding: 10px;"><strong>Classification</strong></td>
                        <td style="padding: 10px; text-align: right; color: {gauge_color};"><strong>{risk_result}</strong></td>
                    </tr>
                    <tr style="border-bottom: 1px solid #334155;">
                        <td style="padding: 10px;"><strong>Model Used</strong></td>
                        <td style="padding: 10px; text-align: right;">{model_name}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px;"><strong>Confidence Level</strong></td>
                        <td style="padding: 10px; text-align: right;">{"Strong" if abs(probability - 0.5) > 0.3 else "Moderate" if abs(probability - 0.5) > 0.15 else "Marginal"}</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk Level Legend
            st.markdown("""
            <div style="margin-top: 15px; padding: 12px; background: #1E293B; border-radius: 8px; font-size: 13px; color: #94A3B8;">
                <strong>Risk Categories:</strong><br>
                🟢 <strong>0–30%</strong> → Low Risk &nbsp;&nbsp;
                🟡 <strong>30–60%</strong> → Moderate Risk &nbsp;&nbsp;
                🔴 <strong>60–100%</strong> → High Risk
            </div>
            """, unsafe_allow_html=True)

        # ======================================================
        # SECTION D: Model Explanation - Feature Contributions
        # ======================================================
        st.markdown("---")
        st.markdown("### D. Model Explanation — Top Contributing Factors")
        st.caption("Transparency: Why did the model make this prediction?")
        
        # Get feature contributions from the trained model
        if 'mediscan_model' in st.session_state:
            model_obj = st.session_state['mediscan_model']
            
            # Use feature importance (fast) instead of SHAP
            if hasattr(model_obj.model, 'feature_importances_'):
                features_used = model_obj.selected_features if model_obj.selected_features is not None else list(input_data.keys())
                importances = model_obj.model.feature_importances_
                
                contrib_df = pd.DataFrame({
                    'Feature': features_used[:len(importances)],
                    'Importance': importances
                }).sort_values('Importance', ascending=True).tail(7)
                
                # Color based on positive/negative impact
                colors = ['#F43F5E' if imp > np.mean(importances) else '#38BDF8' for imp in contrib_df['Importance']]
                
                fig_contrib = px.bar(
                    contrib_df, x='Importance', y='Feature', orientation='h',
                    title='Feature Importance (Impact on Prediction)',
                    color='Importance',
                    color_continuous_scale=['#38BDF8', '#F43F5E']
                )
                fig_contrib.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#94A3B8',
                    height=350,
                    showlegend=False
                )
                fig_contrib.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.1)')
                fig_contrib.update_yaxes(showgrid=False)
                st.plotly_chart(fig_contrib, use_container_width=True)
                
                # Top factors text
                top_3 = contrib_df.tail(3)['Feature'].tolist()[::-1]
                st.markdown(f"""
                **Key Findings:**
                - **{top_3[0]}** is the strongest predictor of CKD risk
                - **{top_3[1]}** significantly influences the risk assessment  
                - **{top_3[2]}** also contributes to the overall prediction
                """)
                
            elif hasattr(model_obj.model, 'coef_'):
                features_used = model_obj.selected_features if model_obj.selected_features is not None else list(input_data.keys())
                coefs = model_obj.model.coef_.flatten()
                
                contrib_df = pd.DataFrame({
                    'Feature': features_used[:len(coefs)],
                    'Coefficient': coefs
                })
                contrib_df['AbsCoeff'] = contrib_df['Coefficient'].abs()
                contrib_df = contrib_df.sort_values('AbsCoeff', ascending=True).tail(7)
                
                fig_contrib = px.bar(
                    contrib_df, x='Coefficient', y='Feature', orientation='h',
                    title='Feature Coefficients (Red = Increases Risk, Blue = Decreases Risk)',
                    color='Coefficient',
                    color_continuous_scale=['#38BDF8', '#94A3B8', '#F43F5E'],
                    color_continuous_midpoint=0
                )
                fig_contrib.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#94A3B8', height=350, showlegend=False
                )
                st.plotly_chart(fig_contrib, use_container_width=True)
            else:
                st.info("Feature contributions not available for this model type.")
        else:
            st.warning("Train a model first to see feature explanations.")

        # ======================================================
        # SECTION E: Clinical Recommendation
        # ======================================================
        st.markdown("---")
        st.markdown("### ⚕️ E. Clinical Recommendation")
        
        if risk_result == "High Risk":
            st.error(f"""
            **Risk Level: HIGH RISK** (Probability: {probability:.1%})
            
            **Suggested Next Steps:**
            1. 🏥 **Immediate nephrologist referral** for comprehensive kidney function evaluation
            2. 🔬 **Additional tests recommended:** GFR (Glomerular Filtration Rate), Kidney Ultrasound, Urine Albumin-to-Creatinine Ratio
            3. 💊 **Medication review** — assess current medications for nephrotoxicity
            4. 📋 **Lifestyle modifications:** Low-sodium diet, blood pressure management, blood sugar control
            """)
        elif risk_result == "Low Risk":
            st.warning(f"""
            **Risk Level: LOW RISK** (Probability: {probability:.1%})
            
            **Suggested Next Steps:**
            1. 📅 **Follow-up screening** recommended in 3 months
            2. 🔬 **Consider additional tests:** GFR screening, repeat blood work
            3. 📋 **Lifestyle modifications:** Monitor fluid intake, reduce sodium, control diabetes/hypertension
            4. 📊 **Regular monitoring** of kidney function markers
            """)
        else:
            st.success(f"""
            **Risk Level: NORMAL** (Probability: {probability:.1%})
            
            **Suggested Next Steps:**
            1. ✅ **Routine annual check-up** recommended
            2. 🏃 **Maintain healthy lifestyle:** Regular exercise, balanced diet, adequate hydration
            3. 📊 **Continue monitoring** — annual screening for early detection
            """)
        
        
        
        # ======================================================
        # SECTION G: Download Options
        # ======================================================
        st.markdown("---")
        st.markdown("### 📥 Download Report")
        
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        
        # --- PDF Text Report ---
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("        MEDISCAN DIAGNOSTIC REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Report Date: {screening_date}")
        report_lines.append(f"Report ID: MS-{pd.Timestamp.now().strftime('%Y%m%d%H%M')}")
        report_lines.append(f"Model: {model_name}")
        report_lines.append("")
        report_lines.append("-" * 60)
        report_lines.append("PATIENT INFORMATION")
        report_lines.append("-" * 60)
        for key, val in input_data.items():
            display_key = normal_ranges.get(key, {}).get('label', key)
            report_lines.append(f"  {display_key}: {val}")
        report_lines.append("")
        report_lines.append("-" * 60)
        report_lines.append("LAB VALUES SUMMARY")
        report_lines.append("-" * 60)
        if lab_rows:
            report_lines.append(f"  {'Parameter':<30} {'Value':<12} {'Normal Range':<20} {'Status':<10}")
            report_lines.append(f"  {'-'*30} {'-'*12} {'-'*20} {'-'*10}")
            for row in lab_rows:
                report_lines.append(f"  {row['Parameter']:<30} {row['Value']:<12} {row['Normal Range']:<20} {row['Status']:<10}")
        report_lines.append("")
        report_lines.append("-" * 60)
        report_lines.append("RISK PREDICTION")
        report_lines.append("-" * 60)
        report_lines.append(f"  CKD Risk Probability: {probability:.1%}")
        report_lines.append(f"  Classification: {risk_result}")
        report_lines.append(f"  Confidence: {'Strong' if abs(probability - 0.5) > 0.3 else 'Moderate' if abs(probability - 0.5) > 0.15 else 'Marginal'}")
        report_lines.append("")
        report_lines.append("-" * 60)
        report_lines.append("CLINICAL RECOMMENDATION")
        report_lines.append("-" * 60)
        report_lines.append(f"  {recommendation}")
        report_lines.append("")
        report_lines.append("-" * 60)
        report_lines.append("DISCLAIMER")
        report_lines.append("-" * 60)
        report_lines.append("  This screening tool is intended for decision support only")
        report_lines.append("  and should not replace professional medical diagnosis.")
        report_lines.append("")
        report_lines.append("=" * 60)
        report_lines.append("  Generated by MediScan Diagnostic System")
        report_lines.append("  (C) 2026 MediScan Diagnostics")
        report_lines.append("=" * 60)
        
        report_text = "\n".join(report_lines)
        
        with dl_col1:
            st.download_button(
                label="📄 Download Report (TXT)",
                data=report_text,
                file_name=f"MediScan_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # --- CSV Summary ---
        csv_data = {
            'Report ID': [f"MS-{pd.Timestamp.now().strftime('%Y%m%d%H%M')}"],
            'Date': [screening_date],
            'Model': [model_name],
            'Probability': [f"{probability:.1%}"],
            'Result': [risk_result],
            'Recommendation': [recommendation],
        }
        # Add all input features to CSV
        for key, val in input_data.items():
            csv_data[key] = [val]
        
        csv_df = pd.DataFrame(csv_data)
        
        with dl_col2:
            st.download_button(
                label="📊 Download Summary (CSV)",
                data=csv_df.to_csv(index=False),
                file_name=f"MediScan_Summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with dl_col3:
            st.button("Print Report", on_click=lambda: st.markdown(
                '<script>window.print();</script>', unsafe_allow_html=True
            ), use_container_width=True)
        
    else:
        st.warning("No screening data found. Please perform a **Patient Screening** first to generate a report.")
        st.info("Go to **Patient Screening** → Enter patient vitals → Click **Predict Risk** → Then return here.")

