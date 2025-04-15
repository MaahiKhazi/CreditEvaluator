import streamlit as st
import pandas as pd
import numpy as np
import logging
import time
import os

# Import custom modules
from loan_predictor import LoanPredictor
from visualization_utils import (
    plot_metrics_comparison, 
    plot_counterfactuals, 
    plot_feature_importances,
    plot_fairness_metrics,
    plot_data_distribution
)
from explanation_utils import (
    process_user_input, 
    format_prediction_result,
    explain_counterfactual_concept,
    explain_fairness_concept
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MODEL_PATH = "models"
DATASET_URL = "https://raw.githubusercontent.com/SasinduChanakaPiyumal/Loan-Prediction-Model/refs/heads/main/loan_data_set.csv"
SENSITIVE_FEATURE = "Gender"
PROTECTED_GROUP_VALUE = "Female"
TARGET_COLUMN = "Loan_Status"

# Initialize session state
if "predictor" not in st.session_state:
    st.session_state.predictor = None
if "training_completed" not in st.session_state:
    st.session_state.training_completed = False
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"
if "explanation_instance" not in st.session_state:
    st.session_state.explanation_instance = None
if "loading_data" not in st.session_state:
    st.session_state.loading_data = False
if "training_model" not in st.session_state:
    st.session_state.training_model = False
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# Functions for page navigation
def set_page(page):
    st.session_state.current_page = page

# App title and description
st.set_page_config(page_title="Fairness-Aware Loan Approval", page_icon="💸", layout="wide")

st.title("Fairness-Aware Loan Approval System")
st.markdown("""
This application demonstrates a fairness-aware loan approval prediction system. It:
1. Trains a baseline model on loan approval data
2. Analyzes the model for fairness across demographic groups
3. Generates counterfactual explanations for rejected loans
4. Creates a fairer model through data augmentation
5. Provides interactive prediction with explanations
""")

# Sidebar navigation
st.sidebar.title("Navigation")
if st.sidebar.button("Home", use_container_width=True):
    set_page("home")
if st.sidebar.button("Model Training", use_container_width=True):
    set_page("model_training")
if st.sidebar.button("Fairness Analysis", use_container_width=True):
    set_page("fairness_analysis")
if st.sidebar.button("Loan Prediction", use_container_width=True):
    set_page("loan_prediction")
if st.sidebar.button("Counterfactual Explanations", use_container_width=True):
    set_page("counterfactual")

# Display fairness and counterfactual concepts in sidebar
st.sidebar.divider()
explain_fairness_concept()
explain_counterfactual_concept()

# Add model info if trained
if st.session_state.training_completed:
    st.sidebar.divider()
    st.sidebar.subheader("Model Status")
    st.sidebar.success("✅ Models Trained")
    
    # Add model comparison
    if st.session_state.predictor and st.session_state.predictor.baseline_metrics and st.session_state.predictor.fair_metrics:
        baseline_eod = st.session_state.predictor.baseline_metrics["EqualizedOddsDiff"]
        fair_eod = st.session_state.predictor.fair_metrics["EqualizedOddsDiff"]
        
        st.sidebar.markdown("**Fairness Improvement:**")
        st.sidebar.progress(1 - (fair_eod / baseline_eod) if baseline_eod > 0 else 0)
        st.sidebar.caption(f"Equalized Odds Difference: {baseline_eod:.4f} → {fair_eod:.4f}")

# Home page
if st.session_state.current_page == "home":
    st.header("Overview")
    st.write("""
    ### Why Fairness Matters in Loan Approval
    
    Loan approval decisions have significant impacts on people's lives. When machine learning models are used to 
    automate these decisions, they can unintentionally perpetuate or even amplify existing biases in the data.
    
    This application demonstrates a fairness-aware approach to loan approval prediction that:
    
    1. **Identifies bias** in a baseline prediction model
    2. **Generates explanations** for model decisions using counterfactuals
    3. **Improves fairness** through targeted data augmentation
    4. **Maintains performance** while reducing discriminatory outcomes
    
    ### How to Use This Application
    
    1. Start with the **Model Training** page to load data and train both baseline and fair models
    2. Explore the **Fairness Analysis** page to compare model performance across demographic groups
    3. Try the **Loan Prediction** page to see how both models evaluate loan applications
    4. Learn from the **Counterfactual Explanations** page to understand what changes would lead to approval
    
    ### Key Fairness Metric: Equalized Odds
    
    We use **Equalized Odds Difference** as our primary fairness metric. This measures whether the model has equal 
    false positive and false negative rates across different demographic groups. Lower values indicate better fairness.
    """)
    
    # Show example navigation
    st.divider()
    st.markdown("### Get Started")
    
    if st.button("Begin with Model Training →", use_container_width=True):
        set_page("model_training")

# Model Training page
elif st.session_state.current_page == "model_training":
    st.header("Model Training")
    
    if not st.session_state.predictor:
        st.session_state.predictor = LoanPredictor()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("1. Load and Preprocess Data")
        
        if st.button("Load Loan Dataset", key="load_data"):
            with st.spinner("Loading data..."):
                st.session_state.loading_data = True
                success = st.session_state.predictor.load_data(DATASET_URL)
                if success:
                    success = st.session_state.predictor.preprocess_data()
                    if success:
                        st.session_state.data_loaded = True
                        st.success("Data loaded and preprocessed successfully!")
                    else:
                        st.error("Failed to preprocess data.")
                else:
                    st.error("Failed to load data from URL.")
                st.session_state.loading_data = False
        
        if st.session_state.data_loaded:
            # Show dataset info
            df = st.session_state.predictor.df
            st.write(f"Dataset shape: {df.shape}")
            
            # Display data distribution
            plot_data_distribution(df, SENSITIVE_FEATURE, TARGET_COLUMN)
            
            st.divider()
            st.subheader("2. Train Models")
            
            if not st.session_state.training_completed:
                if st.button("Train Baseline and Fair Models", key="train_models"):
                    with st.spinner("Training baseline model..."):
                        st.session_state.training_model = True
                        # 1. Train baseline
                        success = st.session_state.predictor.train_baseline_model()
                        if not success:
                            st.error("Failed to train baseline model.")
                            st.session_state.training_model = False
                        
                        # 2. Setup counterfactual explainer
                        if success:
                            success = st.session_state.predictor.setup_counterfactual_explainer()
                            if not success:
                                st.error("Failed to setup counterfactual explainer.")
                                st.session_state.training_model = False
                        
                        # 3. Generate counterfactuals
                        if success:
                            st.info("Generating counterfactuals for rejected applications...")
                            success = st.session_state.predictor.generate_counterfactuals()
                            if not success:
                                st.warning("Could not generate sufficient counterfactuals.")
                                st.session_state.training_model = False
                        
                        # 4. Augment data and retrain
                        if success:
                            st.info("Augmenting data and training fair model...")
                            success = st.session_state.predictor.augment_data_and_retrain()
                            if not success:
                                st.error("Failed to train fair model.")
                                st.session_state.training_model = False
                        
                        # Set training completion flag
                        if success:
                            st.session_state.training_completed = True
                            st.success("Models trained successfully!")
                        
                        st.session_state.training_model = False
            else:
                st.success("Models already trained!")
                
                # Option to retrain
                if st.button("Retrain Models", key="retrain_models"):
                    st.session_state.training_completed = False
                    st.rerun()
    
    with col2:
        st.subheader("Training Process")
        
        st.markdown("""
        1. **Data Loading**
           - Load loan dataset
           - Clean and preprocess features
           - Split into train/test sets
           
        2. **Baseline Model**
           - Train LightGBM classifier
           - Evaluate performance & fairness
           
        3. **Counterfactual Generation**
           - Identify rejected protected applicants
           - Generate fair approval scenarios
           - Filter by minimal change threshold
           
        4. **Fair Model Training**
           - Augment data with fair examples
           - Retrain model on extended dataset
           - Evaluate improvement in fairness
        """)
        
        # Show progress indicators
        stages = [
            ("Data Loading", st.session_state.data_loaded or st.session_state.loading_data),
            ("Baseline Model", st.session_state.training_completed or st.session_state.training_model),
            ("Counterfactual Generation", st.session_state.training_completed or st.session_state.training_model),
            ("Fair Model", st.session_state.training_completed)
        ]
        
        for i, (stage, completed) in enumerate(stages):
            progress_color = "#00CC00" if completed else "#CCCCCC"
            progress_icon = "✅" if completed else "⬜"
            
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <div style="background-color: {progress_color}; width: 20px; height: 20px; border-radius: 50%; margin-right: 8px;"></div>
                    <div>{progress_icon} {stage}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Next step button
        if st.session_state.training_completed:
            st.divider()
            if st.button("View Fairness Analysis →", use_container_width=True):
                set_page("fairness_analysis")

# Fairness Analysis page
elif st.session_state.current_page == "fairness_analysis":
    st.header("Fairness Analysis")
    
    if not st.session_state.training_completed:
        st.warning("Please train the models first.")
        if st.button("Go to Model Training", use_container_width=True):
            set_page("model_training")
    else:
        # Get metrics comparison
        metrics_comparison = st.session_state.predictor.get_metrics_comparison()
        
        # Display metrics comparison
        st.subheader("Performance and Fairness Metrics")
        plot_metrics_comparison(metrics_comparison)
        
        # Display group-wise metrics
        st.divider()
        st.subheader("Group-Wise Performance Analysis")
        plot_fairness_metrics(
            st.session_state.predictor.baseline_metrics, 
            st.session_state.predictor.fair_metrics,
            SENSITIVE_FEATURE
        )
        
        # Display feature importance
        st.divider()
        st.subheader("Feature Importance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Baseline Model")
            baseline_importances = st.session_state.predictor.get_feature_importances("baseline")
            plot_feature_importances(baseline_importances)
        
        with col2:
            st.markdown("### Fair Model")
            fair_importances = st.session_state.predictor.get_feature_importances("fair")
            plot_feature_importances(fair_importances)
        
        # Navigation
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Model Training", use_container_width=True):
                set_page("model_training")
        with col2:
            if st.button("Try Loan Prediction →", use_container_width=True):
                set_page("loan_prediction")

# Loan Prediction page
elif st.session_state.current_page == "loan_prediction":
    st.header("Loan Prediction")
    
    if not st.session_state.training_completed:
        st.warning("Please train the models first.")
        if st.button("Go to Model Training", use_container_width=True):
            set_page("model_training")
    else:
        st.write("Enter loan applicant information to get a prediction from both the baseline and fair models.")
        
        # Process user input
        with st.form("loan_prediction_form"):
            instance_df = process_user_input()
            
            col1, col2 = st.columns(2)
            with col1:
                submit_button = st.form_submit_button("Predict", use_container_width=True)
            with col2:
                explain_button = st.form_submit_button("Predict and Explain", use_container_width=True)
        
        if submit_button or explain_button:
            # Make predictions with both models
            baseline_prediction = st.session_state.predictor.predict(instance_df, use_fair_model=False)
            fair_prediction = st.session_state.predictor.predict(instance_df, use_fair_model=True)
            
            # Display predictions side by side
            st.subheader("Prediction Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Baseline Model")
                format_prediction_result(baseline_prediction)
            
            with col2:
                st.markdown("### Fair Model")
                format_prediction_result(fair_prediction)
            
            # If explain button is clicked, generate counterfactual and store for explanation page
            if explain_button:
                # Add target column for counterfactual generation
                instance_with_target = instance_df.copy()
                # Use the fair model prediction
                instance_with_target["Loan_Status"] = fair_prediction["prediction"]
                
                # Store in session state for explanation page
                st.session_state.explanation_instance = instance_with_target
                
                # Navigate to explanation page
                if st.button("View Counterfactual Explanations →", use_container_width=True):
                    set_page("counterfactual")
        
        # Navigation
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Fairness Analysis", use_container_width=True):
                set_page("fairness_analysis")
        with col2:
            if st.button("View Counterfactual Explanations →", use_container_width=True):
                set_page("counterfactual")

# Counterfactual Explanations page
elif st.session_state.current_page == "counterfactual":
    st.header("Counterfactual Explanations")
    
    if not st.session_state.training_completed:
        st.warning("Please train the models first.")
        if st.button("Go to Model Training", use_container_width=True):
            set_page("model_training")
    else:
        st.write("""
        Counterfactual explanations show what changes would be needed to get a different outcome from the model.
        For a rejected loan application, they show what factors would need to change for the loan to be approved.
        """)
        
        # Check if we have an instance to explain
        if st.session_state.explanation_instance is not None:
            instance = st.session_state.explanation_instance
            
            # Check if it was rejected
            prediction = st.session_state.predictor.predict(instance, use_fair_model=True)
            
            if prediction["label"] == "Rejected":
                st.subheader("Explaining Rejected Loan Application")
                
                # Generate counterfactuals
                with st.spinner("Generating counterfactual explanations..."):
                    counterfactuals = st.session_state.predictor.explain_instance(instance)
                
                # Plot counterfactuals
                plot_counterfactuals(
                    instance, 
                    counterfactuals,
                    st.session_state.predictor.numerical_features,
                    st.session_state.predictor.categorical_features
                )
            else:
                st.success("This loan application is already predicted to be approved!")
        else:
            st.info("No loan application to explain. Please use the Loan Prediction page first.")
            
            if st.button("Go to Loan Prediction", use_container_width=True):
                set_page("loan_prediction")
        
        # Navigation
        st.divider()
        if st.button("← Back to Loan Prediction", use_container_width=True):
            set_page("loan_prediction")

# Show footer
st.divider()
st.caption("Fairness-Aware Loan Approval System with Counterfactual Explanations")
