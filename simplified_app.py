import streamlit as st
import pandas as pd
import numpy as np
import logging
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DATASET_URL = "https://raw.githubusercontent.com/SasinduChanakaPiyumal/Loan-Prediction-Model/refs/heads/main/loan_data_set.csv"
SENSITIVE_FEATURE = "Gender"
PROTECTED_GROUP_VALUE = "Female"
TARGET_COLUMN = "Loan_Status"
TARGET_MAP = {'Y': 0, 'N': 1}  # 0 = Approved, 1 = Rejected

# Helper functions
def load_and_preprocess_data():
    """Load and preprocess loan dataset."""
    try:
        # Load data
        df_raw = pd.read_csv(DATASET_URL)
        
        # Basic cleaning
        df = df_raw.drop(columns=['Loan_ID'], errors='ignore')
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map(TARGET_MAP)
        df = df.dropna(subset=[TARGET_COLUMN])
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
        
        # Convert 'Dependents' if needed
        if 'Dependents' in df.columns and df['Dependents'].dtype == 'object':
            df['Dependents'] = df['Dependents'].replace('3+', '3').astype(float)
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def process_user_input():
    """Collect and process user input for loan prediction."""
    st.subheader("Loan Applicant Information")
    
    # Basic loan information
    col1, col2 = st.columns(2)
    
    with col1:
        loan_amount = st.number_input("Loan Amount (thousands)", min_value=0, value=100)
        loan_amount_term = st.number_input("Loan Term (months)", min_value=0, value=360)
        credit_history = st.selectbox("Credit History", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    
    with col2:
        applicant_income = st.number_input("Applicant Income (thousands)", min_value=0, value=5)
        coapplicant_income = st.number_input("Co-applicant Income (thousands)", min_value=0, value=0)
        property_area = st.selectbox("Property Area", options=["Urban", "Semiurban", "Rural"])
    
    # Personal information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", options=["Male", "Female"])
        married = st.selectbox("Married", options=["Yes", "No"])
    
    with col2:
        dependents = st.selectbox("Dependents", options=["0", "1", "2", "3+"])
        education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
    
    with col3:
        self_employed = st.selectbox("Self Employed", options=["Yes", "No"])
    
    # Create DataFrame from inputs
    loan_data = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }
    
    return pd.DataFrame([loan_data])

def make_prediction(instance_df):
    """Mock prediction function."""
    # This is a simplified mock model for demonstration
    # In real implementation, this would use the trained model
    
    # Higher credit history, income, and loan term increase approval chances
    credit_weight = 2.0
    income_weight = 0.5
    term_weight = 0.01
    
    # Base chance of approval
    approval_chance = 0.5
    
    # Factors affecting approval
    if instance_df["Credit_History"].iloc[0] == 1:
        approval_chance += credit_weight
    
    approval_chance += (instance_df["ApplicantIncome"].iloc[0] / 10) * income_weight
    approval_chance += (instance_df["Loan_Amount_Term"].iloc[0] / 360) * term_weight
    
    # Normalize to probability
    approval_chance = min(max(approval_chance, 0.1), 0.9)
    
    # Make prediction
    prediction = 0 if approval_chance > 0.5 else 1
    
    return {
        "prediction": prediction,
        "probability": {
            "approved": approval_chance if prediction == 0 else 1 - approval_chance,
            "rejected": 1 - approval_chance if prediction == 0 else approval_chance
        },
        "label": "Approved" if prediction == 0 else "Rejected"
    }

def format_prediction_result(prediction):
    """Format and display prediction results."""
    if "error" in prediction:
        st.error(f"Prediction error: {prediction['error']}")
        return
    
    # Determine color based on prediction
    color = "#66B2FF" if prediction["label"] == "Approved" else "#FF9999"
    
    # Create styled container for prediction
    st.markdown(
        f"""
        <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="color: white;">Loan {prediction["label"]}</h2>
            <p style="color: white; font-size: 18px;">
                Probability: {prediction["probability"]["approved"]*100:.2f}% Approved / 
                {prediction["probability"]["rejected"]*100:.2f}% Rejected
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

def plot_data_distribution(df):
    """Plot distribution of loan approval by gender."""
    st.subheader("Loan Status Distribution by Gender")
    
    # Count distribution by gender and target
    distribution = pd.crosstab(df[SENSITIVE_FEATURE], df[TARGET_COLUMN])
    distribution.columns = ["Approved", "Rejected"]
    
    # Display as a table
    st.dataframe(distribution)
    
    # Calculate approval rates
    approval_rates = df.groupby(SENSITIVE_FEATURE)[TARGET_COLUMN].mean()
    approval_rates = 1 - approval_rates  # Convert from rejection to approval
    
    # Display as a bar chart
    st.bar_chart(approval_rates)
    
    st.caption("This chart shows the approval rate by gender. Higher values indicate higher approval rates.")

def explain_fairness_concept():
    """Explain the concept of fairness in machine learning."""
    with st.expander("What is fairness in machine learning?"):
        st.markdown("""
        ### Fairness in Machine Learning
        
        Fairness in machine learning is about ensuring that the model doesn't discriminate against 
        certain groups of people based on sensitive attributes like gender, race, or age.
        
        **Key Fairness Metrics:**
        - **Equalized Odds Difference**: Measures whether the model has equal false positive and 
          false negative rates across different demographic groups. Lower values indicate better fairness.
        
        - **Demographic Parity**: Measures whether the acceptance rate is the same across different demographic groups.
        
        - **Accuracy Gap**: The difference in model accuracy between different demographic groups.
        
        **Our Approach to Fairness:**
        This application improves fairness by:
        1. Identifying rejected applications from protected groups
        2. Generating fair counterfactual examples
        3. Augmenting the training data with these examples
        4. Retraining the model to reduce bias
        
        By adding counterfactual examples that show reasonable paths to approval for protected groups,
        we help the model learn fairer decision boundaries without sacrificing overall performance.
        """)

def explain_counterfactual_concept():
    """Explain the concept of counterfactual explanations."""
    with st.expander("What are counterfactual explanations?"):
        st.markdown("""
        ### Counterfactual Explanations
        
        Counterfactual explanations show what changes would be needed to get a different outcome from the model. 
        They answer the question: *"What would need to be different about this loan application to get it approved?"*
        
        **Benefits:**
        - Provides actionable insights for loan applicants
        - Makes model decisions more transparent
        - Helps address potential biases in the model
        
        **Example:**
        If a loan was rejected, a counterfactual might show that increasing income by $10,000 
        or improving credit history would change the prediction to 'approved'.
        
        **Interpretation:**
        - Each counterfactual (CF) shows a set of changes that would flip the prediction
        - The fewer changes required, the closer the application is to approval
        - Some features might be easier to change than others
        """)

def suggest_improvements(instance_df, prediction):
    """Suggest improvements for loan approval."""
    if prediction["label"] == "Approved":
        st.success("Congratulations! Your loan application is likely to be approved.")
        return
    
    st.warning("Your loan application might be rejected. Here are some potential improvements:")
    
    improvements = []
    
    # Check credit history
    if instance_df["Credit_History"].iloc[0] == 0:
        improvements.append("Improve your credit history before applying")
    
    # Check income
    if instance_df["ApplicantIncome"].iloc[0] < 5:
        improvements.append("Increase your income (e.g., find additional income sources)")
    
    # Check loan amount vs. income
    income_ratio = instance_df["LoanAmount"].iloc[0] / (instance_df["ApplicantIncome"].iloc[0] + instance_df["CoapplicantIncome"].iloc[0])
    if income_ratio > 10:
        improvements.append("Reduce the loan amount or increase income to improve the loan-to-income ratio")
    
    # Display improvements
    for i, improvement in enumerate(improvements):
        st.info(f"{i+1}. {improvement}")
    
    if not improvements:
        st.info("Consider having a co-applicant with good credit history and steady income")

# Main application
st.set_page_config(page_title="Loan Approval System", page_icon="💸", layout="wide")

st.title("Loan Approval Prediction System")
st.markdown("""
This application demonstrates a loan approval prediction system with fairness considerations.
It helps identify potential patterns of bias in loan approvals and provides suggestions for
applicants to improve their chances.
""")

# Sidebar
st.sidebar.title("About")
st.sidebar.info("""
### Fairness in Loan Approval
This demo application illustrates how machine learning can be used for fair loan approval decisions.
The app considers various factors including income, credit history, and demographics.
""")

# Add fairness and counterfactual concept explanations
explain_fairness_concept()
explain_counterfactual_concept()

# Tab layout
tab1, tab2 = st.tabs(["Predict Loan Approval", "Data Analysis"])

with tab1:
    st.header("Predict Loan Approval")
    st.write("Enter your loan application details below to get a prediction.")
    
    # Process user input form
    with st.form("loan_prediction_form"):
        instance_df = process_user_input()
        submit_button = st.form_submit_button("Predict", use_container_width=True)
    
    if submit_button:
        with st.spinner("Making prediction..."):
            # In a real implementation, this would use the trained model
            prediction = make_prediction(instance_df)
            
            # Display prediction
            st.subheader("Prediction Result")
            format_prediction_result(prediction)
            
            # Suggest improvements if rejected
            st.subheader("Suggestions for Improvement")
            suggest_improvements(instance_df, prediction)

with tab2:
    st.header("Loan Data Analysis")
    
    # Load and display data
    with st.spinner("Loading data..."):
        df = load_and_preprocess_data()
        
    if df is not None:
        st.write(f"Dataset shape: {df.shape}")
        
        # Display sample of data
        st.subheader("Sample Data")
        st.dataframe(df.head())
        
        # Plot distribution
        st.subheader("Loan Approval Distribution")
        plot_data_distribution(df)
        
        # Display statistics
        st.subheader("Approval Statistics")
        
        # Calculate approval rates by gender
        approval_by_gender = df.groupby(SENSITIVE_FEATURE)[TARGET_COLUMN].mean()
        approval_by_gender = 1 - approval_by_gender  # Convert from rejection to approval
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Male Approval Rate", f"{approval_by_gender['Male']:.2%}")
        with col2:
            st.metric("Female Approval Rate", f"{approval_by_gender['Female']:.2%}")
        
        # Gender gap
        gender_gap = abs(approval_by_gender['Male'] - approval_by_gender['Female'])
        st.metric("Gender Approval Gap", f"{gender_gap:.2%}")
        
        if gender_gap > 0.05:  # 5% threshold for demonstration
            st.warning("There appears to be a significant gender gap in loan approvals. This could indicate bias in the approval process.")
        else:
            st.success("The gender gap in loan approvals is within acceptable limits.")