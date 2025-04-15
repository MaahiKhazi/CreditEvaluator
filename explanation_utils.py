import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

def format_instance_data(raw_data: Dict, numeric_cols: List[str]) -> pd.DataFrame:
    """Format user input data into a DataFrame for prediction."""
    processed_data = {}
    
    for col, value in raw_data.items():
        if col in numeric_cols:
            # Handle potential empty values
            if value == "":
                processed_data[col] = np.nan
            else:
                try:
                    processed_data[col] = float(value)
                except ValueError:
                    processed_data[col] = np.nan
        else:
            processed_data[col] = value
    
    return pd.DataFrame([processed_data])

def process_user_input() -> pd.DataFrame:
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
    
    # Define numeric columns (based on the original dataset)
    numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
    
    # Format as DataFrame
    instance_df = format_instance_data(loan_data, numeric_cols)
    
    return instance_df

def format_prediction_result(prediction: Dict) -> None:
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

def explain_counterfactual_concept() -> None:
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

def explain_fairness_concept() -> None:
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
