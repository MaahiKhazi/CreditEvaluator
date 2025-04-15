import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple

def plot_metrics_comparison(metrics_comparison: Dict) -> None:
    """Plot comparison of metrics between baseline and fair models."""
    if "error" in metrics_comparison:
        st.error(metrics_comparison["error"])
        return
    
    # Extract metrics
    metrics = ["Accuracy", "F1", "AUC", "EqualizedOddsDiff"]
    baseline_values = [metrics_comparison["baseline"][m] for m in metrics]
    fair_values = [metrics_comparison["fair"][m] for m in metrics]
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        "Metric": metrics * 2,
        "Value": baseline_values + fair_values,
        "Model": ["Baseline"] * len(metrics) + ["Fair"] * len(metrics)
    })
    
    # Create and display the plot
    fig = px.bar(
        df, 
        x="Metric", 
        y="Value", 
        color="Model", 
        barmode="group",
        title="Model Performance Comparison",
        labels={"Value": "Score", "Metric": "Metric"},
        color_discrete_sequence=["#FF9999", "#66B2FF"]
    )
    
    # Adjust the y-axis to start from 0
    fig.update_layout(yaxis_range=[0, max(baseline_values + fair_values) * 1.1])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show fairness improvement
    st.subheader("Fairness Improvement")
    
    eod_improvement = metrics_comparison["improvement"]["EqualizedOddsDiff"]
    
    st.metric(
        label="Equalized Odds Difference Reduction", 
        value=f"{abs(eod_improvement):.4f}",
        delta=f"{abs(eod_improvement):.4f}" if eod_improvement >= 0 else f"-{abs(eod_improvement):.4f}"
    )
    
    # Add explanation
    st.caption("Lower Equalized Odds Difference indicates better fairness across demographic groups.")

def plot_counterfactuals(original_instance: pd.DataFrame, counterfactuals, numerical_features: List, categorical_features: List) -> None:
    """Plot comparison between original instance and its counterfactuals."""
    if counterfactuals is None or not hasattr(counterfactuals, "cf_examples_list") or not counterfactuals.cf_examples_list:
        st.error("No valid counterfactuals generated.")
        return
    
    # Get CF DataFrame
    cf_df = counterfactuals.cf_examples_list[0].final_cfs_df
    
    if cf_df is None or cf_df.empty:
        st.error("No valid counterfactuals generated.")
        return
    
    # Prepare original instance data
    original_data = original_instance.iloc[0].drop('Loan_Status', errors='ignore').to_dict()
    
    # Prepare counterfactual data
    cf_data_list = []
    for i, row in cf_df.iterrows():
        cf_data = row.drop('Loan_Status', errors='ignore').to_dict()
        
        # Count changes
        changes = sum(1 for k in cf_data if original_data.get(k) != cf_data.get(k))
        
        cf_dict = {
            "CF_ID": f"CF {i+1}", 
            "Changes": changes,
            **cf_data
        }
        cf_data_list.append(cf_dict)
    
    # Sort counterfactuals by number of changes
    cf_data_list = sorted(cf_data_list, key=lambda x: x["Changes"])
    
    # Create DataFrame with all data
    all_data = [{"CF_ID": "Original", "Changes": 0, **original_data}] + cf_data_list
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(all_data)
    
    # Create a nicer display table
    display_cols = ["CF_ID", "Changes"] + [col for col in comparison_df.columns 
                                         if col not in ["CF_ID", "Changes"]]
    
    # Style the DataFrame
    st.dataframe(
        comparison_df[display_cols].style.apply(
            lambda x: ['background-color: #ffff99' if x.name == 0 else '' for _ in x],
            axis=1
        ),
        use_container_width=True
    )
    
    # Highlight changes in each counterfactual
    st.subheader("Feature Changes in Counterfactuals")
    
    # Only show counterfactuals (skip original)
    for i, cf in enumerate(cf_data_list):
        cols = st.columns([1, 4])
        with cols[0]:
            st.write(f"**CF {i+1}**")
            st.write(f"Changes: {cf['Changes']}")
        
        with cols[1]:
            changes = []
            for feature in sorted(original_data.keys()):
                if original_data[feature] != cf[feature]:
                    changes.append(f"**{feature}**: {original_data[feature]} → {cf[feature]}")
            
            if changes:
                st.write(", ".join(changes))
            else:
                st.write("No changes")
        
        st.divider()

def plot_feature_importances(feature_importances: Dict, top_n: int = 10) -> None:
    """Plot feature importances from the model."""
    if "error" in feature_importances:
        st.error(feature_importances["error"])
        return
    
    # Convert to DataFrame
    importance_df = pd.DataFrame({
        "Feature": list(feature_importances.keys())[:top_n],
        "Importance": list(feature_importances.values())[:top_n]
    })
    
    # Create and display the plot
    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Top {top_n} Feature Importances",
        labels={"Importance": "Importance Score", "Feature": "Feature"},
        color="Importance",
        color_continuous_scale="Blues"
    )
    
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    st.plotly_chart(fig, use_container_width=True)

def plot_fairness_metrics(baseline_metrics: Dict, fair_metrics: Dict, sensitive_feature: str = "Gender") -> None:
    """Plot group-wise metrics comparison."""
    if "GroupMetrics" not in baseline_metrics or "GroupMetrics" not in fair_metrics:
        st.error("Group metrics data not available.")
        return
    
    baseline_group_metrics = baseline_metrics["GroupMetrics"]
    fair_group_metrics = fair_metrics["GroupMetrics"]
    
    if baseline_group_metrics is None or fair_group_metrics is None:
        st.error("Group metrics data not available for comparison.")
        return
    
    # Convert to DataFrames for easier manipulation
    baseline_df = baseline_group_metrics.reset_index()
    fair_df = fair_group_metrics.reset_index()
    
    # Rename for clarity
    baseline_df["Model"] = "Baseline"
    fair_df["Model"] = "Fair"
    
    # Combine data
    combined_df = pd.concat([baseline_df, fair_df])
    
    # Plot accuracy by group
    st.subheader(f"Accuracy by {sensitive_feature} Group")
    
    fig = px.bar(
        combined_df,
        x=sensitive_feature,
        y="accuracy",
        color="Model",
        barmode="group",
        title=f"Accuracy by {sensitive_feature}",
        labels={"accuracy": "Accuracy Score", sensitive_feature: sensitive_feature},
        color_discrete_sequence=["#FF9999", "#66B2FF"]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Plot F1 by group
    st.subheader(f"F1 Score by {sensitive_feature} Group")
    
    fig = px.bar(
        combined_df,
        x=sensitive_feature,
        y="f1",
        color="Model",
        barmode="group",
        title=f"F1 Score by {sensitive_feature}",
        labels={"f1": "F1 Score", sensitive_feature: sensitive_feature},
        color_discrete_sequence=["#FF9999", "#66B2FF"]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate and display the gap between groups
    st.subheader("Performance Gap Between Groups")
    
    # Accuracy gap
    baseline_acc_gap = baseline_df.groupby("Model")["accuracy"].max() - baseline_df.groupby("Model")["accuracy"].min()
    fair_acc_gap = fair_df.groupby("Model")["accuracy"].max() - fair_df.groupby("Model")["accuracy"].min()
    acc_gap_reduction = baseline_acc_gap.values[0] - fair_acc_gap.values[0]
    
    # F1 gap
    baseline_f1_gap = baseline_df.groupby("Model")["f1"].max() - baseline_df.groupby("Model")["f1"].min()
    fair_f1_gap = fair_df.groupby("Model")["f1"].max() - fair_df.groupby("Model")["f1"].min()
    f1_gap_reduction = baseline_f1_gap.values[0] - fair_f1_gap.values[0]
    
    # Display metrics
    cols = st.columns(2)
    with cols[0]:
        st.metric(
            label="Accuracy Gap Reduction", 
            value=f"{acc_gap_reduction:.4f}",
            delta=f"{acc_gap_reduction:.4f}" if acc_gap_reduction > 0 else f"-{abs(acc_gap_reduction):.4f}"
        )
    
    with cols[1]:
        st.metric(
            label="F1 Score Gap Reduction", 
            value=f"{f1_gap_reduction:.4f}",
            delta=f"{f1_gap_reduction:.4f}" if f1_gap_reduction > 0 else f"-{abs(f1_gap_reduction):.4f}"
        )
    
    st.caption("Positive values indicate reduced performance gaps between demographic groups, which suggests improved fairness.")

def plot_data_distribution(df: pd.DataFrame, sensitive_feature: str = "Gender", target_column: str = "Loan_Status") -> None:
    """Plot distribution of data, focusing on sensitive attribute and target."""
    if sensitive_feature not in df.columns or target_column not in df.columns:
        st.error(f"Required columns {sensitive_feature} or {target_column} not found in the dataset.")
        return
    
    # Count distribution by sensitive feature and target
    distribution = df.groupby([sensitive_feature, target_column]).size().reset_index(name="Count")
    
    # Map target values to readable labels
    target_map = {0: "Approved", 1: "Rejected"}
    distribution[target_column] = distribution[target_column].map(target_map)
    
    # Plot
    fig = px.bar(
        distribution,
        x=sensitive_feature,
        y="Count",
        color=target_column,
        barmode="group",
        title=f"Loan Status Distribution by {sensitive_feature}",
        labels={"Count": "Number of Applications", sensitive_feature: sensitive_feature},
        color_discrete_sequence=["#66B2FF", "#FF9999"]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate approval rates
    approval_rates = df.groupby(sensitive_feature)[target_column].mean().reset_index()
    approval_rates["Approval Rate"] = 1 - approval_rates[target_column]  # Convert from rejection to approval
    
    fig = px.bar(
        approval_rates,
        x=sensitive_feature,
        y="Approval Rate",
        title=f"Loan Approval Rate by {sensitive_feature}",
        labels={"Approval Rate": "Approval Rate", sensitive_feature: sensitive_feature},
        color=sensitive_feature,
        color_discrete_sequence=["#66B2FF", "#FF9999"]
    )
    
    fig.update_layout(yaxis_range=[0, 1])
    
    st.plotly_chart(fig, use_container_width=True)
