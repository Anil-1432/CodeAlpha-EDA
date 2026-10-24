import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(
    page_title="CodeAlpha - EDA Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title & Overview
st.title("📊 Exploratory Data Analysis (EDA) Engine")
st.markdown("Upload any CSV dataset to automatically compute statistics, clean data, and visualize distributions.")

# Sidebar - Data Upload
st.sidebar.header("1. Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read Dataset
    df = pd.read_csv(uploaded_file)

    # Tabs for Workflow Navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Data Overview", 
        "🧹 Missing Values & Cleaning", 
        "📈 Univariate Analysis", 
        "🔗 Bivariate & Correlation"
    ])

    # ------------------ TAB 1: DATA OVERVIEW ------------------
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Duplicate Rows", df.duplicated().sum())

        st.subheader("Column Data Types")
        dtype_df = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Non-Null Count": df.notnull().sum().values
        })
        st.dataframe(dtype_df, use_container_width=True)

        st.subheader("Statistical Summary")
        st.dataframe(df.describe(include='all').transpose().fillna("-"), use_container_width=True)

    # ------------------ TAB 2: MISSING VALUES ------------------
    with tab2:
        st.subheader("Missing Value Audit")
        null_counts = df.isnull().sum()
        null_pct = (null_counts / len(df)) * 100
        null_df = pd.DataFrame({
            "Missing Count": null_counts,
            "Percentage (%)": null_pct.round(2)
        }).sort_values(by="Missing Count", ascending=False)

        st.dataframe(null_df, use_container_width=True)

        # Plot Missing Values
        if null_counts.sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(x=null_df.index, y=null_df["Missing Count"], ax=ax, palette="viridis")
            plt.xticks(rotation=45, ha='right')
            plt.title("Missing Values per Feature")
            st.pyplot(fig)
        else:
            st.success("No missing values found in the uploaded dataset!")

    # ------------------ TAB 3: UNIVARIATE ANALYSIS ------------------
    with tab3:
        st.subheader("Single Feature Distribution")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        col_type = st.radio("Select Variable Type:", ["Numeric Features", "Categorical Features"], horizontal=True)

        if col_type == "Numeric Features" and numeric_cols:
            selected_num = st.selectbox("Select Numeric Column:", numeric_cols)
            
            fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [0.2, 0.8]})
            sns.boxplot(x=df[selected_num], ax=ax_box, color="skyblue")
            sns.histplot(df[selected_num], kde=True, ax=ax_hist, color="steelblue")
            ax_box.set(xlabel='')
            st.pyplot(fig)

        elif col_type == "Categorical Features" and categorical_cols:
            selected_cat = st.selectbox("Select Categorical Column:", categorical_cols)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            df[selected_cat].value_counts().head(10).plot(kind='bar', ax=ax, color='teal')
            plt.title(f"Top Categories in {selected_cat}")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("No matching columns found for the selected type.")

    # ------------------ TAB 4: BIVARIATE & CORRELATION ------------------
    with tab4:
        st.subheader("Correlation Heatmap")
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.info("Need at least two numerical columns to compute correlations.")

        st.subheader("Scatter Plot Comparison")
        if len(numeric_cols) >= 2:
            col_x = st.selectbox("X-Axis Feature:", numeric_cols, index=0)
            col_y = st.selectbox("Y-Axis Feature:", numeric_cols, index=min(1, len(numeric_cols)-1))
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.scatterplot(data=df, x=col_x, y=col_y, ax=ax, color="indigo")
            st.pyplot(fig)

else:
    st.info("Please upload a CSV file in the sidebar to run the automated EDA pipeline.")