import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import plotly.express as px

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Employee KPI Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f8f9fc;
}

.stButton>button {
    background-color: #0d6efd;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.6rem 1rem;
    font-weight: 600;
}

div[data-testid="metric-container"] {
    background-color: white;
    border: 1px solid #e6e6e6;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.title("📊 Employee KPI Performance Dashboard")
st.markdown("Professional KPI monitoring and analytics system")

# =========================================================
# STORAGE FOLDER
# =========================================================

BASE_DIR = "performance_data"
Path(BASE_DIR).mkdir(exist_ok=True)

# =========================================================
# REQUIRED COLUMNS
# =========================================================

required_columns = [
    "Name",
    "Employment Status",
    "Designation",
    "Department",
    "KPI 1 (Write out the KPI)",
    "Was KPI 1 completed?",
    "KPI 2 (Write out the KPI)",
    "Was KPI 2 completed?",
    "KPI 3 (Write out the kpi)",
    "Was KPI 3 completed?",
    "Challenges faced during the week"
]

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def calculate_score(row):
    score = 0
    total = 3

    kpi1 = str(row["Was KPI 1 completed?"]).strip().lower()
    kpi2 = str(row["Was KPI 2 completed?"]).strip().lower()
    kpi3 = str(row["Was KPI 3 completed?"]).strip().lower()

    if kpi1 in ["yes", "true", "completed"]:
        score += 1

    if kpi2 in ["yes", "true", "completed"]:
        score += 1

    if kpi3 in ["yes", "true", "completed"]:
        score += 1

    return round((score / total) * 100, 2)


def performance_category(score):
    if score >= 80:
        return "High Performer"
    elif score < 50:
        return "Low Performer"
    else:
        return "Average Performer"


def save_uploaded_file(df, filename):
    now = datetime.now()

    month_folder = now.strftime("%Y-%m")
    month_path = os.path.join(BASE_DIR, month_folder)

    Path(month_path).mkdir(parents=True, exist_ok=True)

    save_path = os.path.join(month_path, filename)

    df.to_csv(save_path, index=False)

    return save_path


def load_all_data():
    all_data = []

    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)

                temp_df = pd.read_csv(file_path)
                all_data.append(temp_df)

    if all_data:
        return pd.concat(all_data, ignore_index=True)

    return pd.DataFrame()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("📁 Weekly KPI Report")

# Folder where CSV files are stored
data_folder = "."

# Get all CSV files
csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

if csv_files:

    # Dropdown select button
    selected_file = st.sidebar.selectbox(
        "Select Weekly KPI File",
        csv_files
    )

    # Load selected file button
    load_file = st.sidebar.button("Open Selected File")

    # =========================================================
    # FILE LOAD PROCESSING
    # =========================================================

    if load_file:
        try:
            file_path = os.path.join(data_folder, selected_file)

            df = pd.read_csv(file_path)

            # Validate columns
            missing_cols = [
                col for col in required_columns
                if col not in df.columns
            ]

            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
            else:
                st.success(f"{selected_file} loaded successfully!")
                st.dataframe(df)

        except Exception as e:
            st.error(f"Error loading file: {e}")

else:
    st.sidebar.warning("No CSV files found.")
# =========================================================
# LOAD ALL SAVED DATA
# =========================================================

data = load_all_data()

# =========================================================
# DASHBOARD
# =========================================================

if not data.empty:

    st.markdown("---")
    st.header("📈 Manager Dashboard")

    # Metrics
    total_employees = data["Name"].nunique()

    avg_score = round(
        data["KPI Score"].mean(),
        2
    )

    high_performers = len(
        data[data["KPI Score"] >= 80]
    )

    low_performers = len(
        data[data["KPI Score"] < 50]
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Employees", total_employees)

    with col2:
        st.metric(
            "Average KPI Score",
            f"{avg_score}%"
        )

    with col3:
        st.metric(
            "High Performers",
            high_performers
        )

    with col4:
        st.metric(
            "Low Performers",
            low_performers
        )

    # =====================================================
    # FILTERS
    # =====================================================

    st.sidebar.header("🔍 Filters")

    departments = st.sidebar.multiselect(
        "Select Department",
        options=data["Department"].unique(),
        default=data["Department"].unique()
    )

    months = st.sidebar.multiselect(
        "Select Month",
        options=data["Month"].unique(),
        default=data["Month"].unique()
    )

    filtered_data = data[
        (data["Department"].isin(departments)) &
        (data["Month"].isin(months))
    ]

    # High Performers
    st.subheader("🏆 High Performers")

    high_df = filtered_data[
        filtered_data["KPI Score"] >= 80
    ]

    st.dataframe(
        high_df[
            [
                "Name",
                "Department",
                "Designation",
                "KPI Score",
                "Week",
                "Month"
            ]
        ],
        use_container_width=True
    )

    # Low Performers
    st.subheader("⚠️ Low Performers")

    low_df = filtered_data[
        filtered_data["KPI Score"] < 50
    ]

    st.dataframe(
        low_df[
            [
                "Name",
                "Department",
                "Designation",
                "KPI Score",
                "Week",
                "Month"
            ]
        ],
        use_container_width=True
    )

    # Department Performance
    st.subheader("🏢 Department Performance")

    dept_df = (
        filtered_data
        .groupby("Department")["KPI Score"]
        .mean()
        .reset_index()
    )

    dept_df["KPI Score"] = dept_df[
        "KPI Score"
    ].round(2)

    fig_dept = px.bar(
        dept_df,
        x="Department",
        y="KPI Score",
        color="KPI Score",
        text="KPI Score",
        color_continuous_scale="Blues",
        title="Average KPI Score by Department"
    )

    st.plotly_chart(
        fig_dept,
        use_container_width=True
    )

    # Performance Distribution
    st.subheader("📊 Performance Distribution")

    pie_fig = px.pie(
        filtered_data,
        names="Performance Category",
        title="Employee Performance Distribution"
    )

    st.plotly_chart(
        pie_fig,
        use_container_width=True
    )

    # All KPI Data
    st.subheader("📝 All KPI Records")

    st.dataframe(
        filtered_data,
        use_container_width=True
    )

    # Download Report
    report_csv = filtered_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download KPI Report",
        data=report_csv,
        file_name="employee_kpi_report.csv",
        mime="text/csv"
    )

else:
    st.info(
        "Click 'Open weeklyKpi.csv' to begin."
    )

# =========================================================
# TEMPLATE DOWNLOAD
# =========================================================

st.markdown("---")

st.subheader("📥 Download KPI Template")

template_df = pd.DataFrame(
    columns=required_columns
)

template_csv = template_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download CSV Template",
    data=template_csv,
    file_name="kpi_template.csv",
    mime="text/csv"
)
