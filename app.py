import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ============================
# LOAD DATA
# ============================
df = pd.read_csv('hrkpidata.csv')

# Add Serial Number (1–50)
df.insert(0, 'S/N', range(1, len(df) + 1))


# ============================
# SAFE DATA PREPROCESSING
# ============================

df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Promotion History'] = df['Promotion History'].map({'Yes': 1, 'No': 0})

df['Department'] = df['Department'].map({
    'HR': 0,
    'IT': 1,
    'Finance': 2,
    'Sales': 3,
    'Marketing': 4,
    'Operations': 5
})

df['Reporting Manager'] = df['Reporting Manager'].astype('category').cat.codes


# Fill any missing values (VERY IMPORTANT FIX)
df = df.fillna(0)


# ============================
# KPI SCORE
# ============================
df['KPI Score'] = (
    df['Performance Rating'] * 0.4 +
    df['Engagement Score'] * 0.3 +
    df['Attendance (%)'] * 0.2 +
    df['Number of Training'] * 0.1
)


# ============================
# FEATURES & TARGETS
# ============================
X = df[['Gender', 'Department', 'Engagement Score',
        'Attendance (%)', 'Number of Training',
        'Total Experience', 'Years at Company',
        'Promotion History']]

y_attrition = df['Attrition']
y_performance = df['Performance Rating']


# ============================
# TRAIN MODELS
# ============================
X_train, X_test, y_train_attr, y_test_attr = train_test_split(
    X, y_attrition, test_size=0.2, random_state=42
)

X_train2, X_test2, y_train_perf, y_test_perf = train_test_split(
    X, y_performance, test_size=0.2, random_state=42
)

attr_model = RandomForestClassifier(random_state=42)
perf_model = RandomForestClassifier(random_state=42)

attr_model.fit(X_train, y_train_attr)
perf_model.fit(X_train2, y_train_perf)


# ============================
# NAVIGATION
# ============================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Predictor", "About", "Profile"])


# ============================
# DASHBOARD
# ============================
if page == "Dashboard":

    st.title("HR KPI Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Avg KPI Score", round(df["KPI Score"].mean(), 2))
    col2.metric("Attrition Rate", f"{df['Attrition'].mean() * 100:.1f}%")
    col3.metric("Avg Performance", round(df["Performance Rating"].mean(), 2))

    st.subheader("Visual Insights")
    st.bar_chart(df["Attrition"].value_counts())
    st.bar_chart(df["Department"].value_counts())
    st.line_chart(df["KPI Score"])

    st.subheader("Employee Dataset")

    search = st.text_input("🔍 Search Employee Name")

    if search:
        filtered_df = df[df["Employee Name"].str.contains(search, case=False, na=False)]
    else:
        filtered_df = df

    display_df = filtered_df.copy()
    display_df["Attrition"] = display_df["Attrition"].map({1: "High Risk", 0: "Low Risk"})

    def highlight(val):
        return (
            "background-color: red; color: white"
            if val == "High Risk"
            else "background-color: green; color: white"
        )

    st.dataframe(display_df.style.map(highlight, subset=["Attrition"]))


# ============================
# PREDICTOR
# ============================
elif page == "Predictor":

    st.title("HR KPI Predictor")

    gender = st.selectbox("Gender", ["Male", "Female"])
    gender = 1 if gender == "Male" else 0

    department = st.selectbox(
        "Department",
        ["HR", "IT", "Finance", "Sales", "Marketing", "Operations"]
    )

    department = {
        "HR": 0, "IT": 1, "Finance": 2,
        "Sales": 3, "Marketing": 4, "Operations": 5
    }[department]

    engagement = st.slider("Engagement Score", 50, 100, 75)
    attendance = st.slider("Attendance (%)", 80, 100, 95)
    training = st.selectbox("Number of Training", [0, 1, 2, 3, 4, 5])

    experience = st.number_input("Total Experience", 0, 40, 5)
    years_company = st.number_input("Years at Company", 0, 40, 3)

    promotion = st.selectbox("Promotion History", ["Yes", "No"])
    promotion = 1 if promotion == "Yes" else 0

    input_data = pd.DataFrame({
        "Gender": [gender],
        "Department": [department],
        "Engagement Score": [engagement],
        "Attendance (%)": [attendance],
        "Number of Training": [training],
        "Total Experience": [experience],
        "Years at Company": [years_company],
        "Promotion History": [promotion]
    })

    if st.button("Predict"):

        attr_pred = attr_model.predict(input_data)
        perf_pred = perf_model.predict(input_data)

        kpi_score = (
            perf_pred[0] * 0.4 +
            engagement * 0.3 +
            attendance * 0.2 +
            training * 0.1
        )

        st.subheader("Results")

        if attr_pred[0] == 1:
            st.error("⚠️ High Attrition Risk")
        else:
            st.success("Low Attrition Risk")

        st.info(f"⭐ Performance Rating: {perf_pred[0]}")
        st.metric("📊 KPI Score", round(kpi_score, 2))


# ============================
# ABOUT
# ============================
elif page == "About":
    st.title("About")
    st.write("HR KPI Machine Learning App")
    st.write("Predicts Attrition, Performance, KPI Score")


# ============================
# PROFILE
# ============================
elif page == "Profile":
    st.title("Profile")
    st.write("Vivian Iyaha - HR Analytics & AI Enthusiast")st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Predictor", "About", "Profile"])


# ============================
# DASHBOARD
# ============================
if page == "Dashboard":

    st.title("HR KPI Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Avg KPI Score", round(df["KPI Score"].mean(), 2))
    col2.metric("Attrition Rate", f"{df['Attrition'].mean() * 100:.1f}%")
    col3.metric("Avg Performance", round(df["Performance Rating"].mean(), 2))

    st.subheader("Visual Insights")
    st.bar_chart(df["Attrition"].value_counts())
    st.bar_chart(df["Department"].value_counts())
    st.line_chart(df["KPI Score"])

    st.subheader("Employee Dataset")

    search = st.text_input("🔍 Search Employee Name")

    if search:
        filtered_df = df[df["Employee Name"].str.contains(search, case=False)]
    else:
        filtered_df = df

    display_df = filtered_df.copy()
    display_df["Attrition"] = display_df["Attrition"].map({1: "High Risk", 0: "Low Risk"})

    def highlight(val):
        return "background-color: red; color: white" if val == "High Risk" else "background-color: green; color: white"

    st.dataframe(display_df.style.map(highlight, subset=["Attrition"]))


# ============================
# PREDICTOR
# ============================
elif page == "Predictor":

    st.title("HR KPI Predictor")

    gender = st.selectbox("Gender", ["Male", "Female"])
    gender = 1 if gender == "Male" else 0

    department = st.selectbox(
        "Department",
        ["HR", "IT", "Finance", "Sales", "Marketing", "Operations"]
    )
    department = {
        "HR": 0, "IT": 1, "Finance": 2,
        "Sales": 3, "Marketing": 4, "Operations": 5
    }[department]

    engagement = st.slider("Engagement Score", 50, 100, 75)
    attendance = st.slider("Attendance (%)", 80, 100, 95)
    training = st.selectbox("Number of Training", [0, 1, 2, 3, 4, 5])

    experience = st.number_input("Total Experience", 0, 15, 5)
    years_company = st.number_input("Years at Company", 0, 10, 3)

    promotion = st.selectbox("Promotion History", ["Yes", "No"])
    promotion = 1 if promotion == "Yes" else 0

    input_data = pd.DataFrame({
        "Gender": [gender],
        "Department": [department],
        "Engagement Score": [engagement],
        "Attendance (%)": [attendance],
        "Number of Training": [training],
        "Total Experience": [experience],
        "Years at Company": [years_company],
        "Promotion History": [promotion]
    })

    if st.button("Predict"):

        attr_pred = attr_model.predict(input_data)
        perf_pred = perf_model.predict(input_data)

        kpi_score = (
            perf_pred[0] * 0.4 +
            engagement * 0.3 +
            attendance * 0.2 +
            training * 0.1
        )

        st.subheader("Results")

        if attr_pred[0] == 1:
            st.error("⚠️ High Attrition Risk")
        else:
            st.success("Low Attrition Risk")

        st.info(f"⭐ Performance Rating: {perf_pred[0]}")
        st.metric("📊 KPI Score", round(kpi_score, 2))


# ============================
# ABOUT
# ============================
elif page == "About":
    st.title("About")
    st.write("HR KPI Machine Learning App")
    st.write("Predicts Attrition, Performance, KPI Score")


# ============================
# PROFILE
# ============================
elif page == "Profile":
    st.title("Profile")
    st.write("Vivian Iyaha - HR Analytics & AI Enthusiast")
