import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(page_title="HR KPI Dashboard", layout="wide")

# ============================
# LOAD DATA
# ============================
df = pd.read_csv("hrkpidata.csv")
df.index = range(1, len(df) + 1)

# ============================
# MAPPINGS (CONSISTENT)
# ============================
gender_map = {"Male": 1, "Female": 0}
promotion_map = {"Yes": 1, "No": 0}
department_map = {
    "HR": 0,
    "IT": 1,
    "Finance": 2,
    "Sales": 3,
    "Marketing": 4,
    "Operations": 5
}

df["Gender"] = df["Gender"].map(gender_map)
df["Promotion History"] = df["Promotion History"].map(promotion_map)
df["Department"] = df["Department"].map(department_map)

df = df.fillna(0)

# ============================
# KPI SCORE (BASELINE)
# ============================
df["KPI Score"] = (
    df["Performance Rating"] * 0.4 +
    df["Engagement Score"] * 0.3 +
    df["Attendance (%)"] * 0.2 +
    df["Number of Training"] * 0.1
)

# ============================
# FEATURES / TARGETS
# ============================
features = [
    "Gender",
    "Department",
    "Engagement Score",
    "Attendance (%)",
    "Number of Training",
    "Total Experience",
    "Years at Company",
    "Promotion History"
]

X = df[features]
y_attrition = df["Attrition"]
y_performance = df["Performance Rating"]

# ============================
# TRAIN TEST SPLIT (FIXED CONSISTENCY)
# ============================
X_train, X_test, y_train_attr, y_test_attr = train_test_split(
    X, y_attrition, test_size=0.2, random_state=42
)

X_train2, X_test2, y_train_perf, y_test_perf = train_test_split(
    X, y_performance, test_size=0.2, random_state=42
)

# ============================
# MODELS
# ============================
attr_model = RandomForestClassifier(random_state=42)
perf_model = RandomForestClassifier(random_state=42)

attr_model.fit(X_train, y_train_attr)
perf_model.fit(X_train2, y_train_perf)

# ============================
# SIDEBAR NAVIGATION
# ============================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Predictor", "About", "Profile"])

# ============================
# DASHBOARD
# ============================
if page == "Dashboard":

    st.title("📊 HR KPI Dashboard")

    st.subheader("Employee Search")
    search = st.text_input("Search Employee Name")

    if search:
        filtered_df = df[df["Employee Name"].str.contains(search, case=False, na=False)]
    else:
        filtered_df = df

    display_df = filtered_df.copy()
    display_df["Attrition Status"] = display_df["Attrition"].map({1: "High Risk", 0: "Low Risk"})

    st.dataframe(display_df)

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("Avg KPI Score", round(df["KPI Score"].mean(), 2))
    col2.metric("Attrition Rate", f"{df['Attrition'].mean() * 100:.1f}%")
    col3.metric("Avg Performance", round(df["Performance Rating"].mean(), 2))

    st.divider()

    st.subheader("Visual Insights")
    st.bar_chart(df["Attrition"].value_counts())
    st.bar_chart(df["Department"].value_counts())
    st.line_chart(df["KPI Score"])

# ============================
# PREDICTOR
# ============================
elif page == "Predictor":

    st.title("🎯 HR KPI Predictor")

    employee_name = st.selectbox("Select Employee", df["Employee Name"].unique())
    emp = df[df["Employee Name"] == employee_name].iloc[0]

    st.subheader("Employee Profile")
    st.write(f"Department: {employee_name}")
    st.write(f"Experience: {emp['Total Experience']} years")
    st.write(f"Years at Company: {emp['Years at Company']}")

    st.divider()

    st.subheader("Prediction Inputs")

    gender = st.selectbox("Gender", list(gender_map.keys()), index=int(emp["Gender"]))
    gender = gender_map[gender]

    department = st.selectbox("Department", list(department_map.keys()), index=int(emp["Department"]))
    department = department_map[department]

    engagement = st.slider("Engagement Score", 50, 100, int(emp["Engagement Score"]))
    attendance = st.slider("Attendance (%)", 80, 100, int(emp["Attendance (%)"]))
    training = st.selectbox("Number of Training", [0, 1, 2, 3, 4, 5], index=int(emp["Number of Training"]))

    experience = st.number_input("Total Experience", 0, 40, int(emp["Total Experience"]))
    years_company = st.number_input("Years at Company", 0, 40, int(emp["Years at Company"]))

    promotion = st.selectbox("Promotion History", list(promotion_map.keys()), index=int(emp["Promotion History"]))
    promotion = promotion_map[promotion]

    input_data = pd.DataFrame([{
        "Gender": gender,
        "Department": department,
        "Engagement Score": engagement,
        "Attendance (%)": attendance,
        "Number of Training": training,
        "Total Experience": experience,
        "Years at Company": years_company,
        "Promotion History": promotion
    }])

    if st.button("Predict"):

        attr_pred = attr_model.predict(input_data)[0]
        perf_pred = perf_model.predict(input_data)[0]

        kpi_score = (
            perf_pred * 0.4 +
            engagement * 0.3 +
            attendance * 0.2 +
            training * 0.1
        )

        st.subheader("Results")

        if attr_pred == 1:
            st.error("⚠️ High Attrition Risk")
        else:
            st.success("Low Attrition Risk")

        st.info(f"⭐ Performance Rating: {perf_pred}")
        st.metric("📊 KPI Score", round(kpi_score, 2))

# ============================
# ABOUT
# ============================
elif page == "About":
    st.title("ℹ️ About")
    st.write("HR KPI Machine Learning Dashboard")
    st.write("Predicts Attrition, Performance Rating, and KPI Score")

# ============================
# PROFILE
# ============================
elif page == "Profile":
    st.title("👤 Profile")
    st.write("Vivian Iyaha")
    st.write("HR Analytics | Machine Learning | AI Enthusiast")    X, y_attrition, test_size=0.2, random_state=42
)

X_train2, X_test2, y_train_perf, y_test_perf = train_test_split(
    X, y_performance, test_size=0.2, random_state=42
)

attr_model = RandomForestClassifier(random_state=42)
perf_model = RandomForestClassifier(random_state=42)

attr_model.fit(X_train, y_train_attr)
perf_model.fit(X_train2, y_train_perf)


# ============================
# SIDEBAR NAVIGATION (ONLY ONCE)
# ============================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Predictor", "About", "Profile"],
    key="main_nav"
)


# ============================
# DASHBOARD
# ============================
if page == "Dashboard":

    st.subheader("Employee Search")

    search = st.text_input("Search Employee Name")

    if search:
        filtered_df = df[df["Employee Name"].str.contains(search, case=False, na=False)]
    else:
        filtered_df = df

    display_df = filtered_df.copy()
    display_df["Attrition Status"] = display_df["Attrition"].map({1: "High Risk", 0: "Low Risk"})

    st.dataframe(display_df)

    st.divider()

st.title("📊 HR KPI Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Avg KPI Score", round(df["KPI Score"].mean(), 2))
    col2.metric("Attrition Rate", f"{df['Attrition'].mean() * 100:.1f}%")
    col3.metric("Avg Performance", round(df["Performance Rating"].mean(), 2))

    st.divider()

    st.subheader("Visual Insights")
    st.bar_chart(df["Attrition"].value_counts())
    st.bar_chart(df["Department"].value_counts())
    st.line_chart(df["KPI Score"])


# ============================
# PREDICTOR
# ============================
elif page == "Predictor":

    st.title("🎯 HR KPI Predictor")

    # Employee selection
    employee_name = st.selectbox("Select Employee", df["Employee Name"].unique())

    emp = df[df["Employee Name"] == employee_name].iloc[0]

    st.subheader("Employee Profile")
    st.write(f"Department: {emp['Department']}")
    st.write(f"Experience: {emp['Total Experience']} years")
    st.write(f"Years at Company: {emp['Years at Company']}")

    st.divider()

    st.subheader("Prediction Inputs")

    gender = st.selectbox("Gender", ["Male", "Female"], index=int(emp["Gender"]))
    gender = 1 if gender == "Male" else 0

    department = st.selectbox(
        "Department",
        ["HR", "IT", "Finance", "Sales", "Marketing", "Operations"],
        index=int(emp["Department"])
    )

    department = {
        "HR": 0, "IT": 1, "Finance": 2,
        "Sales": 3, "Marketing": 4, "Operations": 5
    }[department]

    engagement = st.slider("Engagement Score", 50, 100, int(emp["Engagement Score"]))
    attendance = st.slider("Attendance (%)", 80, 100, int(emp["Attendance (%)"]))
    training = st.selectbox("Number of Training", [0, 1, 2, 3, 4, 5], index=int(emp["Number of Training"]))

    experience = st.number_input("Total Experience", 0, 40, int(emp["Total Experience"]))
    years_company = st.number_input("Years at Company", 0, 40, int(emp["Years at Company"]))

    promotion = st.selectbox("Promotion History", ["Yes", "No"], index=int(emp["Promotion History"]))
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
    st.title("ℹ️ About")
    st.write("HR KPI Machine Learning Dashboard")
    st.write("Predicts Attrition, Performance Rating, and KPI Score")


# ============================
# PROFILE
# ============================
elif page == "Profile":
    st.title("👤 Profile")
    st.write("Vivian Iyaha")
    st.write("HR Analytics | Machine Learning | AI Enthusiast")
