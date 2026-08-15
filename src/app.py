import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# LOAD MODEL + SCALER
# =========================================================

@st.cache_resource
def load_model():
    model = joblib.load("../models/model.pkl")
    scaler = joblib.load("../models/scaler.pkl")
    return model, scaler


model, scaler = load_model()


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f5f7fb;
    }

    /* Header */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 18px;
        margin-bottom: 35px;
    }

    /* Cards */
    .card {
        background-color: white;
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }

    /* Prediction */
    .prediction-card {
        background-color: white;
        padding: 35px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 5px 25px rgba(0,0,0,0.08);
        margin-top: 25px;
    }

    .prediction-title {
        font-size: 28px;
        font-weight: 700;
    }

    .prediction-text {
        font-size: 22px;
        font-weight: 600;
        margin-top: 10px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📊 Employee Attrition Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict whether an employee is likely to leave the organization'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Employee Details")

    st.markdown("Enter employee information below.")

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=70,
        value=30
    )

    daily_rate = st.number_input(
        "Daily Rate",
        min_value=100,
        max_value=1500,
        value=800
    )

    distance = st.number_input(
        "Distance From Home",
        min_value=1,
        max_value=30,
        value=5
    )

    education = st.slider(
        "Education",
        1, 5, 3
    )

    environment = st.slider(
        "Environment Satisfaction",
        1, 4, 3
    )

    hourly_rate = st.number_input(
        "Hourly Rate",
        min_value=30,
        max_value=100,
        value=70
    )

    job_involvement = st.slider(
        "Job Involvement",
        1, 4, 3
    )

    job_level = st.slider(
        "Job Level",
        1, 5, 2
    )

    job_satisfaction = st.slider(
        "Job Satisfaction",
        1, 4, 3
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=20000,
        value=5000
    )

    monthly_rate = st.number_input(
        "Monthly Rate",
        min_value=2000,
        max_value=30000,
        value=15000
    )

    num_companies = st.number_input(
        "Number of Companies Worked",
        min_value=0,
        max_value=10,
        value=2
    )

    overtime = st.selectbox(
        "OverTime",
        ["No", "Yes"]
    )

    salary_hike = st.number_input(
        "Percent Salary Hike",
        min_value=10,
        max_value=30,
        value=15
    )

    performance = st.slider(
        "Performance Rating",
        1, 4, 3
    )

    relationship = st.slider(
        "Relationship Satisfaction",
        1, 4, 3
    )

    stock_option = st.slider(
        "Stock Option Level",
        0, 3, 1
    )

    total_working_years = st.number_input(
        "Total Working Years",
        min_value=0,
        max_value=40,
        value=8
    )

    training = st.slider(
        "Training Times Last Year",
        0, 10, 3
    )

    work_life = st.slider(
        "Work Life Balance",
        1, 4, 3
    )

    years_company = st.number_input(
        "Years At Company",
        min_value=0,
        max_value=40,
        value=5
    )

    years_role = st.number_input(
        "Years In Current Role",
        min_value=0,
        max_value=20,
        value=3
    )

    years_promotion = st.number_input(
        "Years Since Last Promotion",
        min_value=0,
        max_value=15,
        value=1
    )

    years_manager = st.number_input(
        "Years With Current Manager",
        min_value=0,
        max_value=20,
        value=3
    )


# =========================================================
# MAIN PAGE
# =========================================================

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("👤 Personal Information")

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

    st.markdown("</div>", unsafe_allow_html=True)


with col2:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("💼 Professional Information")

    business_travel = st.selectbox(
        "Business Travel",
        [
            "Non-Travel",
            "Travel_Rarely",
            "Travel_Frequently"
        ]
    )

    department = st.selectbox(
        "Department",
        [
            "Research & Development",
            "Sales"
        ]
    )

    education_field = st.selectbox(
        "Education Field",
        [
            "Life Sciences",
            "Marketing",
            "Medical",
            "Other",
            "Technical Degree"
        ]
    )

    job_role = st.selectbox(
        "Job Role",
        [
            "Human Resources",
            "Laboratory Technician",
            "Manager",
            "Manufacturing Director",
            "Research Director",
            "Research Scientist",
            "Sales Executive",
            "Sales Representative"
        ]
    )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# CREATE MODEL INPUT
# =========================================================

if st.button(
    "🔮 Predict Employee Attrition",
    use_container_width=True
):

    # ---------------------------------------------
    # Encode categorical variables
    # ---------------------------------------------

    employee = pd.DataFrame([{

        "Age": age,
        "DailyRate": daily_rate,
        "DistanceFromHome": distance,
        "Education": education,
        "EnvironmentSatisfaction": environment,

        "Gender": 1 if gender == "Male" else 0,

        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobSatisfaction": job_satisfaction,

        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies,

        "OverTime": 1 if overtime == "Yes" else 0,

        "PercentSalaryHike": salary_hike,
        "PerformanceRating": performance,
        "RelationshipSatisfaction": relationship,
        "StockOptionLevel": stock_option,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training,
        "WorkLifeBalance": work_life,
        "YearsAtCompany": years_company,
        "YearsInCurrentRole": years_role,
        "YearsSinceLastPromotion": years_promotion,
        "YearsWithCurrManager": years_manager,

        # Business Travel
        "BusinessTravel_Travel_Frequently":
            1 if business_travel == "Travel_Frequently" else 0,

        "BusinessTravel_Travel_Rarely":
            1 if business_travel == "Travel_Rarely" else 0,

        # Department
        "Department_Research & Development":
            1 if department == "Research & Development" else 0,

        "Department_Sales":
            1 if department == "Sales" else 0,

        # Education Field
        "EducationField_Life Sciences":
            1 if education_field == "Life Sciences" else 0,

        "EducationField_Marketing":
            1 if education_field == "Marketing" else 0,

        "EducationField_Medical":
            1 if education_field == "Medical" else 0,

        "EducationField_Other":
            1 if education_field == "Other" else 0,

        "EducationField_Technical Degree":
            1 if education_field == "Technical Degree" else 0,

        # Job Role
        "JobRole_Human Resources":
            1 if job_role == "Human Resources" else 0,

        "JobRole_Laboratory Technician":
            1 if job_role == "Laboratory Technician" else 0,

        "JobRole_Manager":
            1 if job_role == "Manager" else 0,

        "JobRole_Manufacturing Director":
            1 if job_role == "Manufacturing Director" else 0,

        "JobRole_Research Director":
            1 if job_role == "Research Director" else 0,

        "JobRole_Research Scientist":
            1 if job_role == "Research Scientist" else 0,

        "JobRole_Sales Executive":
            1 if job_role == "Sales Executive" else 0,

        "JobRole_Sales Representative":
            1 if job_role == "Sales Representative" else 0,

        # Marital Status
        "MaritalStatus_Married":
            1 if marital_status == "Married" else 0,

        "MaritalStatus_Single":
            1 if marital_status == "Single" else 0
    }])


    # =====================================================
    # ENSURE CORRECT FEATURE ORDER
    # =====================================================

    feature_names = [
        'Age',
        'DailyRate',
        'DistanceFromHome',
        'Education',
        'EnvironmentSatisfaction',
        'Gender',
        'HourlyRate',
        'JobInvolvement',
        'JobLevel',
        'JobSatisfaction',
        'MonthlyIncome',
        'MonthlyRate',
        'NumCompaniesWorked',
        'OverTime',
        'PercentSalaryHike',
        'PerformanceRating',
        'RelationshipSatisfaction',
        'StockOptionLevel',
        'TotalWorkingYears',
        'TrainingTimesLastYear',
        'WorkLifeBalance',
        'YearsAtCompany',
        'YearsInCurrentRole',
        'YearsSinceLastPromotion',
        'YearsWithCurrManager',
        'BusinessTravel_Travel_Frequently',
        'BusinessTravel_Travel_Rarely',
        'Department_Research & Development',
        'Department_Sales',
        'EducationField_Life Sciences',
        'EducationField_Marketing',
        'EducationField_Medical',
        'EducationField_Other',
        'EducationField_Technical Degree',
        'JobRole_Human Resources',
        'JobRole_Laboratory Technician',
        'JobRole_Manager',
        'JobRole_Manufacturing Director',
        'JobRole_Research Director',
        'JobRole_Research Scientist',
        'JobRole_Sales Executive',
        'JobRole_Sales Representative',
        'MaritalStatus_Married',
        'MaritalStatus_Single'
    ]

    employee = employee[feature_names]


    # =====================================================
    # SCALE
    # =====================================================

    employee_scaled = scaler.transform(employee)


    # =====================================================
    # PREDICTION
    # =====================================================

    prediction = model.predict(employee_scaled)[0]

    probability = model.predict_proba(employee_scaled)[0]

    stay_probability = probability[0]
    leave_probability = probability[1]


    # =====================================================
    # RESULT
    # =====================================================

    if prediction == 1:

        st.markdown(
            """<div class="prediction-card">
<div class="prediction-title" style="color: #ef4444;">🔴 High Attrition Risk</div>
<div class="prediction-text">This employee is likely to leave.</div>
</div>""",
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """<div class="prediction-card">
<div class="prediction-title" style="color: #10b981;">🟢 Low Attrition Risk</div>
<div class="prediction-text">This employee is likely to stay.</div>
</div>""",
            unsafe_allow_html=True
        )


    # =====================================================
    # PROBABILITY METRICS
    # =====================================================

    st.markdown("### 📈 Prediction Confidence")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Stay Probability",
            f"{stay_probability:.1%}"
        )

        st.progress(stay_probability)

    with col2:

        st.metric(
            "Leave Probability",
            f"{leave_probability:.1%}"
        )

        st.progress(leave_probability)


    # =====================================================
    # SUMMARY
    # =====================================================

    st.markdown("### 📋 Employee Summary")

    summary = pd.DataFrame({
        "Attribute": [
            "Age",
            "Job Level",
            "Monthly Income",
            "Job Satisfaction",
            "Environment Satisfaction",
            "OverTime",
            "Years At Company",
            "Business Travel",
            "Job Role"
        ],

        "Value": [
            age,
            job_level,
            f"${monthly_income:,}",
            job_satisfaction,
            environment,
            overtime,
            years_company,
            business_travel,
            job_role
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )