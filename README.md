# EduPro Course Demand & Revenue Forecasting

This project provides a complete machine learning pipeline and interactive dashboard for the EduPro online learning platform. It predicts course demand (student enrollments) and forecasts platform revenue (course and category levels) to drive strategic, data-driven decisions.

## Objectives
- **Predict Enrollment Demand:** Forecast course registrations.
- **Forecast Platform Revenue:** Model revenue trends at course and category levels.
- **Analyze Feature Importances:** Identify top operational and financial success drivers.
- **Support Strategy Decisions:** Guide pricing structures, course launches, and instructor recruitment.

## Dataset
The project utilizes the following sheets from `data/raw/EduPro_Dataset.xlsx`:
- **Users:** Platform user demographic details.
- **Courses:** Title, category, duration, type (Free vs. Paid), level, and price.
- **Teachers:** Instructor ratings, experience, and expertise.
- **Transactions:** User registration records, payment amounts, and transaction dates.

## Technology Stack
- **Programming:** Python 3.14+
- **Data Engineering:** Pandas, NumPy, OpenPyXL
- **Visualization:** Plotly, Matplotlib, Seaborn, Streamlit
- **Machine Learning:** Scikit-Learn, XGBoost
- **Testing:** Pytest
- **Reporting:** ReportLab

## Project Folder Structure
```
EduPro-Predictive-Analytics/
│
├── README.md
├── requirements.txt
├── app.py
│
├── data/
│   ├── raw/
│   │   └── EduPro_Dataset.xlsx
│   └── processed/
│       ├── cleaned_courses.csv
│       ├── cleaned_teachers.csv
│       ├── cleaned_transactions.csv
│       ├── merged_dataset.csv
│       └── feature_dataset.csv
│
├── notebooks/
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Model_Training.ipynb
│   ├── 06_Model_Evaluation.ipynb
│   └── 07_Business_Insights.ipynb
│
├── src/
│   ├── data/
│   │   ├── cleaning.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   └── train.py
│   └── utils/
│       └── generate_reports.py
│
├── models/
│   ├── static_preprocessor.pkl
│   ├── static_enrollment_model.pkl
│   ├── static_revenue_model.pkl
│   └── static_features_list.pkl
│
├── reports/
│   ├── figures/
│   ├── model_evaluation_metrics.csv
│   ├── executive_summary.pdf
│   ├── business_report.pdf
│   └── model_report.pdf
│
├── docs/
│   └── research_paper.md
│
└── tests/
    ├── test_preprocessing.py
    └── test_predictions.py
```

## Model Results

### 1. Revenue Forecast Models (Monthly Level)
The best model is the **Ridge Regressor**, which fits the linear relationship between Course Price and Gross Revenue exceptionally well:
- **R² Score:** 0.8727
- **Mean Absolute Error (MAE):** $389.93
- **Root Mean Squared Error (RMSE):** $796.23

### 2. Enrollment Demand Models
- **R² Score:** ~ 0.0 (baseline mean predictor is mathematically optimal)
- **Analysis:** Monthly registrations per course are highly uniform across the platform (mean = 13.88, SD = 3.64). This low variance, random-walk behavior makes historical average rates the most robust forecast baseline.

---

## Getting Started

### 1. Installation
Clone this repository and install the dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Run Processing & Modeling Pipeline
To clean the data, engineer features, and train the models, run:
```bash
# Clean raw sheets
python src/data/cleaning.py

# Perform feature engineering & merging
python src/data/feature_engineering.py

# Train ML models & serialize output
python src/models/train.py
```

### 3. Run Automated Tests
Execute the unit test suite to verify code compliance:
```bash
python -m pytest tests/
```

### 4. Launch Streamlit Dashboard
Start the interactive dashboard locally:
```bash
streamlit run app.py
```

---

## Business Impact
1. **Pricing Decisions:** Recommends focusing on the **Medium Pricing Tier ($150-$350)**, which yields 3x higher average revenues per course than cheaper courses.
2. **Monetization Funnels:** Identifies that **64% of registrations are Free**. Introducing paid micro-credentials for free courses will capture significant revenue.
3. **Instructor Recruitment:** Highlights that **instructors with 6-10 years of experience** generate 42% higher platform revenues on average.
