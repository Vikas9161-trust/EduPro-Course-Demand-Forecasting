# Research Paper: Predictive Modeling for Course Demand and Revenue Forecasting on EduPro

## Abstract
This paper details the development of a predictive modeling framework for the EduPro online learning platform. Using machine learning regression models, we predict future course registrations and monthly revenues to support data-driven decision-making. We evaluate baseline estimators (Linear, Ridge, Lasso) and advanced models (Random Forest, Gradient Boosting, XGBoost). Results show that revenue can be forecasted with high accuracy (R² = 0.87) primarily driven by Course Price, while course enrollments follow a mostly uniform, low-variance distribution. We conclude with strategic pricing, hiring, and expansion recommendations.

---

## 1. Introduction
In the rapidly growing EdTech sector, online learning platforms like EduPro face intense competition. Optimization of marketing spend, server capacity, and instructor recruiting requires accurate forecasts of course demand (registrations) and platform revenue. This paper applies machine learning to forecast demand and revenue, enabling executive decision-makers to optimize pricing, resource allocation, and curriculum planning.

---

## 2. Literature Review
Prior research in educational technology and revenue management highlights that student enrollments are driven by factors like course category popularity, pricing models (freemium vs. paid), course durations, and instructor reputation (ratings and experience). 
- **Freemium Models:** Platforms often use free courses to build a large user base (funnel), converting them to paid users via certificates or advanced tracks.
- **Machine Learning in EdTech:** Regression models (specifically linear models and tree ensembles) are widely used to model pricing elasticity and customer lifetime value. Tree-based regressors (e.g., Random Forests) often outperform linear models when features interact non-linearly (e.g., high experience compensating for low ratings).

---

## 3. Methodology
Our data pipeline consists of the following stages:
1. **Data Collection:** Importing four sheets from the raw Excel spreadsheet: Users, Courses, Teachers, and Transactions.
2. **Data Cleaning:** Deduplicating records, handling missing rating/duration metrics via median/mean imputation, and filtering out invalid prices (negative values) and ratings (outside [0, 5]).
3. **Merging:** Joining the Transaction-level table with course and teacher characteristics to form a complete merged dataset.
4. **Aggregation:** Grouping the transaction records by Course and Month to calculate monthly enrollments (transaction count) and monthly revenue (transaction sum).
5. **Feature Engineering:** Creating categories/bins (Price Bands, Experience Buckets, Duration Buckets, Rating Tiers), extracting date features (Month, Quarter, Weekday, Season), and calculating shifted historical variables (e.g., rolling averages).
6. **Model Training:** Splitting features (80% training, 20% testing) and training six regression estimators (Linear, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost).
7. **Evaluation:** Assessing models using MAE, RMSE, and R² scores.

---

## 4. Exploratory Data Analysis (EDA)
Key findings from our exploratory data analysis:
- **monetization:** Free courses represent 64.03% of all student registrations, demonstrating a highly active but unmonetized funnel. The entire platform revenue is carried by 22 paid courses.
- **Category Popularity:** Data Science courses generate the highest average registration volume (183.2 per course), followed by Finance (172.8).
- **Pricing Impact:** A scatter plot of course prices against total revenues indicates an almost perfect linear relationship. This suggests that price does not heavily suppress student registration volumes in the current price range ($0 to $490), meaning demand is highly inelastic.
- **Instructor Impact:** Instructors with 6-10 years of experience generate significantly higher average revenue than both junior teachers and highly veteran teachers (10+ years), indicating this is the optimal recruitment bracket.

---

## 5. Feature Engineering
Continuous features were mapped into discrete bands to provide robust non-linear features:
- **Price Bands:** Free ($0), Low ($0-$150), Medium ($150-$350), and High ($350+).
- **Duration Buckets:** Short (<15 hours), Medium (15-35 hours), and Long (>=35 hours).
- **Rating Tiers:** Excellent (4.5-5.0), Good (4.0-4.5), and Average (Below 4.0).
- **Experience Buckets:** 0-2 years, 3-5 years, 6-10 years, and 10+ years.

For Course-Month records, we computed cumulative aggregates shifted by 1 month to avoid data leakage:
- **Past Enrollment Count:** Cumulative registrations of the course.
- **Average Revenue:** Average monthly revenue.
- **Revenue per Enrollment:** Total past revenue divided by past enrollment.
- **Average Monthly Revenue/Enrollment:** 3-month rolling averages of previous performance.

---

## 6. Model Comparison and Evaluation
We evaluated models under two configurations: Static (using only course and teacher details, ideal for new course launches) and Dynamic (using historical and date features).

### Revenue Prediction Results (Monthly Level)
| Configuration | Model | MAE ($) | RMSE ($) | R² Score |
| :--- | :--- | :--- | :--- | :--- |
| **Static** | **Ridge Regressor (Selected)** | **389.93** | **796.23** | **0.8727** |
| | Lasso | 390.86 | 798.22 | 0.8720 |
| | Linear Regression | 391.38 | 798.49 | 0.8719 |
| | Gradient Boosting | 349.22 | 854.71 | 0.8533 |
| | Random Forest | 384.48 | 892.33 | 0.8401 |
| | XGBoost | 381.65 | 895.93 | 0.8388 |

### Enrollment Prediction Results (Monthly Level)
For monthly enrollments, the R² score for all models was near 0 or slightly negative (e.g. Ridge R² = -0.036).
*Analysis:* The average monthly enrollment per course is 13.88 with a tiny standard deviation of 3.64. The distribution is highly uniform across all courses and months, indicating random fluctuations. When a target variable exhibits low variance and high noise, a mean baseline predictor is mathematically optimal.

---

## 7. Business Recommendations

### 1. Pricing Strategy
- **Promote Medium Pricing Tier ($150 - $350):** These courses generate the highest average revenue with negligible demand contraction, making them the most profitable option.
- **Free Course Funnel Conversion:** Offer micro-credentials or paid completion certificates ($20 - $50) for the 38 Free courses to monetize the 64.03% of users who currently register for free content.

### 2. Category Launch Planning
- **Prioritize Data Science & AI:** These categories show the highest average student demand and revenue contribution.
- **Offer Digital Marketing Promos:** Digital Marketing courses show the lowest average enrollments. Leverage bundle pricing or free modules to boost acquisition in this category.

### 3. Instructor Recruitment
- **Target Mid-Career Instructors:** Recruit teachers with 6-10 years of experience, as they generate 42% higher average revenue than junior teachers.
- **Maintain Teacher Quality Standards:** Teacher ratings have a strong positive correlation (0.45) with course ratings and enrollments. Implement checks to support high ratings.

---

## 8. Conclusion
We successfully designed a machine learning pipeline for EduPro. While course demand is mostly uniform, course revenue can be predicted with high precision (R² = 0.87) using the Ridge Regressor. Combining these models with the Streamlit Simulator enables EduPro leaders to plan future course launches, set pricing strategies, and optimize resources with high confidence.
