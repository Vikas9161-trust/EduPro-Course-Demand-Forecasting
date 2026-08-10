import pandas as pd
import numpy as np
import os

def create_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates bins for Price, Duration, Rating, and Experience.
    """
    df = df.copy()
    
    # 1. Price Bands: Free (0), Low (0-150), Medium (150-350), High (350+)
    bins_price = [-1, 0.01, 150, 350, np.inf]
    labels_price = ['Free', 'Low', 'Medium', 'High']
    df['PriceBand'] = pd.cut(df['CoursePrice'], bins=bins_price, labels=labels_price)
    
    # 2. Duration Buckets: Short (<15), Medium (15-35), Long (>=35)
    bins_duration = [-1, 15, 35, np.inf]
    labels_duration = ['Short', 'Medium', 'Long']
    df['DurationBucket'] = pd.cut(df['CourseDuration'], bins=bins_duration, labels=labels_duration)
    
    # 3. Rating Tiers: Excellent (4.5-5.0), Good (4.0-4.5), Average (Below 4)
    bins_rating = [-1, 4.0, 4.5, 5.05]
    labels_rating = ['Average', 'Good', 'Excellent']
    df['RatingTier'] = pd.cut(df['CourseRating'], bins=bins_rating, labels=labels_rating, right=False)
    
    # 4. Experience Buckets: 0-2 years, 3-5 years, 6-10 years, 10+
    bins_exp = [-1, 2, 5, 10, np.inf]
    labels_exp = ['0-2 years', '3-5 years', '6-10 years', '10+']
    df['ExperienceBucket'] = pd.cut(df['YearsOfExperience'], bins=bins_exp, labels=labels_exp)
    
    return df

def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts date features from TransactionDate.
    """
    df = df.copy()
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    
    df['Month'] = df['TransactionDate'].dt.month
    df['Year'] = df['TransactionDate'].dt.year
    df['Quarter'] = df['TransactionDate'].dt.quarter
    df['Weekday'] = df['TransactionDate'].dt.weekday
    
    # Season: Winter (12, 1, 2), Spring (3, 4, 5), Summer (6, 7, 8), Autumn (9, 10, 11)
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'
            
    df['Season'] = df['Month'].apply(get_season)
    return df

def generate_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates historical features for each Course-Month.
    df must be sorted by CourseID, Year, and Month.
    """
    df = df.copy()
    
    # We will compute cumulative features shift by 1 month to prevent data leakage.
    # Group by CourseID and compute rolling/cumulative features.
    
    # Past Enrollment Count (cumulative enrollments up to previous month)
    df['PastEnrollmentCount'] = df.groupby('CourseID')['MonthlyEnrollments'].cumsum() - df['MonthlyEnrollments']
    
    # Cumulative Revenue up to previous month
    df['PastRevenue'] = df.groupby('CourseID')['MonthlyRevenue'].cumsum() - df['MonthlyRevenue']
    
    # Count of previous months
    df['PrevMonthsCount'] = df.groupby('CourseID').cumcount()
    
    # Average Revenue (average monthly revenue up to previous month)
    df['AverageRevenue'] = np.where(
        df['PrevMonthsCount'] > 0,
        df['PastRevenue'] / df['PrevMonthsCount'],
        0.0
    )
    
    # Revenue per Enrollment (cumulative revenue / cumulative enrollment up to previous month)
    df['RevenuePerEnrollment'] = np.where(
        df['PastEnrollmentCount'] > 0,
        df['PastRevenue'] / df['PastEnrollmentCount'],
        0.0
    )
    
    # Average Monthly Revenue (rolling average of past 3 months, shifted by 1)
    df['AverageMonthlyRevenue'] = df.groupby('CourseID')['MonthlyRevenue'].shift(1).rolling(window=3, min_periods=1).mean().fillna(0.0)
    
    # Average Monthly Enrollment (rolling average of past 3 months, shifted by 1)
    df['AverageMonthlyEnrollment'] = df.groupby('CourseID')['MonthlyEnrollments'].shift(1).rolling(window=3, min_periods=1).mean().fillna(0.0)
    
    # Drop temporary columns
    df = df.drop(columns=['PastRevenue', 'PrevMonthsCount'])
    
    return df

def process_features(processed_dir: str):
    """
    Loads cleaned CSVs, merges them, engineers features, and saves:
    - merged_dataset.csv (transaction level)
    - feature_dataset.csv (course-month level)
    - course_level_dataset.csv (course level summary)
    """
    courses_path = os.path.join(processed_dir, "cleaned_courses.csv")
    teachers_path = os.path.join(processed_dir, "cleaned_teachers.csv")
    transactions_path = os.path.join(processed_dir, "cleaned_transactions.csv")
    users_path = os.path.join(processed_dir, "cleaned_users.csv")
    
    courses = pd.read_csv(courses_path)
    teachers = pd.read_csv(teachers_path)
    transactions = pd.read_csv(transactions_path)
    users = pd.read_csv(users_path)
    
    # Step 3: Merge Dataset (Transaction-level)
    print("Merging datasets...")
    # Transactions has CourseID, TeacherID, and UserID. Course does not have TeacherID.
    merged = transactions.merge(courses, on="CourseID", how="left")
    merged = merged.merge(teachers, on="TeacherID", how="left")
    merged = merged.merge(users, on="UserID", how="left", suffixes=('_teacher', '_user'))
    
    # Save transaction-level merged dataset
    merged.to_csv(os.path.join(processed_dir, "merged_dataset.csv"), index=False)
    print(f"Transaction-level merged dataset saved. Shape: {merged.shape}")
    
    # Extract date features on transaction level
    merged = extract_date_features(merged)
    
    # Create bins on transaction level
    merged = create_bins(merged)
    
    # Step 4: Aggregate to Course-Month level for time-based forecasting
    print("Aggregating to Course-Month level...")
    # Group by CourseID, Year, Month and aggregate
    # For course and teacher attributes, we take the first value (since they are constant per Course/Teacher)
    agg_dict = {
        'TransactionID': 'count', # Monthly enrollments
        'Amount': 'sum',          # Monthly revenue
        'CourseName': 'first',
        'CourseCategory': 'first',
        'CourseType': 'first',
        'CourseLevel': 'first',
        'CoursePrice': 'first',
        'CourseDuration': 'first',
        'CourseRating': 'first',
        'TeacherID': 'first',
        'TeacherName': 'first',
        'Expertise': 'first',
        'YearsOfExperience': 'first',
        'TeacherRating': 'first',
        'PriceBand': 'first',
        'DurationBucket': 'first',
        'RatingTier': 'first',
        'ExperienceBucket': 'first',
        'Quarter': 'first',
        'Season': 'first'
    }
    
    course_month = merged.groupby(['CourseID', 'Year', 'Month']).agg(agg_dict).reset_index()
    course_month = course_month.rename(columns={
        'TransactionID': 'MonthlyEnrollments',
        'Amount': 'MonthlyRevenue'
    })
    
    # Sort for chronological historical calculation
    course_month = course_month.sort_values(by=['CourseID', 'Year', 'Month']).reset_index(drop=True)
    
    # Calculate historical features
    print("Calculating historical features...")
    course_month = generate_historical_features(course_month)
    
    # Save Course-Month level feature dataset
    course_month.to_csv(os.path.join(processed_dir, "feature_dataset.csv"), index=False)
    print(f"Course-Month level feature dataset saved. Shape: {course_month.shape}")
    
    # Aggregate to Course level (static course summary for static demand model)
    print("Aggregating to Course level...")
    course_agg_dict = {
        'MonthlyEnrollments': 'sum',
        'MonthlyRevenue': 'sum',
        'CourseName': 'first',
        'CourseCategory': 'first',
        'CourseType': 'first',
        'CourseLevel': 'first',
        'CoursePrice': 'first',
        'CourseDuration': 'first',
        'CourseRating': 'first',
        'TeacherID': 'first',
        'TeacherName': 'first',
        'Expertise': 'first',
        'YearsOfExperience': 'first',
        'TeacherRating': 'first',
        'PriceBand': 'first',
        'DurationBucket': 'first',
        'RatingTier': 'first',
        'ExperienceBucket': 'first'
    }
    course_level = course_month.groupby('CourseID').agg(course_agg_dict).reset_index()
    course_level = course_level.rename(columns={
        'MonthlyEnrollments': 'TotalEnrollments',
        'MonthlyRevenue': 'TotalRevenue'
    })
    # Add Average Monthly Enrollments and Revenue
    # Since we have 12 months in 2025
    course_level['AverageMonthlyEnrollments'] = course_level['TotalEnrollments'] / 12.0
    course_level['AverageMonthlyRevenue'] = course_level['TotalRevenue'] / 12.0
    
    course_level.to_csv(os.path.join(processed_dir, "course_level_dataset.csv"), index=False)
    print(f"Course-level dataset saved. Shape: {course_level.shape}")
    
    return merged, course_month, course_level

if __name__ == "__main__":
    processed_dir = r"c:\Users\hp\Desktop\revenue\data\processed"
    process_features(processed_dir)
