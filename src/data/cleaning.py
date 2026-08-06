import pandas as pd
import numpy as np
import os

def clean_courses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the Courses dataframe.
    - Removes duplicates.
    - Corrects data types.
    - Handles missing values.
    - Filters invalid prices and ratings.
    """
    df = df.copy()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Correct data types
    df['CourseID'] = df['CourseID'].astype(str).str.strip()
    df['CourseCategory'] = df['CourseCategory'].astype(str).str.strip()
    df['CourseType'] = df['CourseType'].astype(str).str.strip()
    df['CourseLevel'] = df['CourseLevel'].astype(str).str.strip()
    
    df['CoursePrice'] = pd.to_numeric(df['CoursePrice'], errors='coerce')
    df['CourseDuration'] = pd.to_numeric(df['CourseDuration'], errors='coerce')
    df['CourseRating'] = pd.to_numeric(df['CourseRating'], errors='coerce')
    
    # Handle missing values
    # Prices: fill with median
    median_price = df['CoursePrice'].median()
    df['CoursePrice'] = df['CoursePrice'].fillna(median_price if not pd.isna(median_price) else 0.0)
    
    # Duration: fill with median
    median_duration = df['CourseDuration'].median()
    df['CourseDuration'] = df['CourseDuration'].fillna(median_duration if not pd.isna(median_duration) else 0.0)
    
    # Rating: fill with median or mean
    mean_rating = df['CourseRating'].mean()
    df['CourseRating'] = df['CourseRating'].fillna(mean_rating if not pd.isna(mean_rating) else 4.0)
    
    # Remove invalid prices (e.g. <= 0 or extremely high anomalies if any)
    # Price must be >= 0
    df = df[df['CoursePrice'] >= 0]
    
    # Remove invalid ratings (not in [1, 5] or [0, 5])
    df = df[(df['CourseRating'] >= 0) & (df['CourseRating'] <= 5.0)]
    
    return df

def clean_teachers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the Teachers dataframe.
    - Removes duplicates.
    - Corrects data types.
    - Handles missing values.
    - Filters invalid ratings and years of experience.
    """
    df = df.copy()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Correct data types
    df['TeacherID'] = df['TeacherID'].astype(str).str.strip()
    df['Expertise'] = df['Expertise'].astype(str).str.strip()
    
    df['YearsOfExperience'] = pd.to_numeric(df['YearsOfExperience'], errors='coerce')
    df['TeacherRating'] = pd.to_numeric(df['TeacherRating'], errors='coerce')
    
    # Handle missing values
    median_exp = df['YearsOfExperience'].median()
    df['YearsOfExperience'] = df['YearsOfExperience'].fillna(median_exp if not pd.isna(median_exp) else 0.0)
    
    mean_rating = df['TeacherRating'].mean()
    df['TeacherRating'] = df['TeacherRating'].fillna(mean_rating if not pd.isna(mean_rating) else 4.0)
    
    # Remove invalid ratings
    df = df[(df['TeacherRating'] >= 0) & (df['TeacherRating'] <= 5.0)]
    
    # Years of experience must be >= 0
    df = df[df['YearsOfExperience'] >= 0]
    
    return df

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the Transactions dataframe.
    - Removes duplicates.
    - Converts TransactionDate to datetime.
    - Handles missing values.
    - Filters invalid amounts.
    """
    df = df.copy()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Correct data types
    df['TransactionID'] = df['TransactionID'].astype(str).str.strip()
    df['CourseID'] = df['CourseID'].astype(str).str.strip()
    
    # Convert TransactionDate to datetime
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce')
    
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    
    # Drop rows with invalid transaction dates
    df = df.dropna(subset=['TransactionDate'])
    
    # Handle missing amounts (fill with median or drop)
    median_amount = df['Amount'].median()
    df['Amount'] = df['Amount'].fillna(median_amount if not pd.isna(median_amount) else 0.0)
    
    # Remove invalid amounts (must be >= 0)
    df = df[df['Amount'] >= 0]
    
    return df

def clean_and_save_data(excel_path: str, output_dir: str):
    """
    Loads raw Excel sheets, cleans them individually, and saves the cleaned
    data as separate CSVs in output_dir.
    """
    print(f"Loading data from {excel_path}...")
    xl = pd.ExcelFile(excel_path)
    
    # Check sheet names
    sheet_names = xl.sheet_names
    print("Sheets found:", sheet_names)
    
    # Initialize dataframes
    courses_df = xl.parse("Courses") if "Courses" in sheet_names else None
    teachers_df = xl.parse("Teachers") if "Teachers" in sheet_names else None
    transactions_df = xl.parse("Transactions") if "Transactions" in sheet_names else None
    
    if courses_df is None or teachers_df is None or transactions_df is None:
        raise ValueError("Excel file must contain Courses, Teachers, and Transactions sheets.")
        
    print("Cleaning Courses...")
    courses_clean = clean_courses(courses_df)
    
    print("Cleaning Teachers...")
    teachers_clean = clean_teachers(teachers_df)
    
    print("Cleaning Transactions...")
    transactions_clean = clean_transactions(transactions_df)
    
    # Save processed CSVs
    os.makedirs(output_dir, exist_ok=True)
    
    courses_clean.to_csv(os.path.join(output_dir, "cleaned_courses.csv"), index=False)
    teachers_clean.to_csv(os.path.join(output_dir, "cleaned_teachers.csv"), index=False)
    transactions_clean.to_csv(os.path.join(output_dir, "cleaned_transactions.csv"), index=False)
    
    print(f"Cleaned CSVs saved to {output_dir}")
    print(f"Courses cleaned shape: {courses_clean.shape}")
    print(f"Teachers cleaned shape: {teachers_clean.shape}")
    print(f"Transactions cleaned shape: {transactions_clean.shape}")
    
    return courses_clean, teachers_clean, transactions_clean

if __name__ == "__main__":
    raw_excel = r"c:\Users\hp\Desktop\revenue\data\raw\EduPro_Dataset.xlsx"
    processed_dir = r"c:\Users\hp\Desktop\revenue\data\processed"
    clean_and_save_data(raw_excel, processed_dir)
