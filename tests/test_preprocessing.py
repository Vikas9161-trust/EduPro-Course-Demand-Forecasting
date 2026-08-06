import pytest
import pandas as pd
import numpy as np
from src.data.cleaning import clean_courses, clean_teachers, clean_transactions
from src.data.feature_engineering import create_bins, extract_date_features

def test_clean_courses():
    # Mock course data with duplicates, missing values, and invalid rows
    data = {
        'CourseID': ['CR01 ', 'CR01 ', 'CR02', 'CR03'],
        'CourseName': ['Basics', 'Basics', 'Advanced', 'Invalid Course'],
        'CourseCategory': ['Tech', 'Tech', 'Business', 'Tech'],
        'CourseType': ['Free', 'Free', 'Paid', 'Paid'],
        'CourseLevel': ['Beginner', 'Beginner', 'Advanced', 'Intermediate'],
        'CoursePrice': [0.0, 0.0, 50.0, -10.0],  # CR03 has invalid price
        'CourseDuration': [10.0, 10.0, np.nan, 15.0],  # CR02 has missing duration (will be filled)
        'CourseRating': [4.5, 4.5, 4.2, 6.0]  # CR03 has invalid rating
    }
    df = pd.DataFrame(data)
    cleaned = clean_courses(df)
    
    # Check duplicates removed and invalid rows dropped
    assert len(cleaned) == 2
    # Check CourseID stripped
    assert 'CR01' in cleaned['CourseID'].values
    # Check invalid price fixed
    assert (cleaned['CoursePrice'] >= 0).all()
    # Check missing duration handled (median filled)
    assert not cleaned['CourseDuration'].isna().any()
    # Check invalid rating handled (filtered out)
    assert (cleaned['CourseRating'] <= 5.0).all()

def test_clean_teachers():
    # Mock teacher data with missing/invalid values
    data = {
        'TeacherID': ['TC01', 'TC02'],
        'Expertise': ['AI', 'Math'],
        'YearsOfExperience': [-1, 5],  # Invalid exp
        'TeacherRating': [4.5, np.nan]  # Missing rating
    }
    df = pd.DataFrame(data)
    cleaned = clean_teachers(df)
    
    # Check invalid experience filtered/filled
    assert (cleaned['YearsOfExperience'] >= 0).all()
    # Check missing rating filled
    assert not cleaned['TeacherRating'].isna().any()

def test_create_bins():
    # Test binning thresholds
    data = {
        'CoursePrice': [0.0, 50.0, 200.0, 400.0],
        'CourseDuration': [10.0, 25.0, 45.0],
        'CourseRating': [3.5, 4.2, 4.8],
        'YearsOfExperience': [1, 4, 8, 12]
    }
    # To test cut, we need matching lengths, so we'll construct structured rows
    df = pd.DataFrame({
        'CoursePrice': [0.0, 100.0, 250.0, 450.0],
        'CourseDuration': [5.0, 20.0, 40.0, 15.0],
        'CourseRating': [3.0, 4.1, 4.7, 4.4],
        'YearsOfExperience': [1, 4, 8, 12]
    })
    
    binned = create_bins(df)
    
    # Verify columns exist
    assert 'PriceBand' in binned.columns
    assert 'DurationBucket' in binned.columns
    assert 'RatingTier' in binned.columns
    assert 'ExperienceBucket' in binned.columns
    
    # Verify correct mapping
    # Price bands: Free (0), Low (0-150), Medium (150-350), High (350+)
    assert binned.loc[0, 'PriceBand'] == 'Free'
    assert binned.loc[1, 'PriceBand'] == 'Low'
    assert binned.loc[2, 'PriceBand'] == 'Medium'
    assert binned.loc[3, 'PriceBand'] == 'High'
    
    # Experience buckets: 0-2 years, 3-5 years, 6-10 years, 10+
    assert binned.loc[0, 'ExperienceBucket'] == '0-2 years'
    assert binned.loc[1, 'ExperienceBucket'] == '3-5 years'
    assert binned.loc[2, 'ExperienceBucket'] == '6-10 years'
    assert binned.loc[3, 'ExperienceBucket'] == '10+'
