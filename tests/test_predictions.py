import pytest
import pickle
import os
import pandas as pd
import numpy as np

def test_models_exist():
    # Verify model pickles are present
    models_dir = r"c:\Users\hp\Desktop\revenue\models"
    assert os.path.exists(os.path.join(models_dir, "static_preprocessor.pkl"))
    assert os.path.exists(os.path.join(models_dir, "static_enrollment_model.pkl"))
    assert os.path.exists(os.path.join(models_dir, "static_revenue_model.pkl"))

def test_predictions():
    models_dir = r"c:\Users\hp\Desktop\revenue\models"
    
    with open(os.path.join(models_dir, "static_preprocessor.pkl"), 'rb') as f:
        preprocessor = pickle.load(f)
    with open(os.path.join(models_dir, "static_enrollment_model.pkl"), 'rb') as f:
        enroll_model = pickle.load(f)
    with open(os.path.join(models_dir, "static_revenue_model.pkl"), 'rb') as f:
        rev_model = pickle.load(f)
        
    # Mock course input
    mock_input = pd.DataFrame([{
        'CoursePrice': 99.0,
        'CourseDuration': 24.0,
        'CourseRating': 4.5,
        'YearsOfExperience': 5,
        'TeacherRating': 4.2,
        'CourseCategory': 'Data Science',
        'CourseType': 'Paid',
        'CourseLevel': 'Beginner'
    }])
    
    # Preprocess
    proc_features = preprocessor.transform(mock_input)
    assert proc_features.shape[0] == 1
    
    # Predict
    pred_enroll = enroll_model.predict(proc_features)
    pred_rev = rev_model.predict(proc_features)
    
    # Assert output shapes
    assert len(pred_enroll) == 1
    assert len(pred_rev) == 1
    
    # Assert sanity bounds (predictions are numbers)
    assert isinstance(pred_enroll[0], (int, float, np.floating, np.integer))
    assert isinstance(pred_rev[0], (int, float, np.floating, np.integer))
