import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_models(X_train, X_test, y_train, y_test, target_name):
    """
    Trains and evaluates multiple models, returning a summary dataframe and the best model.
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, learning_rate=0.08)
    }
    
    results = []
    best_model_name = None
    best_r2 = -np.inf
    best_model = None
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        # Predict
        preds = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        results.append({
            'Model': name,
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'R2': round(r2, 4)
        })
        
        # We prefer models with higher R2 score. If R2 is close, Random Forest or Gradient Boosting is preferred.
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model = model
            
    results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)
    print(f"\n--- Model Evaluation Results for: {target_name} ---")
    print(results_df.to_string(index=False))
    print(f"Best Model Selected: {best_model_name} with R2 of {best_r2:.4f}")
    
    return results_df, best_model, best_model_name

def train_pipeline(data_path: str, models_dir: str):
    """
    Main training pipeline:
    1. Loads dataset
    2. Splits into Static features and Dynamic features
    3. Preprocesses features using Pipeline & ColumnTransformer
    4. Trains and selects the best model for Enrollment and Revenue
    5. Serializes models, encoders, and scalers
    """
    df = pd.read_csv(data_path)
    
    # ------------------
    # STATIC MODELS SETUP (No historical features)
    # Useful for new course demand prediction
    # ------------------
    print("\n--- Training Static Models (For new course demand prediction) ---")
    static_num_cols = ['CoursePrice', 'CourseDuration', 'CourseRating', 'YearsOfExperience', 'TeacherRating']
    static_cat_cols = ['CourseCategory', 'CourseType', 'CourseLevel']
    
    # Preprocessor for Static features
    static_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), static_num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), static_cat_cols)
        ]
    )
    
    X_static = df[static_num_cols + static_cat_cols]
    y_enrollment = df['MonthlyEnrollments']
    y_revenue = df['MonthlyRevenue']
    
    # Split
    X_train_se, X_test_se, y_train_se, y_test_se = train_test_split(X_static, y_enrollment, test_size=0.2, random_state=42)
    X_train_sr, X_test_sr, y_train_sr, y_test_sr = train_test_split(X_static, y_revenue, test_size=0.2, random_state=42)
    
    # Preprocess splits
    X_train_se_proc = static_preprocessor.fit_transform(X_train_se)
    X_test_se_proc = static_preprocessor.transform(X_test_se)
    
    X_train_sr_proc = static_preprocessor.fit_transform(X_train_sr)
    X_test_sr_proc = static_preprocessor.transform(X_test_sr)
    
    # Get feature names after encoding for importance charting
    # Fitted OneHotEncoder is the second transformer in static_preprocessor
    ohe = static_preprocessor.named_transformers_['cat']
    encoded_cat_names = list(ohe.get_feature_names_out(static_cat_cols))
    all_static_features = static_num_cols + encoded_cat_names
    
    # Evaluate
    results_se, best_se_model, name_se = evaluate_models(X_train_se_proc, X_test_se_proc, y_train_se, y_test_se, "Static Enrollments")
    results_sr, best_sr_model, name_sr = evaluate_models(X_train_sr_proc, X_test_sr_proc, y_train_sr, y_test_sr, "Static Revenue")
    
    # Save static preprocessing pipeline and models
    os.makedirs(models_dir, exist_ok=True)
    
    with open(os.path.join(models_dir, "static_preprocessor.pkl"), 'wb') as f:
        pickle.dump(static_preprocessor, f)
        
    with open(os.path.join(models_dir, "static_enrollment_model.pkl"), 'wb') as f:
        pickle.dump(best_se_model, f)
        
    with open(os.path.join(models_dir, "static_revenue_model.pkl"), 'wb') as f:
        pickle.dump(best_sr_model, f)
        
    with open(os.path.join(models_dir, "static_features_list.pkl"), 'wb') as f:
        pickle.dump(all_static_features, f)
        
    print("\nStatic Models successfully saved to:", models_dir)
    
    # ------------------
    # DYNAMIC MODELS SETUP (With date & historical features)
    # Useful for time-series forecasting of existing courses
    # ------------------
    print("\n--- Training Dynamic Models (For forecasting and trends) ---")
    dynamic_num_cols = [
        'CoursePrice', 'CourseDuration', 'CourseRating', 'YearsOfExperience', 'TeacherRating',
        'Month', 'Quarter', 'PastEnrollmentCount', 'AverageRevenue', 'RevenuePerEnrollment',
        'AverageMonthlyRevenue', 'AverageMonthlyEnrollment'
    ]
    dynamic_cat_cols = ['CourseCategory', 'CourseType', 'CourseLevel', 'Season']
    
    dynamic_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), dynamic_num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), dynamic_cat_cols)
        ]
    )
    
    X_dynamic = df[dynamic_num_cols + dynamic_cat_cols]
    
    # Split
    X_train_de, X_test_de, y_train_de, y_test_de = train_test_split(X_dynamic, y_enrollment, test_size=0.2, random_state=42)
    X_train_dr, X_test_dr, y_train_dr, y_test_dr = train_test_split(X_dynamic, y_revenue, test_size=0.2, random_state=42)
    
    # Preprocess
    X_train_de_proc = dynamic_preprocessor.fit_transform(X_train_de)
    X_test_de_proc = dynamic_preprocessor.transform(X_test_de)
    
    X_train_dr_proc = dynamic_preprocessor.fit_transform(X_train_dr)
    X_test_dr_proc = dynamic_preprocessor.transform(X_test_dr)
    
    # Get feature names
    ohe_dyn = dynamic_preprocessor.named_transformers_['cat']
    encoded_cat_names_dyn = list(ohe_dyn.get_feature_names_out(dynamic_cat_cols))
    all_dynamic_features = dynamic_num_cols + encoded_cat_names_dyn
    
    # Evaluate
    results_de, best_de_model, name_de = evaluate_models(X_train_de_proc, X_test_de_proc, y_train_de, y_test_de, "Dynamic Enrollments")
    results_dr, best_dr_model, name_dr = evaluate_models(X_train_dr_proc, X_test_dr_proc, y_train_dr, y_test_dr, "Dynamic Revenue")
    
    # Save dynamic models
    with open(os.path.join(models_dir, "dynamic_preprocessor.pkl"), 'wb') as f:
        pickle.dump(dynamic_preprocessor, f)
        
    with open(os.path.join(models_dir, "dynamic_enrollment_model.pkl"), 'wb') as f:
        pickle.dump(best_de_model, f)
        
    with open(os.path.join(models_dir, "dynamic_revenue_model.pkl"), 'wb') as f:
        pickle.dump(best_dr_model, f)
        
    with open(os.path.join(models_dir, "dynamic_features_list.pkl"), 'wb') as f:
        pickle.dump(all_dynamic_features, f)
        
    print("\nDynamic Models successfully saved to:", models_dir)
    
    # Save a CSV model report to reports folder
    os.makedirs(r"c:\Users\hp\Desktop\revenue\reports", exist_ok=True)
    report_df = pd.concat([
        results_se.assign(Target="Static Enrollments"),
        results_sr.assign(Target="Static Revenue"),
        results_de.assign(Target="Dynamic Enrollments"),
        results_dr.assign(Target="Dynamic Revenue")
    ], ignore_index=True)
    
    report_df.to_csv(r"c:\Users\hp\Desktop\revenue\reports\model_evaluation_metrics.csv", index=False)
    print("Model evaluation report saved to reports/model_evaluation_metrics.csv")

if __name__ == "__main__":
    data_path = r"c:\Users\hp\Desktop\revenue\data\processed\feature_dataset.csv"
    models_dir = r"c:\Users\hp\Desktop\revenue\models"
    train_pipeline(data_path, models_dir)
