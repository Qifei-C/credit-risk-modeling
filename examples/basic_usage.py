"""
Basic Usage Example for Credit Risk Modeling
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.credit_risk_model import CreditRiskModel
from src.data_preprocessing import DataPreprocessor

def main():
    """
    Demonstrate basic usage of the credit risk modeling system
    """
    print("=== Credit Risk Modeling - Basic Usage Example ===\n")
    
    # Create sample data (in real usage, load from file)
    print("1. Creating sample data...")
    sample_data = create_sample_data()
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Default rate: {sample_data['default'].mean():.2%}")
    print()
    
    # Initialize preprocessor
    print("2. Preprocessing data...")
    preprocessor = DataPreprocessor()
    
    # Prepare data
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        sample_data, 
        target_column='default',
        test_size=0.2,
        random_state=42
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of features: {len(X_train.columns)}")
    print()
    
    # Train model
    print("3. Training XGBoost model...")
    model = CreditRiskModel(model_type='xgboost')
    model.fit(X_train, y_train)
    print("Model training completed!")
    print()
    
    # Evaluate model
    print("4. Evaluating model performance...")
    metrics = model.evaluate(X_test, y_test)
    print("Performance Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()
    
    # Make predictions
    print("5. Making predictions...")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    print(f"Number of predicted defaults: {predictions.sum()}")
    print(f"Predicted default rate: {predictions.mean():.2%}")
    print(f"Actual default rate: {y_test.mean():.2%}")
    print()
    
    # Assess individual risk
    print("6. Assessing individual credit risk...")
    sample_applicant = X_test.iloc[0:1]  # Take first test sample
    risk_assessment = model.assess_credit_risk(sample_applicant)
    
    print("Sample Risk Assessment:")
    for key, value in risk_assessment.items():
        print(f"  {key}: {value}")
    print()
    
    # Feature importance
    print("7. Feature importance analysis...")
    feature_importance = model.get_feature_importance()
    print("Top 10 Most Important Features:")
    print(feature_importance.head(10))
    print()
    
    # Save model
    print("8. Saving model...")
    model.save_model('models/credit_risk_model.joblib')
    print("Model saved successfully!")
    print()
    
    # Load model
    print("9. Loading model...")
    loaded_model = CreditRiskModel.load_model('models/credit_risk_model.joblib')
    print("Model loaded successfully!")
    
    # Test loaded model
    test_prediction = loaded_model.predict(sample_applicant)
    print(f"Test prediction from loaded model: {test_prediction[0]}")
    print()
    
    print("=== Example completed successfully! ===")

def create_sample_data(n_samples=1000):
    """
    Create sample credit data for demonstration
    """
    np.random.seed(42)
    
    # Generate synthetic credit data
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'gender': np.random.choice([1, 2], n_samples),
        'education': np.random.choice([1, 2, 3, 4], n_samples),
        'marriage': np.random.choice([1, 2, 3], n_samples),
        'limit_balance': np.random.randint(10000, 500000, n_samples),
    }
    
    # Add payment history features
    for i in range(1, 7):
        data[f'pay_{i}'] = np.random.choice([-1, 1, 2, 3, 4, 5, 6, 7, 8, 9], n_samples)
    
    # Add bill amount features
    for i in range(1, 7):
        data[f'bill_amt_{i}'] = np.random.randint(0, 100000, n_samples)
    
    # Add payment amount features
    for i in range(1, 7):
        data[f'pay_amt_{i}'] = np.random.randint(0, 50000, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate target variable with some logic
    # Higher risk for: older age, higher limit balance, poor payment history
    risk_score = (
        (df['age'] > 40) * 0.1 +
        (df['limit_balance'] > 200000) * 0.2 +
        (df['pay_1'] > 2) * 0.3 +
        (df['pay_2'] > 2) * 0.2 +
        np.random.random(n_samples) * 0.3
    )
    
    df['default'] = (risk_score > 0.5).astype(int)
    
    return df

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Run main example
    main()