"""
Credit Risk Model Implementation
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
import pickle
import joblib
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class CreditRiskModel:
    """
    A comprehensive credit risk modeling class that supports multiple algorithms
    and provides unified interface for training, prediction, and evaluation.
    """
    
    def __init__(self, model_type: str = 'xgboost', random_state: int = 42):
        """
        Initialize the Credit Risk Model
        
        Args:
            model_type: Type of model to use ('xgboost', 'lightgbm', 'random_forest', 'logistic', 'gradient_boosting')
            random_state: Random state for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.is_fitted = False
        
        # Initialize model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on the specified type"""
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='logloss'
            )
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                verbose=-1
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state
            )
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> 'CreditRiskModel':
        """
        Train the credit risk model
        
        Args:
            X: Training features
            y: Training target
            validation_data: Optional validation data for early stopping
            
        Returns:
            self: Fitted model
        """
        self.feature_names = list(X.columns)
        
        # Fit the model
        if validation_data and self.model_type in ['xgboost', 'lightgbm']:
            X_val, y_val = validation_data
            
            if self.model_type == 'xgboost':
                self.model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=10,
                    verbose=False
                )
            elif self.model_type == 'lightgbm':
                self.model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=10,
                    verbose=False
                )
        else:
            self.model.fit(X, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X: Features for prediction
            
        Returns:
            Binary predictions (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Features for prediction
            
        Returns:
            Probability estimates
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict_proba(X)
    
    def assess_credit_risk(self, applicant_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess credit risk for an applicant
        
        Args:
            applicant_data: Single applicant's data
            
        Returns:
            Risk assessment dictionary
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before assessing risk")
        
        # Get prediction and probability
        prediction = self.predict(applicant_data)[0]
        proba = self.predict_proba(applicant_data)[0]
        
        # Calculate risk score (probability of default)
        risk_score = proba[1]
        
        # Determine risk level
        if risk_score <= 0.3:
            risk_level = "Low"
        elif risk_score <= 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            'prediction': int(prediction),
            'risk_score': float(risk_score),
            'risk_level': risk_level,
            'approval_recommendation': 'Approve' if prediction == 0 else 'Decline'
        }
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Test features
            y: Test target
            
        Returns:
            Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Make predictions
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y, y_proba)
        
        # Get classification report
        report = classification_report(y, y_pred, output_dict=True)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='roc_auc')
        
        return {
            'auc_score': auc_score,
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std()
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            raise ValueError("Model does not support feature importance")
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def save_model(self, filepath: str):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'CreditRiskModel':
        """
        Load a trained model
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded CreditRiskModel instance
        """
        model_data = joblib.load(filepath)
        
        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.is_fitted = model_data['is_fitted']
        
        return instance


class EnsembleCreditRiskModel:
    """
    Ensemble model combining multiple credit risk models
    """
    
    def __init__(self, models: List[str] = None, random_state: int = 42):
        """
        Initialize ensemble model
        
        Args:
            models: List of model types to ensemble
            random_state: Random state for reproducibility
        """
        if models is None:
            models = ['xgboost', 'lightgbm', 'random_forest']
        
        self.models = []
        self.random_state = random_state
        
        for model_type in models:
            self.models.append(CreditRiskModel(model_type=model_type, random_state=random_state))
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None):
        """
        Train all models in the ensemble
        """
        for model in self.models:
            model.fit(X, y, validation_data)
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities using ensemble average
        """
        probas = []
        for model in self.models:
            probas.append(model.predict_proba(X))
        
        # Average probabilities
        ensemble_proba = np.mean(probas, axis=0)
        return ensemble_proba
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using ensemble
        """
        probas = self.predict_proba(X)
        return (probas[:, 1] > 0.5).astype(int)
    
    def assess_credit_risk(self, applicant_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess credit risk using ensemble
        """
        prediction = self.predict(applicant_data)[0]
        proba = self.predict_proba(applicant_data)[0]
        
        risk_score = proba[1]
        
        if risk_score <= 0.3:
            risk_level = "Low"
        elif risk_score <= 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            'prediction': int(prediction),
            'risk_score': float(risk_score),
            'risk_level': risk_level,
            'approval_recommendation': 'Approve' if prediction == 0 else 'Decline'
        }