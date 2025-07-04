"""
Data Preprocessing Module for Credit Risk Modeling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Comprehensive data preprocessing for credit risk modeling
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_names = None
        self.target_name = None
        
    def load_data(self, filepath: str, target_column: str = 'default') -> pd.DataFrame:
        """
        Load data from various file formats
        
        Args:
            filepath: Path to the data file
            target_column: Name of the target column
            
        Returns:
            Loaded DataFrame
        """
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
                df = pd.read_excel(filepath)
            elif filepath.endswith('.json'):
                df = pd.read_json(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
            
            self.target_name = target_column
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")
    
    def basic_data_info(self, df: pd.DataFrame) -> Dict:
        """
        Get basic information about the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with data information
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        # Add target distribution if target column exists
        if self.target_name and self.target_name in df.columns:
            info['target_distribution'] = df[self.target_name].value_counts().to_dict()
            info['target_balance'] = df[self.target_name].value_counts(normalize=True).to_dict()
        
        return info
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            strategy: Imputation strategy ('mean', 'median', 'most_frequent')
            
        Returns:
            DataFrame with missing values handled
        """
        df_processed = df.copy()
        
        # Separate numeric and categorical columns
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_processed.select_dtypes(exclude=[np.number]).columns
        
        # Handle numeric columns
        if len(numeric_cols) > 0:
            if 'numeric_imputer' not in self.imputers:
                self.imputers['numeric_imputer'] = SimpleImputer(strategy=strategy)
                df_processed[numeric_cols] = self.imputers['numeric_imputer'].fit_transform(df_processed[numeric_cols])
            else:
                df_processed[numeric_cols] = self.imputers['numeric_imputer'].transform(df_processed[numeric_cols])
        
        # Handle categorical columns
        if len(categorical_cols) > 0:
            if 'categorical_imputer' not in self.imputers:
                self.imputers['categorical_imputer'] = SimpleImputer(strategy='most_frequent')
                df_processed[categorical_cols] = self.imputers['categorical_imputer'].fit_transform(df_processed[categorical_cols])
            else:
                df_processed[categorical_cols] = self.imputers['categorical_imputer'].transform(df_processed[categorical_cols])
        
        return df_processed
    
    def encode_categorical_variables(self, df: pd.DataFrame, encoding_method: str = 'onehot') -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df: Input DataFrame
            encoding_method: Encoding method ('onehot', 'label')
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df_processed = df.copy()
        categorical_cols = df_processed.select_dtypes(exclude=[np.number]).columns
        
        # Exclude target column if it exists
        if self.target_name and self.target_name in categorical_cols:
            categorical_cols = categorical_cols.drop(self.target_name)
        
        if len(categorical_cols) > 0:
            if encoding_method == 'onehot':
                # One-hot encoding
                for col in categorical_cols:
                    if col not in self.encoders:
                        self.encoders[col] = OneHotEncoder(sparse=False, handle_unknown='ignore')
                        encoded = self.encoders[col].fit_transform(df_processed[[col]])
                        feature_names = [f"{col}_{category}" for category in self.encoders[col].categories_[0]]
                    else:
                        encoded = self.encoders[col].transform(df_processed[[col]])
                        feature_names = [f"{col}_{category}" for category in self.encoders[col].categories_[0]]
                    
                    # Create DataFrame with encoded features
                    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df_processed.index)
                    
                    # Drop original column and add encoded columns
                    df_processed = df_processed.drop(columns=[col])
                    df_processed = pd.concat([df_processed, encoded_df], axis=1)
            
            elif encoding_method == 'label':
                # Label encoding
                for col in categorical_cols:
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                        df_processed[col] = self.encoders[col].fit_transform(df_processed[col])
                    else:
                        df_processed[col] = self.encoders[col].transform(df_processed[col])
        
        return df_processed
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            df: Input DataFrame
            method: Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
            DataFrame with scaled features
        """
        df_processed = df.copy()
        
        # Get numeric columns (excluding target)
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        if self.target_name and self.target_name in numeric_cols:
            numeric_cols = numeric_cols.drop(self.target_name)
        
        if len(numeric_cols) > 0:
            if method == 'standard':
                from sklearn.preprocessing import StandardScaler
                scaler_class = StandardScaler
            elif method == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler_class = MinMaxScaler
            elif method == 'robust':
                from sklearn.preprocessing import RobustScaler
                scaler_class = RobustScaler
            else:
                raise ValueError(f"Unsupported scaling method: {method}")
            
            if 'feature_scaler' not in self.scalers:
                self.scalers['feature_scaler'] = scaler_class()
                df_processed[numeric_cols] = self.scalers['feature_scaler'].fit_transform(df_processed[numeric_cols])
            else:
                df_processed[numeric_cols] = self.scalers['feature_scaler'].transform(df_processed[numeric_cols])
        
        return df_processed
    
    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from numerical features
        
        Args:
            df: Input DataFrame
            method: Outlier detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        df_processed = df.copy()
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        
        # Exclude target column
        if self.target_name and self.target_name in numeric_cols:
            numeric_cols = numeric_cols.drop(self.target_name)
        
        if method == 'iqr':
            for col in numeric_cols:
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_processed = df_processed[(df_processed[col] >= lower_bound) & (df_processed[col] <= upper_bound)]
        
        elif method == 'zscore':
            from scipy import stats
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(df_processed[col]))
                df_processed = df_processed[z_scores < threshold]
        
        return df_processed
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features for credit risk modeling
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new features
        """
        df_processed = df.copy()
        
        # Common credit risk features
        try:
            # Debt-to-income ratio
            if 'limit_balance' in df_processed.columns and 'pay_amt_1' in df_processed.columns:
                df_processed['debt_to_income'] = df_processed['limit_balance'] / (df_processed['pay_amt_1'] + 1)
            
            # Payment ratio
            if 'pay_amt_1' in df_processed.columns and 'bill_amt_1' in df_processed.columns:
                df_processed['payment_ratio'] = df_processed['pay_amt_1'] / (df_processed['bill_amt_1'] + 1)
            
            # Average payment amount
            pay_columns = [col for col in df_processed.columns if col.startswith('pay_amt_')]
            if len(pay_columns) > 0:
                df_processed['avg_payment'] = df_processed[pay_columns].mean(axis=1)
            
            # Average bill amount
            bill_columns = [col for col in df_processed.columns if col.startswith('bill_amt_')]
            if len(bill_columns) > 0:
                df_processed['avg_bill'] = df_processed[bill_columns].mean(axis=1)
            
            # Payment history score
            pay_history_columns = [col for col in df_processed.columns if col.startswith('pay_') and not col.startswith('pay_amt_')]
            if len(pay_history_columns) > 0:
                df_processed['payment_history_score'] = df_processed[pay_history_columns].sum(axis=1)
            
            # Age groups
            if 'age' in df_processed.columns:
                df_processed['age_group'] = pd.cut(df_processed['age'], 
                                                 bins=[0, 25, 35, 45, 55, 100], 
                                                 labels=['18-25', '26-35', '36-45', '46-55', '55+'])
            
            # Credit utilization
            if 'limit_balance' in df_processed.columns and 'bill_amt_1' in df_processed.columns:
                df_processed['credit_utilization'] = df_processed['bill_amt_1'] / (df_processed['limit_balance'] + 1)
        
        except Exception as e:
            print(f"Warning: Feature creation failed for some features: {str(e)}")
        
        return df_processed
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'default',
                    test_size: float = 0.2, random_state: int = 42,
                    remove_outliers: bool = True, scale_features: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Complete data preparation pipeline
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
            remove_outliers: Whether to remove outliers
            scale_features: Whether to scale features
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.target_name = target_column
        
        # Make a copy of the data
        df_processed = df.copy()
        
        # Basic preprocessing
        df_processed = self.handle_missing_values(df_processed)
        df_processed = self.create_features(df_processed)
        df_processed = self.encode_categorical_variables(df_processed)
        
        # Remove outliers if requested
        if remove_outliers:
            df_processed = self.remove_outliers(df_processed)
        
        # Separate features and target
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features if requested
        if scale_features:
            X_train = self.scale_features(X_train)
            X_test = self.scale_features(X_test)
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        return X_train, X_test, y_train, y_test
    
    def load_and_split_data(self, filepath: str, target_column: str = 'default',
                           test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load data and split into train/test sets
        
        Args:
            filepath: Path to the data file
            target_column: Name of the target column
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        df = self.load_data(filepath, target_column)
        return self.prepare_data(df, target_column, test_size, random_state)
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessors
        
        Args:
            df: New data to transform
            
        Returns:
            Transformed DataFrame
        """
        df_processed = df.copy()
        
        # Apply same preprocessing steps
        df_processed = self.handle_missing_values(df_processed)
        df_processed = self.create_features(df_processed)
        df_processed = self.encode_categorical_variables(df_processed)
        df_processed = self.scale_features(df_processed)
        
        # Ensure same columns as training data
        if self.feature_names:
            # Add missing columns with zeros
            for col in self.feature_names:
                if col not in df_processed.columns:
                    df_processed[col] = 0
            
            # Remove extra columns and reorder
            df_processed = df_processed[self.feature_names]
        
        return df_processed