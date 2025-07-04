"""
Feature Analysis Module for Credit Risk Modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Optional, Any
import shap


class FeatureAnalyzer:
    """
    Comprehensive feature analysis for credit risk modeling
    """
    
    def __init__(self):
        self.feature_importance_scores = {}
        self.correlation_matrix = None
        self.statistical_tests = {}
        
    def analyze_feature_importance(self, model: Any, X: pd.DataFrame, 
                                 y: pd.Series = None, method: str = 'default') -> pd.DataFrame:
        """
        Analyze feature importance using various methods
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector (needed for some methods)
            method: Importance method ('default', 'permutation', 'shap')
            
        Returns:
            DataFrame with feature importance scores
        """
        if method == 'default':
            # Use model's built-in feature importance
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance_scores = np.abs(model.coef_[0])
            else:
                raise ValueError("Model does not have feature importance attribute")
                
        elif method == 'permutation':
            # Permutation importance
            if y is None:
                raise ValueError("Target variable y is required for permutation importance")
            perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
            importance_scores = perm_importance.importances_mean
            
        elif method == 'shap':
            # SHAP values
            try:
                explainer = shap.Explainer(model, X)
                shap_values = explainer(X)
                importance_scores = np.abs(shap_values.values).mean(0)
            except Exception as e:
                print(f"SHAP analysis failed: {e}")
                # Fallback to default method
                return self.analyze_feature_importance(model, X, y, 'default')
        else:
            raise ValueError(f"Unsupported importance method: {method}")
        
        # Create importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        self.feature_importance_scores[method] = feature_importance
        
        return feature_importance
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, 
                              top_n: int = 20, title: str = "Feature Importance"):
        """
        Plot feature importance
        
        Args:
            feature_importance: Feature importance DataFrame
            top_n: Number of top features to plot
            title: Plot title
        """
        top_features = feature_importance.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def correlation_analysis(self, X: pd.DataFrame, y: pd.Series = None, 
                           method: str = 'pearson') -> pd.DataFrame:
        """
        Perform correlation analysis
        
        Args:
            X: Feature matrix
            y: Target vector
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Correlation matrix
        """
        # Combine features and target if provided
        if y is not None:
            data = X.copy()
            data['target'] = y
        else:
            data = X.copy()
        
        # Calculate correlation matrix
        self.correlation_matrix = data.corr(method=method)
        
        return self.correlation_matrix
    
    def plot_correlation_matrix(self, figsize: Tuple[int, int] = (12, 10)):
        """
        Plot correlation heatmap
        
        Args:
            figsize: Figure size
        """
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix not computed. Run correlation_analysis first.")
        
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool))
        sns.heatmap(self.correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def find_highly_correlated_features(self, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """
        Find pairs of highly correlated features
        
        Args:
            threshold: Correlation threshold
            
        Returns:
            List of tuples (feature1, feature2, correlation)
        """
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix not computed. Run correlation_analysis first.")
        
        highly_correlated = []
        
        # Get upper triangle of correlation matrix
        upper_triangle = np.triu(np.ones_like(self.correlation_matrix), k=1).astype(bool)
        
        for i in range(len(self.correlation_matrix)):
            for j in range(i+1, len(self.correlation_matrix)):
                if upper_triangle[i, j]:
                    corr_value = self.correlation_matrix.iloc[i, j]
                    if abs(corr_value) >= threshold:
                        feature1 = self.correlation_matrix.index[i]
                        feature2 = self.correlation_matrix.columns[j]
                        highly_correlated.append((feature1, feature2, corr_value))
        
        return highly_correlated
    
    def univariate_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                                   k: int = 10, score_func=f_classif) -> pd.DataFrame:
        """
        Perform univariate feature selection
        
        Args:
            X: Feature matrix
            y: Target vector
            k: Number of top features to select
            score_func: Scoring function
            
        Returns:
            DataFrame with feature scores
        """
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)
        
        # Get feature scores
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_,
            'p_value': selector.pvalues_
        }).sort_values('score', ascending=False)
        
        self.statistical_tests['univariate'] = feature_scores
        
        return feature_scores
    
    def mutual_information_analysis(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Analyze mutual information between features and target
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            DataFrame with mutual information scores
        """
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        mi_df = pd.DataFrame({
            'feature': X.columns,
            'mutual_information': mi_scores
        }).sort_values('mutual_information', ascending=False)
        
        self.statistical_tests['mutual_information'] = mi_df
        
        return mi_df
    
    def feature_distribution_analysis(self, X: pd.DataFrame, y: pd.Series, 
                                    features: List[str] = None):
        """
        Analyze feature distributions by target class
        
        Args:
            X: Feature matrix
            y: Target vector
            features: List of features to analyze (if None, analyze all)
        """
        if features is None:
            features = X.columns[:10]  # Analyze first 10 features by default
        
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, feature in enumerate(features):
            if i < len(axes):
                ax = axes[i]
                
                # Plot distributions for each class
                for class_val in sorted(y.unique()):
                    data = X[y == class_val][feature]
                    ax.hist(data, alpha=0.7, label=f'Class {class_val}', bins=30)
                
                ax.set_title(f'Distribution of {feature}')
                ax.set_xlabel(feature)
                ax.set_ylabel('Frequency')
                ax.legend()
        
        # Hide empty subplots
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def feature_target_relationship(self, X: pd.DataFrame, y: pd.Series, 
                                  categorical_features: List[str] = None,
                                  numerical_features: List[str] = None):
        """
        Analyze relationship between features and target
        
        Args:
            X: Feature matrix
            y: Target vector
            categorical_features: List of categorical features
            numerical_features: List of numerical features
        """
        if categorical_features is None:
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if numerical_features is None:
            numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Categorical features analysis
        if categorical_features:
            n_cat = len(categorical_features)
            n_cols = 2
            n_rows = (n_cat + n_cols - 1) // n_cols
            
            if n_cat > 0:
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
                if n_rows == 1:
                    axes = [axes] if n_cols == 1 else axes
                else:
                    axes = axes.flatten()
                
                for i, feature in enumerate(categorical_features[:len(axes)]):
                    ax = axes[i]
                    
                    # Create crosstab
                    crosstab = pd.crosstab(X[feature], y, normalize='index')
                    crosstab.plot(kind='bar', ax=ax, rot=45)
                    ax.set_title(f'{feature} vs Target')
                    ax.set_ylabel('Proportion')
                
                # Hide empty subplots
                for i in range(n_cat, len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                plt.show()
        
        # Numerical features analysis
        if numerical_features:
            n_num = min(len(numerical_features), 9)  # Limit to 9 features
            n_cols = 3
            n_rows = (n_num + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, feature in enumerate(numerical_features[:n_num]):
                ax = axes[i]
                
                # Box plot for each class
                data_to_plot = [X[y == class_val][feature] for class_val in sorted(y.unique())]
                ax.boxplot(data_to_plot, labels=[f'Class {val}' for val in sorted(y.unique())])
                ax.set_title(f'{feature} by Target Class')
                ax.set_ylabel(feature)
            
            # Hide empty subplots
            for i in range(n_num, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.show()
    
    def feature_selection_comparison(self, X: pd.DataFrame, y: pd.Series, 
                                   model: Any, top_k: int = 20) -> pd.DataFrame:
        """
        Compare different feature selection methods
        
        Args:
            X: Feature matrix
            y: Target vector
            model: Trained model
            top_k: Number of top features to compare
            
        Returns:
            DataFrame comparing different methods
        """
        results = pd.DataFrame({'feature': X.columns})
        
        # Model-based importance
        try:
            model_importance = self.analyze_feature_importance(model, X, y, 'default')
            results = results.merge(
                model_importance[['feature', 'importance']].rename(columns={'importance': 'model_importance'}),
                on='feature', how='left'
            )
        except Exception as e:
            print(f"Model importance failed: {e}")
        
        # Univariate selection
        try:
            univariate_scores = self.univariate_feature_selection(X, y, k=len(X.columns))
            results = results.merge(
                univariate_scores[['feature', 'score']].rename(columns={'score': 'univariate_score'}),
                on='feature', how='left'
            )
        except Exception as e:
            print(f"Univariate selection failed: {e}")
        
        # Mutual information
        try:
            mi_scores = self.mutual_information_analysis(X, y)
            results = results.merge(
                mi_scores[['feature', 'mutual_information']],
                on='feature', how='left'
            )
        except Exception as e:
            print(f"Mutual information failed: {e}")
        
        # Correlation with target
        try:
            correlation_with_target = X.corrwith(y).abs()
            results['correlation_with_target'] = results['feature'].map(correlation_with_target)
        except Exception as e:
            print(f"Correlation analysis failed: {e}")
        
        return results.fillna(0)
    
    def plot_feature_selection_comparison(self, comparison_df: pd.DataFrame, top_n: int = 15):
        """
        Plot comparison of feature selection methods
        
        Args:
            comparison_df: DataFrame from feature_selection_comparison
            top_n: Number of top features to plot
        """
        # Normalize scores to 0-1 range for comparison
        numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
        normalized_df = comparison_df.copy()
        
        for col in numeric_cols:
            if col != 'feature':
                max_val = normalized_df[col].max()
                if max_val > 0:
                    normalized_df[col] = normalized_df[col] / max_val
        
        # Get top features based on average score
        normalized_df['average_score'] = normalized_df[numeric_cols].mean(axis=1)
        top_features = normalized_df.nlargest(top_n, 'average_score')
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(top_features))
        width = 0.2
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, col in enumerate(numeric_cols):
            if col != 'average_score':
                ax.bar(x + i * width, top_features[col], width, 
                      label=col.replace('_', ' ').title(), color=colors[i % len(colors)])
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Normalized Score')
        ax.set_title('Feature Selection Methods Comparison')
        ax.set_xticks(x + width * (len(numeric_cols) - 2) / 2)
        ax.set_xticklabels(top_features['feature'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_feature_report(self, X: pd.DataFrame, y: pd.Series, model: Any) -> Dict[str, Any]:
        """
        Generate comprehensive feature analysis report
        
        Args:
            X: Feature matrix
            y: Target vector
            model: Trained model
            
        Returns:
            Dictionary containing analysis results
        """
        report = {}
        
        # Basic statistics
        report['basic_stats'] = {
            'n_features': len(X.columns),
            'n_samples': len(X),
            'missing_values': X.isnull().sum().to_dict(),
            'feature_types': X.dtypes.to_dict()
        }
        
        # Feature importance
        try:
            report['feature_importance'] = self.analyze_feature_importance(model, X, y)
        except Exception as e:
            report['feature_importance_error'] = str(e)
        
        # Correlation analysis
        try:
            report['correlation_matrix'] = self.correlation_analysis(X, y)
            report['highly_correlated_pairs'] = self.find_highly_correlated_features()
        except Exception as e:
            report['correlation_error'] = str(e)
        
        # Statistical tests
        try:
            report['univariate_scores'] = self.univariate_feature_selection(X, y)
            report['mutual_information'] = self.mutual_information_analysis(X, y)
        except Exception as e:
            report['statistical_tests_error'] = str(e)
        
        # Feature selection comparison
        try:
            report['feature_comparison'] = self.feature_selection_comparison(X, y, model)
        except Exception as e:
            report['feature_comparison_error'] = str(e)
        
        return report