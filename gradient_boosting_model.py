import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GradientBoostingModel:

    def __init__(self, model_type='regression'):
        self.model_type = model_type
        self.model = None
        self.feature_names = []
        self.metrics = {}
        
        if model_type == 'regression':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            )
    
    def train(self, X, y, feature_names=None):
   
        try:
            self.feature_names = feature_names if feature_names is not None else [f"feature_{i}" for i in range(X.shape[1])]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if len(X_train) > 30:
                logger.info("Performing hyperparameter tuning for Gradient Boosting model")
                
                if len(X_train) < 100:
                    param_grid = {
                        'n_estimators': [50, 100],
                        'learning_rate': [0.05, 0.1],
                        'max_depth': [2, 3]
                    }
                else:
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'max_depth': [2, 3, 4],
                        'min_samples_split': [2, 5],
                        'subsample': [0.8, 1.0]
                    }
                
                grid_search = GridSearchCV(
                    estimator=self.model,
                    param_grid=param_grid,
                    cv=min(5, len(X_train) // 10),  # Ensure we have enough samples per fold
                    scoring='neg_mean_squared_error' if self.model_type == 'regression' else 'accuracy',
                    n_jobs=-1
                )
                
                grid_search.fit(X_train, y_train)
                
                self.model = grid_search.best_estimator_
                logger.info(f"Best parameters: {grid_search.best_params_}")
            else:
                if len(X_train) < 20:
                    if self.model_type == 'regression':
                        self.model = GradientBoostingRegressor(
                            n_estimators=50,
                            learning_rate=0.1,
                            max_depth=2,
                            min_samples_split=2,
                            random_state=42
                        )
                    else:
                        self.model = GradientBoostingClassifier(
                            n_estimators=50,
                            learning_rate=0.1,
                            max_depth=2,
                            min_samples_split=2,
                            random_state=42
                        )   
                logger.info("Training Gradient Boosting model with default parameters")
                self.model.fit(X_train, y_train)
            
            y_pred = self.model.predict(X_test)
            
            if self.model_type == 'regression':
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                self.metrics = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'test_size': len(y_test)
                }
                
                logger.info(f"Model evaluation - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
            else:
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                self.metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'test_size': len(y_test)
                }
                
                logger.info(f"Model evaluation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error training Gradient Boosting model: {e}")
            return {}
    
    def predict(self, X):
       
        if self.model is None:
            logger.error("Model not trained yet")
            return np.array([])
        
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return np.array([])
    
    def get_feature_importance(self):
       
        if self.model is None:
            logger.error("Model not trained yet")
            return pd.DataFrame()
        
        try:
            importance = self.model.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importance
            })
            
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            importance_df['Rank'] = range(1, len(importance_df) + 1)
            
            def categorize_feature(feature_name):
                if 'AVG' in feature_name:
                    return 'Average'
                elif 'STD' in feature_name:
                    return 'Variability'
                elif 'EMA' in feature_name:
                    return 'Trend'
                elif 'OPP' in feature_name:
                    return 'Opponent'
                elif 'REST' in feature_name or 'BACK_TO_BACK' in feature_name:
                    return 'Rest'
                elif 'HOME' in feature_name:
                    return 'Location'
                elif 'SEASON' in feature_name:
                    return 'Season Phase'
                else:
                    return 'Other'
            
            importance_df['Category'] = importance_df['Feature'].apply(categorize_feature)
            
            importance_df['Cumulative_Importance'] = importance_df['Importance'].cumsum()
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return pd.DataFrame()
    
    def save_model(self, filepath):
        if self.model is None:
            logger.error("Model not trained yet")
            return False
        
        try:
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'metrics': self.metrics,
                'model_type': self.model_type
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    @classmethod
    def load_model(cls, filepath):
        try:
            model_data = joblib.load(filepath)
            
            instance = cls(model_type=model_data['model_type'])
            instance.model = model_data['model']
            instance.feature_names = model_data['feature_names']
            instance.metrics = model_data['metrics']
            
            logger.info(f"Model loaded from {filepath}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
