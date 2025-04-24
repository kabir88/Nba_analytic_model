import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RandomForestModel:
    def __init__(self, model_type='regression', n_estimators=100, random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.n_estimators = n_estimators
        
        if model_type == 'regression':
            self.model = RandomForestRegressor(
                n_estimators=n_estimators, 
                random_state=random_state
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators, 
                random_state=random_state
            )
        
        self.metrics = {}
        self.feature_names = []
    
    def train(self, X, y, feature_names=None, test_size=0.2, auto_optimize=True):
        try:
            if len(X) == 0 or len(y) == 0:
                logger.warning("Empty training data provided")
                return {}
            
            if feature_names:
                self.feature_names = feature_names
                
            if len(X) < 10:
                logger.warning(f"Very limited data available ({len(X)} samples), using simplified model")
                if self.model_type == 'regression':
                    self.model = RandomForestRegressor(
                        n_estimators=max(5, min(50, len(X) * 5)),
                        max_depth=3,
                        min_samples_leaf=1,
                        random_state=self.random_state
                    )
                else:
                    self.model = RandomForestClassifier(
                        n_estimators=max(5, min(50, len(X) * 5)),
                        max_depth=3,
                        min_samples_leaf=1,
                        class_weight='balanced',
                        random_state=self.random_state
                    )
                    
                if len(X) < 5:
                    self.model.fit(X, y)
                    logger.info("Trained with all available data points due to very small dataset")
                    
                    y_train_pred = self.model.predict(X)
                    if self.model_type == 'regression':
                        self.metrics = {
                            'MAE': mean_absolute_error(y, y_train_pred),
                            'RMSE': np.sqrt(mean_squared_error(y, y_train_pred)),
                            'r2': r2_score(y, y_train_pred) if len(np.unique(y)) > 1 else 0,
                            'Test_Size': 0,
                            'Data_size': len(X),
                            'Limited_data': True
                        }
                    else:
                        self.metrics = {
                            'Accuracy': accuracy_score(y, y_train_pred),
                            'Test_size': 0,
                            'data_size': len(X),
                            'limited_data': True
                        }
                    return self.metrics
                else:
                    test_size = max(1, int(len(X) * 0.2)) 
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=self.random_state
                    )
                    self.model.fit(X_train, y_train)
                    logger.info(f"Trained with {len(X_train)} samples, testing with {len(X_test)} samples")
                    
                    y_pred = self.model.predict(X_test)
                    if self.model_type == 'regression':
                        self.metrics = {
                            'MAE': mean_absolute_error(y_test, y_pred),
                            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                            'r2': r2_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else 0,
                            'test_size': len(y_test),
                            'data_size': len(X),
                            'limited_data': True
                        }
                    else:
                        self.metrics = {
                            'accuracy': accuracy_score(y_test, y_pred),
                            'test_size': len(y_test),
                            'data_size': len(X),
                            'limited_data': True
                        }
                    return self.metrics
            
            if auto_optimize and len(X) >= 50:
                logger.info("Auto-optimizing hyperparameters...")
                
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=test_size, random_state=self.random_state
                )
                
                if self.model_type == 'regression':
                    base_model = RandomForestRegressor(random_state=self.random_state)
                    scoring = 'r2'
                else:
                    base_model = RandomForestClassifier(random_state=self.random_state, class_weight='balanced')
                    scoring = 'f1_weighted'
                
                if len(X_train) < 30:
                    n_cv = 2 
                else:
                    n_cv = min(3, len(X_train) // 10) 
                n_cv = max(2, n_cv)
                
                logger.info(f"Using {n_cv}-fold cross-validation due to dataset size of {len(X_train)} samples")
            
                grid_search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=n_cv, 
                    scoring=scoring,
                    n_jobs=-1,
                    verbose=1
                )           
                grid_search.fit(X_train, y_train)               
                best_params = grid_search.best_params_
                logger.info(f"Optimal hyperparameters found: {best_params}")
                
                if self.model_type == 'regression':
                    self.model = RandomForestRegressor(random_state=self.random_state, **best_params)
                else:
                    self.model = RandomForestClassifier(random_state=self.random_state, class_weight='balanced', **best_params)
                self.model.fit(X_train, y_train)
                
                self.best_params = best_params
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=self.random_state
                )
                
                if len(X) < 50:
                    logger.info("Small dataset detected, using enhanced default settings")
                    if self.model_type == 'regression':
                        self.model = RandomForestRegressor(
                            n_estimators=300,
                            max_depth=None,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            max_features='sqrt',
                            bootstrap=True,
                            random_state=self.random_state,
                            n_jobs=-1
                        )
                    else:
                        self.model = RandomForestClassifier(
                            n_estimators=300,
                            max_depth=None,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            max_features='sqrt',
                            bootstrap=True,
                            class_weight='balanced',
                            random_state=self.random_state,
                            n_jobs=-1
                        )
                self.model.fit(X_train, y_train)
            if auto_optimize and len(X) >= 50:
                X_test_final = X_val
                y_test_final = y_val
            else:
                X_test_final = X_test
                y_test_final = y_test
            y_pred = self.model.predict(X_test_final)
            
            if self.model_type == 'regression':
                self.metrics = {
                    'MAE': mean_absolute_error(y_test_final, y_pred),
                    'mse': mean_squared_error(y_test_final, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test_final, y_pred)),
                    'r2': r2_score(y_test_final, y_pred),
                    'test_size': len(y_test_final)
                }
                logger.info(f"Enhanced model trained with MAE: {self.metrics['MAE']:.2f}, RMSE: {self.metrics['RMSE']:.2f}, r2: {self.metrics['r2']:.2f}")
            else:
                self.metrics = {
                    'accuracy': accuracy_score(y_test_final, y_pred),
                    'precision': precision_score(y_test_final, y_pred, average='weighted'),
                    'recall': recall_score(y_test_final, y_pred, average='weighted'),
                    'f1': f1_score(y_test_final, y_pred, average='weighted'),
                    'test_size': len(y_test_final)
                }
                logger.info(f"Enhanced model trained with Accuracy: {self.metrics['accuracy']:.2f}, F1: {self.metrics['f1']:.2f}")
            
            if hasattr(self.model, 'feature_importances_') and self.feature_names:
                importances = self.model.feature_importances_
                indices = np.argsort(importances)[::-1]
                top_n = min(10, len(self.feature_names))
                
                logger.info(f"Top {top_n} most important features:")
                for i in range(top_n):
                    feature_idx = indices[i]
                    if feature_idx < len(self.feature_names):
                        logger.info(f"{self.feature_names[feature_idx]}: {importances[feature_idx]:.4f}")
            
            return self.metrics
        
        except Exception as e:
            logger.error(f"Error training enhanced Random Forest model: {e}")
            if "n_splits" in str(e) and "cannot be greater than" in str(e):
                logger.warning("Cross-validation error due to limited samples per class. Using simplified metrics.")
                if self.model_type == 'regression':
                    self.model = RandomForestRegressor(
                        n_estimators=50,
                        max_depth=5,
                        min_samples_leaf=2,
                        random_state=self.random_state
                    )
                    self.model.fit(X, y)
                    y_pred = self.model.predict(X)
                    self.metrics = {
                        'MAE': mean_absolute_error(y, y_pred),
                        'mse': mean_squared_error(y, y_pred),
                        'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
                        'r2': r2_score(y, y_pred),
                        'test_size': 0,
                        'data_size': len(X),
                        'limited_data': True
                    }
                    return self.metrics
                else:
                    self.model = RandomForestClassifier(
                        n_estimators=50,
                        max_depth=5,
                        min_samples_leaf=2,
                        random_state=self.random_state
                    )
                    self.model.fit(X, y)  
                    y_pred = self.model.predict(X)
                    self.metrics = {
                        'accuracy': accuracy_score(y, y_pred),
                        'test_size': 0,
                        'data_size': len(X),
                        'limited_data': True
                    }
                    return self.metrics
            return {}
    
    def optimize_hyperparameters(self, X, y, test_size=0.2):
        try:
            if len(X) == 0 or len(y) == 0:
                logger.warning("Empty training data provided for hyperparameter optimization")
                return {}
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
            
            param_grid = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [None, 10, 20, 30, 50, 70],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False],
                'criterion': ['gini', 'entropy'] if self.model_type == 'classification' else ['squared_error', 'absolute_error', 'poisson']
            }
            
            if len(X_train) < 50:
                n_cv = 2 
            elif len(X_train) < 100:
                n_cv = 3 
            else:
                n_cv = min(5, len(X_train) // 15) 
            n_cv = max(2, n_cv)
            
            logger.info(f"Using {n_cv}-fold cross-validation due to dataset size of {len(X_train)} samples")
            
            if self.model_type == 'regression':
                grid_search = GridSearchCV(
                    RandomForestRegressor(random_state=self.random_state),
                    param_grid,
                    cv=n_cv,
                    scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
                    refit='r2', 
                    n_jobs=-1,
                    verbose=1
                )
            else:
                grid_search = GridSearchCV(
                    RandomForestClassifier(random_state=self.random_state, class_weight='balanced'),
                    param_grid,
                    cv=n_cv,
                    scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'],
                    refit='f1_weighted',
                    n_jobs=-1,
                    verbose=1
                )
            
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            logger.info(f"Best hyperparameters: {best_params}")
            
            if self.model_type == 'regression':
                self.model = RandomForestRegressor(random_state=self.random_state, **best_params)
            else:
                self.model = RandomForestClassifier(random_state=self.random_state, class_weight='balanced', **best_params)
            
            self.model.fit(X_train, y_train)
            
            y_pred = self.model.predict(X_test)
            
            if self.model_type == 'regression':
                self.metrics = {
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'r2': r2_score(y_test, y_pred),
                    'test_size': len(y_test)
                }
                logger.info(f"Optimized model metrics: MAE={self.metrics['MAE']:.2f}, RMSE={self.metrics['RMSE']:.2f}, R²={self.metrics['r2']:.2f}")
            else:
                self.metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1': f1_score(y_test, y_pred, average='weighted'),
                    'test_size': len(y_test)
                }
                logger.info(f"Optimized model metrics: Accuracy={self.metrics['accuracy']:.2f}, F1={self.metrics['f1']:.2f}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {e}")
            return {}
    
    def predict(self, X):
        try:
            if X is None or len(X) == 0:
                logger.warning("Empty prediction data provided")
                return None
            
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error predicting with Random Forest: {e}")
            return None
    
    def get_feature_importances(self, normalized=True, limit=None):
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return {}
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        features = {}
        if self.feature_names:
            for i in range(min(len(self.feature_names), len(importances))):
                idx = indices[i]
                if idx < len(self.feature_names):
                    feature_name = self.feature_names[idx]
                    features[feature_name] = float(importances[idx])
                if limit and i >= limit - 1:
                    break
        else:
            for i in range(len(importances)):
                idx = indices[i]
                features[f"feature_{idx}"] = float(importances[idx])
                if limit and i >= limit - 1:
                    break
        
        if normalized and features:
            max_importance = max(features.values())
            if max_importance > 0:
                for feature in features:
                    features[feature] /= max_importance
        
        return features
    
    def save_model(self, filepath):
        try:
            if not os.path.exists(os.path.dirname(filepath)):
                os.makedirs(os.path.dirname(filepath))
            
            model_data = {
                'model': self.model,
                'metrics': self.metrics,
                'feature_names': self.feature_names,
                'model_type': self.model_type
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath):
        try:
            if not os.path.exists(filepath):
                logger.error(f"Model file {filepath} does not exist")
                return False
                
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.metrics = model_data['metrics']
            self.feature_names = model_data['feature_names']
            self.model_type = model_data['model_type']
            
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

class ReinforcementLearningModel:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.95, random_state=42):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.random_state = random_state
        self.q_table = {}
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.metrics = {}
        self.feature_names = []
        
        np.random.seed(random_state)
    
    def _get_state_key(self, state):
        if isinstance(state, np.ndarray):
            return tuple(state.flatten())
        else:
            return tuple(state)
    
    def train(self, X, y, feature_names=None, episodes=1000, epsilon=0.1, test_size=0.2):
        try:
            if len(X) == 0 or len(y) == 0:
                logger.warning("Empty training data provided")
                return {}
            
            if feature_names:
                self.feature_names = feature_names
                
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
            
            for episode in range(episodes):
                if episode % (episodes // 10) == 0:
                    logger.info(f"Training RL model: episode {episode}/{episodes}")
                
                for i in range(len(X_train)):
                    state_key = self._get_state_key(X_train[i])
                    
                    if state_key not in self.q_table:
                        self.q_table[state_key] = np.zeros(self.action_size)
                    
                    if np.random.random() < epsilon:
                        action = np.random.randint(0, self.action_size)
                    else:
                        action = np.argmax(self.q_table[state_key])
                    
                    predicted_value = (action + 1) / self.action_size * np.max(y_train)
                    reward = -np.abs(predicted_value - y_train[i])
                    
                    if i < len(X_train) - 1:
                        next_state_key = self._get_state_key(X_train[i+1])
                        if next_state_key not in self.q_table:
                            self.q_table[next_state_key] = np.zeros(self.action_size)
                        
                        max_future_q = np.max(self.q_table[next_state_key])
                        current_q = self.q_table[state_key][action]
                        
                        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
                        self.q_table[state_key][action] = new_q
                    else:
                        self.q_table[state_key][action] = reward
                    
                    self.state_history.append(state_key)
                    self.action_history.append(action)
                    self.reward_history.append(reward)
            
            logger.info(f"RL model trained on {len(X_train)} samples over {episodes} episodes")
            
            y_pred = self.predict(X_test)
            
            if y_pred is not None and len(y_pred) > 0 and len(y_test) == len(y_pred):
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else 0
                
                self.metrics = {
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'R2': r2,
                    'test_size': len(y_test)
                }
                
                logger.info(f"RL model trained with MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
                
                return self.metrics
            else:
                logger.warning("Unable to calculate metrics for RL model due to prediction issues")
                return {}
        
        except Exception as e:
            logger.error(f"Error training RL model: {e}")
            return {}
    
    def predict(self, X):
        try:
            if X is None or len(X) == 0:
                logger.warning("Empty prediction data provided")
                return None
            
            y_pred = []
            
            for i in range(len(X)):
                state_key = self._get_state_key(X[i])
                
                if state_key in self.q_table:
                    action = np.argmax(self.q_table[state_key])
                    predicted_value = (action + 1) / self.action_size * 50
                    y_pred.append(predicted_value)
                else:
                    similar_state = None
                    min_distance = float('inf')
                    
                    for known_state in list(self.q_table.keys())[:100]:
                        if len(known_state) == len(state_key):
                            distance = np.sum(np.abs(np.array(known_state) - np.array(state_key)))
                            if distance < min_distance:
                                min_distance = distance
                                similar_state = known_state
                    
                    if similar_state is not None:
                        action = np.argmax(self.q_table[similar_state])
                        predicted_value = (action + 1) / self.action_size * 50
                        y_pred.append(predicted_value)
                    else:
                        y_pred.append(np.mean(y_pred) if len(y_pred) > 0 else 25)
            
            return np.array(y_pred)
            
        except Exception as e:
            logger.error(f"Error predicting with RL model: {e}")
            return None
    
    def get_feature_importances(self, normalized=True, limit=None):
        if not self.feature_names:
            logger.warning("No feature names available")
            return {}
        
        feature_importance = {}
        
        for feature_idx in range(min(self.state_size, len(self.feature_names))):
            feature_name = self.feature_names[feature_idx]
            importance = 0
            
            for state_key in self.q_table:
                if len(state_key) > feature_idx:
                    importance += np.abs(state_key[feature_idx]) * np.max(self.q_table[state_key])
            
            feature_importance[feature_name] = float(importance)
        
        if feature_importance:
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            if limit:
                sorted_importance = sorted_importance[:limit]
            
            result = dict(sorted_importance)
            
            if normalized and result:
                max_importance = max(result.values())
                if max_importance > 0:
                    for feature in result:
                        result[feature] /= max_importance
            
            return result
        else:
            return {}
    
    def save_model(self, filepath):
        try:
            if not os.path.exists(os.path.dirname(filepath)):
                os.makedirs(os.path.dirname(filepath))
            
            model_data = {
                'q_table': self.q_table,
                'state_size': self.state_size,
                'action_size': self.action_size,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'random_state': self.random_state,
                'metrics': self.metrics,
                'feature_names': self.feature_names
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"RL model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving RL model: {e}")
            return False
    
    def load_model(self, filepath):
        try:
            if not os.path.exists(filepath):
                logger.error(f"Model file {filepath} does not exist")
                return False
                
            model_data = joblib.load(filepath)
            
            self.q_table = model_data['q_table']
            self.state_size = model_data['state_size']
            self.action_size = model_data['action_size']
            self.learning_rate = model_data['learning_rate']
            self.discount_factor = model_data['discount_factor']
            self.random_state = model_data['random_state']
            self.metrics = model_data['metrics']
            self.feature_names = model_data['feature_names']
            
            logger.info(f"RL model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading RL model: {e}")
            return False

class GradientBoostingModel:
    def __init__(self, model_type='regression', n_estimators=100, learning_rate=0.1, random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        
        if model_type == 'regression':
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators, 
                learning_rate=learning_rate,
                random_state=random_state
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators, 
                learning_rate=learning_rate,
                random_state=random_state
            )
        
        self.metrics = {}
        self.feature_names = []
    
    def train(self, X, y, feature_names=None, test_size=0.2, auto_optimize=True):
        try:
            if len(X) == 0 or len(y) == 0:
                logger.warning("Empty training data provided")
                return {}
            
            if feature_names:
                self.feature_names = feature_names
                
            if len(X) < 10:
                logger.warning(f"Very limited data available ({len(X)} samples), using simplified model")
                if self.model_type == 'regression':
                    self.model = GradientBoostingRegressor(
                        n_estimators=max(5, min(50, len(X) * 5)),
                        max_depth=3,
                        learning_rate=0.1,
                        random_state=self.random_state
                    )
                else:
                    self.model = GradientBoostingClassifier(
                        n_estimators=max(5, min(50, len(X) * 5)),
                        max_depth=3,
                        learning_rate=0.1,
                        random_state=self.random_state
                    )
                    
                if len(X) < 5:
                    self.model.fit(X, y)
                    logger.info("Trained with all available data points due to very small dataset")
                    
                    y_train_pred = self.model.predict(X)
                    if self.model_type == 'regression':
                        self.metrics = {
                            'MAE': mean_absolute_error(y, y_train_pred),
                            'RMSE': np.sqrt(mean_squared_error(y, y_train_pred)),
                            'r2': r2_score(y, y_train_pred) if len(np.unique(y)) > 1 else 0,
                            'Test_Size': 0,
                            'Data_size': len(X),
                            'Limited_data': True
                        }
                    else:
                        self.metrics = {
                            'Accuracy': accuracy_score(y, y_train_pred),
                            'Test_size': 0,
                            'data_size': len(X),
                            'limited_data': True
                        }
                    return self.metrics
                else:
                    test_size = max(1, int(len(X) * 0.2)) 
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=self.random_state
                    )
                    self.model.fit(X_train, y_train)
                    logger.info(f"Trained with {len(X_train)} samples, testing with {len(X_test)} samples")
                    
                    y_pred = self.model.predict(X_test)
                    if self.model_type == 'regression':
                        self.metrics = {
                            'MAE': mean_absolute_error(y_test, y_pred),
                            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                            'r2': r2_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else 0,
                            'test_size': len(y_test),
                            'data_size': len(X),
                            'limited_data': True
                        }
                    else:
                        self.metrics = {
                            'accuracy': accuracy_score(y_test, y_pred),
                            'test_size': len(y_test),
                            'data_size': len(X),
                            'limited_data': True
                        }
                    return self.metrics
            
            if auto_optimize and len(X) >= 50:
                logger.info("Auto-optimizing hyperparameters...")
                
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, 9],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=test_size, random_state=self.random_state
                )
                
                if self.model_type == 'regression':
                    base_model = GradientBoostingRegressor(random_state=self.random_state)
                    scoring = 'r2'
                else:
                    base_model = GradientBoostingClassifier(random_state=self.random_state)
                    scoring = 'f1_weighted'
                
                if len(X_train) < 30:
                    n_cv = 2 
                else:
                    n_cv = min(3, len(X_train) // 10) 
                n_cv = max(2, n_cv)
                
                logger.info(f"Using {n_cv}-fold cross-validation due to dataset size of {len(X_train)} samples")
            
                grid_search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=n_cv, 
                    scoring=scoring,
                    n_jobs=-1,
                    verbose=1
                )           
                grid_search.fit(X_train, y_train)               
                best_params = grid_search.best_params_
                logger.info(f"Optimal hyperparameters found: {best_params}")
                
                if self.model_type == 'regression':
                    self.model = GradientBoostingRegressor(random_state=self.random_state, **best_params)
                else:
                    self.model = GradientBoostingClassifier(random_state=self.random_state, **best_params)
                self.model.fit(X_train, y_train)
                
                self.best_params = best_params
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=self.random_state
                )
                
                if len(X) < 50:
                    logger.info("Small dataset detected, using enhanced default settings")
                    if self.model_type == 'regression':
                        self.model = GradientBoostingRegressor(
                            n_estimators=200,
                            max_depth=5,
                            learning_rate=0.1,
                            min_samples_split=5,
                            min_samples_leaf=2,
                            random_state=self.random_state
                        )
                    else:
                        self.model = GradientBoostingClassifier(
                            n_estimators=200,
                            max_depth=5,
                            learning_rate=0.1,
                            min_samples_split=5,
                            min_samples_leaf=2,
                            random_state=self.random_state
                        )
                self.model.fit(X_train, y_train)
            if auto_optimize and len(X) >= 50:
                X_test_final = X_val
                y_test_final = y_val
            else:
                X_test_final = X_test
                y_test_final = y_test
            y_pred = self.model.predict(X_test_final)
            
            if self.model_type == 'regression':
                self.metrics = {
                    'MAE': mean_absolute_error(y_test_final, y_pred),
                    'mse': mean_squared_error(y_test_final, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test_final, y_pred)),
                    'r2': r2_score(y_test_final, y_pred),
                    'test_size': len(y_test_final)
                }
                logger.info(f"Enhanced model trained with MAE: {self.metrics['MAE']:.2f}, RMSE: {self.metrics['RMSE']:.2f}, r2: {self.metrics['r2']:.2f}")
            else:
                self.metrics = {
                    'accuracy': accuracy_score(y_test_final, y_pred),
                    'precision': precision_score(y_test_final, y_pred, average='weighted'),
                    'recall': recall_score(y_test_final, y_pred, average='weighted'),
                    'f1': f1_score(y_test_final, y_pred, average='weighted'),
                    'test_size': len(y_test_final)
                }
                logger.info(f"Enhanced model trained with Accuracy: {self.metrics['accuracy']:.2f}, F1: {self.metrics['f1']:.2f}")
            
            if hasattr(self.model, 'feature_importances_') and self.feature_names:
                importances = self.model.feature_importances_
                indices = np.argsort(importances)[::-1]
                top_n = min(10, len(self.feature_names))
                
                logger.info(f"Top {top_n} most important features:")
                for i in range(top_n):
                    feature_idx = indices[i]
                    if feature_idx < len(self.feature_names):
                        logger.info(f"{self.feature_names[feature_idx]}: {importances[feature_idx]:.4f}")
            
            return self.metrics
        
        except Exception as e:
            logger.error(f"Error training enhanced Gradient Boosting model: {e}")
            if "n_splits" in str(e) and "cannot be greater than" in str(e):
                logger.warning("Cross-validation error due to limited samples per class. Using simplified metrics.")
                if self.model_type == 'regression':
                    self.model = GradientBoostingRegressor(
                        n_estimators=50,
                        max_depth=3,
                        learning_rate=0.1,
                        random_state=self.random_state
                    )
                    self.model.fit(X, y)
                    y_pred = self.model.predict(X)
                    self.metrics = {
                        'MAE': mean_absolute_error(y, y_pred),
                        'mse': mean_squared_error(y, y_pred),
                        'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
                        'r2': r2_score(y, y_pred),
                        'test_size': 0,
                        'data_size': len(X),
                        'limited_data': True
                    }
                    return self.metrics
                else:
                    self.model = GradientBoostingClassifier(
                        n_estimators=50,
                        max_depth=3,
                        learning_rate=0.1,
                        random_state=self.random_state
                    )
                    self.model.fit(X, y)  
                    y_pred = self.model.predict(X)
                    self.metrics = {
                        'accuracy': accuracy_score(y, y_pred),
                        'test_size': 0,
                        'data_size': len(X),
                        'limited_data': True
                    }
                    return self.metrics
            return {}
    
    def predict(self, X):
        try:
            if X is None or len(X) == 0:
                logger.warning("Empty prediction data provided")
                return None
            
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error predicting with Gradient Boosting: {e}")
            return None
    
    def get_feature_importances(self, normalized=True, limit=None):
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return {}
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        features = {}
        if self.feature_names:
            for i in range(min(len(self.feature_names), len(importances))):
                idx = indices[i]
                if idx < len(self.feature_names):
                    feature_name = self.feature_names[idx]
                    features[feature_name] = float(importances[idx])
                if limit and i >= limit - 1:
                    break
        else:
            for i in range(len(importances)):
                idx = indices[i]
                features[f"feature_{idx}"] = float(importances[idx])
                if limit and i >= limit - 1:
                    break
        
        if normalized and features:
            max_importance = max(features.values())
            if max_importance > 0:
                for feature in features:
                    features[feature] /= max_importance
        
        return features
    
    def save_model(self, filepath):
        try:
            if not os.path.exists(os.path.dirname(filepath)):
                os.makedirs(os.path.dirname(filepath))
            
            model_data = {
                'model': self.model,
                'metrics': self.metrics,
                'feature_names': self.feature_names,
                'model_type': self.model_type
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath):
        try:
            if not os.path.exists(filepath):
                logger.error(f"Model file {filepath} does not exist")
                return False
                
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.metrics = model_data['metrics']
            self.feature_names = model_data['feature_names']
            self.model_type = model_data['model_type']
            
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

class ModelEvaluator:
    def __init__(self):
        self.models = {}
        self.evaluation_results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_metric = None
        self.best_score = float('-inf')
        self.feature_names = []
        
    def add_model(self, name, model):
        self.models[name] = model
        logger.info(f"Added model: {name}")
        
    def evaluate_all(self, X, y, X_test=None, y_test=None, cv=5, metric='r2', test_size=0.2, random_state=42):
        results = {}
        
        try:
            if not self.models:
                logger.warning("No models to evaluate")
                return {}
                
            if len(X) == 0 or len(y) == 0:
                logger.warning("Empty evaluation data provided")
                return {}
            
            if X_test is None or y_test is None:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
            else:
                X_train, y_train = X, y
                
            logger.info(f"Evaluating {len(self.models)} models on {len(X_train)} training and {len(X_test)} testing samples")
            
            for name, model in self.models.items():
                logger.info(f"Evaluating model: {name}")
                
                if hasattr(model, 'train'):
                    if hasattr(model, 'feature_names'):
                        model_metrics = model.train(X_train, y_train, test_size=0.2)
                    else:
                        model_metrics = model.train(X_train, y_train)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    if metric == 'r2':
                        score = r2_score(y_test, y_pred)
                    elif metric == 'mae':
                        score = -mean_absolute_error(y_test, y_pred)
                    elif metric == 'rmse':
                        score = -np.sqrt(mean_squared_error(y_test, y_pred))
                    elif metric == 'accuracy':
                        score = accuracy_score(y_test, y_pred)
                    elif metric == 'f1':
                        score = f1_score(y_test, y_pred, average='weighted')
                    else:
                        score = r2_score(y_test, y_pred)
                    
                    model_metrics = {
                        'test_score': score,
                        'MAE': mean_absolute_error(y_test, y_pred),
                        'MSE': mean_squared_error(y_test, y_pred),
                        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'R2': r2_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else 0,
                    }
                    
                if hasattr(model, 'metrics') and model.metrics:
                    model_metrics.update(model.metrics)
                    
                results[name] = model_metrics
                
                metric_score = None
                if metric.lower() == 'r2' and 'r2' in model_metrics:
                    metric_score = model_metrics['r2']
                elif metric.lower() == 'mae' and 'MAE' in model_metrics:
                    metric_score = -model_metrics['MAE']
                elif metric.lower() == 'rmse' and 'RMSE' in model_metrics:
                    metric_score = -model_metrics['RMSE']
                elif metric.lower() == 'accuracy' and 'accuracy' in model_metrics:
                    metric_score = model_metrics['accuracy']
                elif metric.lower() == 'f1' and 'f1' in model_metrics:
                    metric_score = model_metrics['f1']
                elif 'test_score' in model_metrics:
                    metric_score = model_metrics['test_score']
                elif 'r2' in model_metrics:
                    metric_score = model_metrics['r2']
                
                if metric_score is not None and (self.best_score is None or metric_score > self.best_score):
                    self.best_model = model
                    self.best_model_name = name
                    self.best_metric = metric
                    self.best_score = metric_score
            
            self.evaluation_results = results
            
            logger.info(f"Model evaluation complete. Best model: {self.best_model_name} with {metric} = {self.best_score:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            return results
    
    def get_best_model(self):
        return self.best_model_name, self.best_model, self.best_score
    
    def get_ensemble_prediction(self, X, weights=None):
        try:
            if not self.models:
                logger.warning("No models for ensemble prediction")
                return None
                
            if X is None or len(X) == 0:
                logger.warning("Empty prediction data provided")
                return None
            
            all_predictions = []
            valid_models = []
            
            for name, model in self.models.items():
                try:
                    y_pred = model.predict(X)
                    if y_pred is not None and len(y_pred) == len(X):
                        all_predictions.append(y_pred)
                        valid_models.append(name)
                except Exception as e:
                    logger.warning(f"Error getting predictions from model {name}: {e}")
            
            if not all_predictions:
                logger.warning("No valid predictions from any model")
                return None
            
            if weights is None:
                if self.evaluation_results:
                    weights = []
                    metric_key = 'r2' if self.best_metric == 'r2' else 'test_score'
                    
                    for name in valid_models:
                        if name in self.evaluation_results and metric_key in self.evaluation_results[name]:
                            score = max(0.01, self.evaluation_results[name][metric_key])
                            weights.append(score)
                        else:
                            weights.append(1.0)
                    
                    total = sum(weights)
                    if total > 0:
                        weights = [w / total for w in weights]
                    else:
                        weights = [1.0 / len(valid_models)] * len(valid_models)
                else:
                    weights = [1.0 / len(valid_models)] * len(valid_models)
            
            ensemble_pred = np.zeros(len(X))
            for i, pred in enumerate(all_predictions):
                ensemble_pred += pred * weights[i]
            
            logger.info(f"Ensemble prediction complete using {len(valid_models)} models")
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return None
    
    def get_feature_importance_summary(self, limit=10):
        if not self.models:
            logger.warning("No models for feature importance analysis")
            return {}
        
        all_importances = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'get_feature_importances'):
                try:
                    importances = model.get_feature_importances(normalized=True, limit=None)
                    if importances:
                        for feature, importance in importances.items():
                            if feature not in all_importances:
                                all_importances[feature] = []
                            all_importances[feature].append(importance)
                except Exception as e:
                    logger.warning(f"Error getting feature importances from model {name}: {e}")
        
        if not all_importances:
            logger.warning("No feature importances available from any model")
            return {}
        
        average_importances = {}
        for feature, values in all_importances.items():
            average_importances[feature] = float(np.mean(values))
        
        sorted_importances = sorted(average_importances.items(), key=lambda x: x[1], reverse=True)
        
        if limit:
            sorted_importances = sorted_importances[:limit]
        
        result = dict(sorted_importances)
        
        logger.info(f"Feature importance summary generated for {len(result)} features")
        
        return result