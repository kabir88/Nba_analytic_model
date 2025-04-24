# NBA Analytics & Predictive Modeling System

This application fetches data from the NBA API, stores it in MongoDB, and implements predictive models using both Random Forest and Reinforcement Learning algorithms to predict player performance and team wins.

## Features

- **Data Collection**: Fetch and store NBA data in MongoDB to avoid repeated API calls
- **Player Performance Prediction**: Train models to predict player statistics for upcoming games
- **Team Win Prediction**: Train models to predict team win probability
- **Model Evaluation**: Compare performance of different models and track accuracy metrics

## Technology Stack

- **Python**: Core programming language
- **MongoDB**: Database for storing NBA data
- **NBA API**: Source of NBA data
- **Pandas/NumPy**: Data manipulation and analysis
- **Scikit-learn**: Implementation of Random Forest models
- **TensorFlow**: Implementation of Reinforcement Learning models
- **Streamlit**: Web interface

## Models Implemented

### Random Forest

The system implements Random Forest for both regression (player stats prediction) and classification (team win prediction) tasks. Key features of our Random Forest implementation:

- Feature importance analysis
- Model evaluation metrics (MAE, RMSE, R2 for regression; Accuracy, Precision, Recall, F1 for classification)
- Hyperparameter optimization
- Model persistence (saving/loading)

### Reinforcement Learning

The system also implements neural network-based Reinforcement Learning models using TensorFlow/Keras for the same prediction tasks. Key features:

- Deep Q-Network (DQN) approach
- Early stopping to prevent overfitting
- Customizable architecture
- Model persistence (saving/loading)

## Model Evaluation

The system includes a comprehensive model evaluation framework that:

- Evaluates models on common test data
- Compares different algorithms on the same task
- Tracks and visualizes performance metrics
- Identifies the best model for each task

## Usage

1. Start by fetching team and player data in the "Data Collection" page
2. Navigate to "Player Performance Prediction" to train and evaluate models for player stats
3. Use "Team Win Prediction" to train and evaluate models for predicting team victories
4. The "Model Evaluation" page allows you to load saved models and review their performance

## Future Enhancements

- Add more advanced models (e.g., XGBoost, Deep Learning)
- Implement more sophisticated feature engineering
- Add visualization components for model predictions
- Extend to additional sports leagues
