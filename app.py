import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib

from db_manager import DatabaseManager
from nba_data_service import NBADataService
from data_processor import DataProcessor
from models import RandomForestModel, ReinforcementLearningModel, GradientBoostingModel, ModelEvaluator
from model_utils import safe_predict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.db_manager = None
    st.session_state.data_service = None
    st.session_state.data_processor = None
    st.session_state.rf_model = None
    st.session_state.rl_model = None
    st.session_state.gb_model = None
    st.session_state.model_evaluator = None

os.makedirs("models", exist_ok=True)

def initialize_components():
    if not st.session_state.initialized:
        try:
            if 'mongo_uri' not in st.session_state:
                st.session_state.mongo_uri = "mongodb+srv://ks1751:Olaoluwa88@cluster0.kyiza.mongodb.net/test"
                
            st.session_state.db_manager = DatabaseManager(connection_string=st.session_state.mongo_uri)
            st.session_state.data_service = NBADataService(st.session_state.db_manager)
            st.session_state.data_processor = DataProcessor()
            st.session_state.model_evaluator = ModelEvaluator()
            st.session_state.initialized = True
            logger.info("Components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            st.error(f"Error initializing components: {e}")

st.title("NBA Analytics & Predictive Modeling System")
st.write("Fetch NBA data, store it in MongoDB, and build predictive models using Random Forest and Reinforcement Learning.")

with st.expander("Database Connection Settings"):
    mongo_uri = st.text_input("MongoDB Connection String", value=st.session_state.get('mongo_uri', "mongodb+srv://ks1751:Olaoluwa88@cluster0.kyiza.mongodb.net/test"), 
                            help="Enter your MongoDB connection string")
    
    if mongo_uri != st.session_state.get('mongo_uri', "mongodb+srv://ks1751:Olaoluwa88@cluster0.kyiza.mongodb.net/test"):
        st.session_state.mongo_uri = mongo_uri
        st.session_state.initialized = False
        st.info("MongoDB connection updated. System will reinitialize with the new connection.")
    
    if st.button("Initialize System"):
        st.session_state.initialized = False
        initialize_components()
        st.success("System initialized successfully!")

if not st.session_state.initialized:
    initialize_components()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", 
                       ["Data Collection", "Player Performance Prediction", "Team Win Prediction", "Model Evaluation"])

if page == "Data Collection":
    st.header("NBA Data Collection")
    st.write("Fetch and store NBA data in the MongoDB database.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fetch Teams Data")
        if st.button("Fetch Teams"):
            with st.spinner("Fetching teams data..."):
                teams_df = st.session_state.data_service.get_teams(refresh=True)
                st.success(f"Successfully fetched {len(teams_df)} teams!")
                st.dataframe(teams_df)
    
    with col2:
        st.subheader("Fetch Recent Games")
        days = st.slider("Number of days to look back", 1, 30, 7)
        if st.button("Fetch Recent Games"):
            with st.spinner(f"Fetching games from the last {days} days..."):
                games_df = st.session_state.data_service.get_recent_games(days=days, refresh=True)
                st.success(f"Successfully fetched {len(games_df)} games!")
                st.dataframe(games_df)
    
    st.subheader("Fetch Team & Player Data")
    
    teams_df = st.session_state.data_service.get_teams()
    if not teams_df.empty:
        team_options = teams_df.to_dict('records')
        team_dict = {team['full_name']: team['id'] for team in team_options}
        
        selected_team = st.selectbox("Select Team", options=list(team_dict.keys()))
        selected_team_id = team_dict[selected_team]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Fetch Team Game Logs")
            season = st.text_input("Season (e.g., 2023-24)", st.session_state.data_service.get_current_season())
            if st.button("Fetch Team Games"):
                with st.spinner(f"Fetching game logs for {selected_team}..."):
                    games_df = st.session_state.data_service.get_team_game_logs(
                        team_id=selected_team_id, 
                        season=season,
                        refresh=True
                    )
                    st.success(f"Successfully fetched {len(games_df)} games!")
                    st.dataframe(games_df)
        
        with col2:
            st.subheader("Fetch Team Players")
            if st.button("Fetch Players"):
                with st.spinner(f"Fetching players for {selected_team}..."):
                    players_df = st.session_state.data_service.get_players(
                        team_id=selected_team_id,
                        refresh=True
                    )
                    st.success(f"Successfully fetched {len(players_df)} players!")
                    st.dataframe(players_df)
                    
                    if 'players' not in st.session_state:
                        st.session_state.players = {}
                    st.session_state.players[selected_team] = players_df.to_dict('records')
        
        st.subheader("Fetch Player Game Logs")
        
        if 'players' in st.session_state and selected_team in st.session_state.players:
            player_options = st.session_state.players[selected_team]
            player_dict = {player.get('PLAYER', player.get('full_name', '')): 
                          player.get('PLAYER_ID', player.get('id', 0)) 
                          for player in player_options}
            
            selected_player = st.selectbox("Select Player", options=list(player_dict.keys()))
            selected_player_id = player_dict[selected_player]
            
            if st.button("Fetch Player Games"):
                with st.spinner(f"Fetching game logs for {selected_player}..."):
                    player_games_df = st.session_state.data_service.get_player_game_logs(
                        player_id=selected_player_id,
                        season=season,
                        last_n_games=0,
                        refresh=True
                    )
                    st.success(f"Successfully fetched {len(player_games_df)} games!")
                    st.dataframe(player_games_df)
        else:
            st.info("Please fetch team players first")

elif page == "Player Performance Prediction":
    st.header("Player Performance Prediction")
    
    prediction_type = "Season-Level Performance"
    st.info("NBA Dashboard now focuses exclusively on season-level predictions for more comprehensive player analysis.")
    
    teams_df = st.session_state.data_service.get_teams()
    if not teams_df.empty:
        team_options = teams_df.to_dict('records')
        team_dict = {team['full_name']: team['id'] for team in team_options}
        
        selected_team = st.selectbox("Select Team", options=list(team_dict.keys()))
        selected_team_id = team_dict[selected_team]
        
        players_df = st.session_state.data_service.get_players(team_id=selected_team_id)
        
        if not players_df.empty:
            player_options = players_df.to_dict('records')
            player_dict = {player.get('PLAYER', player.get('full_name', '')): 
                          player.get('PLAYER_ID', player.get('id', 0)) 
                          for player in player_options}
            
            selected_player = st.selectbox("Select Player", options=list(player_dict.keys()))
            selected_player_id = player_dict[selected_player]
            
            season = st.text_input("Season", st.session_state.data_service.get_current_season())
            
            col1, col2 = st.columns(2)
            
            with col1:
                stat_category = st.selectbox("Stat to Predict", 
                                           options=["PTS", "AST", "REB", "STL", "BLK", "FG3M", "FGM"])
            
            with col2:
                model_type = st.selectbox("Model Type", 
                                        options=["Random Forest", "Reinforcement Learning", "Gradient Boosting", "All (Compare)"])
            
            if prediction_type == "Season-Level Performance":
                st.info("Season-level prediction uses historical season data to predict performance for the upcoming season")
                
                seasons_to_fetch = st.slider("Number of Historical Seasons", min_value=1, max_value=10, value=5,
                                           help="Number of past seasons to fetch for training the model")
                
                if season:
                    try:
                        current_season_year = int(season.split("-")[0])
                        season_list = [f"{year-1}-{str(year)[-2:]}" for year in range(current_season_year - seasons_to_fetch + 1, current_season_year + 1)]
                        st.write(f"Will fetch data for seasons: {', '.join(season_list)}")
                    except:
                        st.warning(f"Could not parse season format: {season}. Using recent seasons.")
            
            if st.button("Train Models"):
                if prediction_type == "Season-Level Performance":
                    with st.spinner("Fetching player data across multiple seasons..."):
                        all_seasons_data = []
                        
                        if season:
                            try:
                                current_season_year = int(season.split("-")[0])
                                season_list = [f"{year-1}-{str(year)[-2:]}" for year in range(current_season_year - seasons_to_fetch + 1, current_season_year + 1)]
                                
                                for s in season_list:
                                    st.write(f"Fetching data for season {s}...")
                                    season_data = st.session_state.data_service.get_player_game_logs(
                                        player_id=selected_player_id,
                                        season=s,
                                        last_n_games=0,
                                        refresh=False
                                    )
                                    
                                    if not season_data.empty:
                                        all_seasons_data.append(season_data)
                                    else:
                                        st.info(f"No data available for season {s}")
                                
                                if all_seasons_data:
                                    player_games_df = pd.concat(all_seasons_data, ignore_index=True)
                                    st.success(f"Successfully fetched data from {len(all_seasons_data)} seasons, total of {len(player_games_df)} games")
                                else:
                                    st.warning("Could not fetch data from any of the requested seasons")
                                    player_games_df = pd.DataFrame()
                            
                            except Exception as e:
                                st.error(f"Error fetching multiple seasons data: {e}")
                                player_games_df = pd.DataFrame()
                        else:
                            st.warning("Please specify a valid season")
                            player_games_df = pd.DataFrame()
                    
                    if not player_games_df.empty:
                        with st.spinner("Preparing data for season-level prediction..."):
                            try:
                                X, y, feature_names, df_clean = st.session_state.data_processor.prepare_season_prediction_data(
                                    player_games_df, stat_category=stat_category, group_by_season=True
                                )
                                
                                st.subheader("Processed Season Data")
                                display_cols = ['SEASON_YEAR', stat_category]
                                
                                for col in ['GAMES_PLAYED', 'MIN', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'AST', 'REB', 'STL', 'BLK', 'TOV']:
                                    if col in df_clean.columns:
                                        display_cols.append(col)
                                
                                st.dataframe(df_clean[display_cols])
                                
                            except Exception as e:
                                st.error(f"Error preparing season data: {e}")
                                X, y, feature_names = np.array([]), np.array([]), []
                
                if not player_games_df.empty and len(X) > 0 and len(y) > 0:
                    st.success(f"Data prepared with {len(X)} samples and {len(feature_names)} features")
                    
                    if model_type in ["Random Forest", "All (Compare)"]:
                        with st.spinner("Training Random Forest model..."):
                            use_feature_selection = st.checkbox("Use Feature Selection to Improve Accuracy", 
                                                              value=True, 
                                                              help="Select only the most important features to improve model accuracy")
                            
                            top_n_features = 6
                            
                            rf_model = RandomForestModel(model_type='regression')
                            rf_metrics = rf_model.train(X, y, feature_names=feature_names)
                            
                            if use_feature_selection:
                                with st.spinner("Performing feature selection to improve accuracy..."):
                                    importance_df = rf_model.get_feature_importances()
                                    
                                    st.write(f"Using top {top_n_features} most important features for prediction")
                                    selected_indices = np.argsort(rf_model.model.feature_importances_)[::-1][:top_n_features]
                                    selected_feature_names = [feature_names[i] for i in selected_indices]
                                    
                                    X_selected = X[:, selected_indices]
                                    
                                    st.info(f"Training improved model with {len(selected_feature_names)} important features instead of {len(feature_names)} original features")
                                    rf_model = RandomForestModel(model_type='regression')
                                    rf_metrics = rf_model.train(X_selected, y, feature_names=selected_feature_names)
                            
                            st.session_state.rf_model = rf_model
                            
                            st.write("Random Forest Model Performance:")
                            if rf_metrics and len(rf_metrics) > 0:
                                metrics_df = pd.DataFrame({
                                    'Metric': list(rf_metrics.keys()),
                                    'Value': list(rf_metrics.values())
                                })
                                st.dataframe(metrics_df)
                            else:
                                if rf_model.model_type == 'regression':
                                    default_metrics = {
                                        'mae': 0.0,
                                        'rmse': 0.0,
                                        'r2': 0.0
                                    }
                                    metrics_df = pd.DataFrame({
                                        'Metric': list(default_metrics.keys()),
                                        'Value': list(default_metrics.values())
                                    })
                                    st.dataframe(metrics_df)
                            
                            st.success("Random Forest model trained successfully!")
                            
                            if hasattr(rf_model, 'get_feature_importances'):
                                st.subheader("Feature Importance")
                                importance_dict = rf_model.get_feature_importances(normalized=True, limit=20)
                                
                                if importance_dict:
                                    importance_df = pd.DataFrame({
                                        'Feature': list(importance_dict.keys()),
                                        'Importance': list(importance_dict.values())
                                    }).sort_values(by='Importance', ascending=False)
                                    
                                    fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
                                    ax.barh(importance_df['Feature'][:10], importance_df['Importance'][:10])
                                    ax.set_xlabel('Importance')
                                    ax.set_title(f'Top 10 Important Features for {stat_category} Prediction')
                                    plt.tight_layout()
                                    st.pyplot(fig)

                    if model_type in ["Reinforcement Learning", "All (Compare)"]:
                        with st.spinner("Training Reinforcement Learning model..."):
                            n_states = X.shape[1]  
                            n_actions = 20
                            
                            rl_model = ReinforcementLearningModel(state_size=n_states, action_size=n_actions)
                            rl_metrics = rl_model.train(X, y, feature_names=feature_names)
                            
                            st.session_state.rl_model = rl_model
                            
                            st.write("Reinforcement Learning Model Performance:")
                            if rl_metrics and len(rl_metrics) > 0:
                                metrics_df = pd.DataFrame({
                                    'Metric': list(rl_metrics.keys()),
                                    'Value': list(rl_metrics.values())
                                })
                                st.dataframe(metrics_df)
                            else:
                                default_metrics = {
                                    'MAE': 0.0,
                                    'R2': 0.0
                                }
                                metrics_df = pd.DataFrame({
                                    'Metric': list(default_metrics.keys()),
                                    'Value': list(default_metrics.values())
                                })
                                st.dataframe(metrics_df)
                                
                            st.success("Reinforcement Learning model trained successfully!")
                    
                    if model_type in ["Gradient Boosting", "All (Compare)"]:
                        with st.spinner("Training Gradient Boosting model..."):
                            gb_model = GradientBoostingModel(model_type='regression')
                            gb_metrics = gb_model.train(X, y, feature_names=feature_names)
                            
                            st.session_state.gb_model = gb_model
                            
                            st.write("Gradient Boosting Model Performance:")
                            if gb_metrics and len(gb_metrics) > 0:
                                metrics_df = pd.DataFrame({
                                    'Metric': list(gb_metrics.keys()),
                                    'Value': list(gb_metrics.values())
                                })
                                st.dataframe(metrics_df)
                            else:
                                if gb_model.model_type == 'regression':
                                    default_metrics = {
                                        'mae': 0.0,
                                        'rmse': 0.0,
                                        'r2': 0.0
                                    }
                                    metrics_df = pd.DataFrame({
                                        'Metric': list(default_metrics.keys()),
                                        'Value': list(default_metrics.values())
                                    })
                                    st.dataframe(metrics_df)
                                    
                            st.success("Gradient Boosting model trained successfully!")
                            
                            if hasattr(gb_model, 'get_feature_importances'):
                                st.subheader("Gradient Boosting Feature Importance")
                                importance_dict = gb_model.get_feature_importances(normalized=True, limit=20)
                                
                                if importance_dict:
                                    importance_df = pd.DataFrame({
                                        'Feature': list(importance_dict.keys()),
                                        'Importance': list(importance_dict.values())
                                    }).sort_values(by='Importance', ascending=False)
                                    
                                    fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
                                    ax.barh(importance_df['Feature'][:10], importance_df['Importance'][:10])
                                    ax.set_xlabel('Importance')
                                    ax.set_title(f'Top 10 Important Features for {stat_category} Prediction (Gradient Boosting)')
                                    plt.tight_layout()
                                    st.pyplot(fig)
                    
                    if model_type == "All (Compare)":
                        model_evaluator = ModelEvaluator()
                        if hasattr(st.session_state, 'rf_model') and st.session_state.rf_model:
                            model_evaluator.add_model("Random Forest", st.session_state.rf_model)
                        if hasattr(st.session_state, 'rl_model') and st.session_state.rl_model:
                            model_evaluator.add_model("Reinforcement Learning", st.session_state.rl_model)
                        if hasattr(st.session_state, 'gb_model') and st.session_state.gb_model:
                            model_evaluator.add_model("Gradient Boosting", st.session_state.gb_model)
                        
                        evaluation_results = model_evaluator.evaluate_all(X, y)
                        
                        if evaluation_results:
                            st.subheader("Model Comparison")
                            results_list = []
                            
                            for model_name, metrics in evaluation_results.items():
                                for metric_name, value in metrics.items():
                                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                                        results_list.append({
                                            'Model': model_name,
                                            'Metric': metric_name,
                                            'Value': value
                                        })
                            
                            if results_list:
                                results_df = pd.DataFrame(results_list)
                                
                                pivot_df = pd.pivot_table(
                                    results_df,
                                    values='Value',
                                    index='Model',
                                    columns='Metric',
                                    aggfunc='first'
                                )
                                
                                st.dataframe(pivot_df)
                                
                                best_model, model_obj, best_score = model_evaluator.get_best_model()
                                st.success(f"Best model: {best_model} with score: {best_score:.4f}")
                            else:
                                st.warning("No comparable metrics found across models")
                        else:
                            st.warning("Could not compare models. Make sure all models were trained successfully.")
                    
                    if prediction_type == "Season-Level Performance":
                        st.subheader("Make Predictions for Next Season")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            expected_age = st.number_input("Expected Age", min_value=18, max_value=45, value=30)
                        with col2:
                            expected_minutes = st.number_input("Expected Minutes Per Game", min_value=0, max_value=48, value=32)
                        with col3:
                            expected_games = st.number_input("Expected Games", min_value=1, max_value=82, value=75)
                        
                        target_season_year = ""
                        if season:
                            try:
                                current_season_year = int(season.split("-")[0])
                                target_season_year = f"{current_season_year}-{str(current_season_year+1)[-2:]}"
                                st.write(f"Predicting for season: {target_season_year}")
                            except:
                                st.warning(f"Could not parse season format: {season}")
                        
                        if st.button("Predict Next Season"):
                            if not player_games_df.empty and len(X) > 0:
                                with st.spinner(f"Predicting {stat_category} for next season..."):
                                    try:
                                        df_clean.sort_values('SEASON_YEAR', ascending=False, inplace=True)
                                        
                                        player_info = {
                                            'AGE': expected_age,
                                            'MIN': expected_minutes,
                                            'GAMES_PLAYED': expected_games,
                                            'NEXT_SEASON': target_season_year
                                        }
                                        
                                        prediction_input = st.session_state.data_processor.create_season_prediction_input(
                                            df_clean, feature_names, player_info
                                        )
                                        
                                        if prediction_input is not None:
                                            predictions = {}
                                            
                                            if hasattr(st.session_state, 'rf_model') and st.session_state.rf_model and model_type in ["Random Forest", "All (Compare)"]:
                                                rf_model = st.session_state.rf_model
                                                
                                                player_context = st.session_state.data_processor.get_player_context(
                                                    player_id=selected_player_id, 
                                                    stat_category=stat_category
                                                )

                                                safe_prediction = safe_predict(rf_model, prediction_input)
                                                
                                                if safe_prediction is not None:
                                                    enhanced_prediction = st.session_state.data_processor.enhance_prediction(
                                                        safe_prediction, 
                                                        player_context=player_context,
                                                        stat_category=stat_category,
                                                        minutes=expected_minutes,
                                                        confidence_interval=True
                                                    )
                                                    
                                                    predictions["Random Forest"] = enhanced_prediction
                                            
                                            if hasattr(st.session_state, 'rl_model') and st.session_state.rl_model and model_type in ["Reinforcement Learning", "All (Compare)"]:
                                                rl_model = st.session_state.rl_model
                                                safe_prediction = safe_predict(rl_model, prediction_input)
                                                
                                                if safe_prediction is not None:
                                                    enhanced_prediction = st.session_state.data_processor.enhance_prediction(
                                                        safe_prediction, 
                                                        player_context=None, 
                                                        stat_category=stat_category, 
                                                        minutes=expected_minutes
                                                    )
                                                    predictions["Reinforcement Learning"] = enhanced_prediction
                                            
                                            if hasattr(st.session_state, 'gb_model') and st.session_state.gb_model and model_type in ["Gradient Boosting", "All (Compare)"]:
                                                gb_model = st.session_state.gb_model
                                                safe_prediction = safe_predict(gb_model, prediction_input)
                                                
                                                if safe_prediction is not None:
                                                    enhanced_prediction = st.session_state.data_processor.enhance_prediction(
                                                        safe_prediction, 
                                                        player_context=None, 
                                                        stat_category=stat_category, 
                                                        minutes=expected_minutes
                                                    )
                                                    predictions["Gradient Boosting"] = enhanced_prediction
                                            
                                            if model_type == "All (Compare)" and hasattr(st.session_state, 'model_evaluator') and st.session_state.model_evaluator:
                                                model_evaluator = st.session_state.model_evaluator
                                                ensemble_prediction = model_evaluator.get_ensemble_prediction(prediction_input)
                                                
                                                if ensemble_prediction is not None and len(ensemble_prediction) > 0:
                                                    safe_prediction = ensemble_prediction[0]
                                                    enhanced_prediction = st.session_state.data_processor.enhance_prediction(
                                                        safe_prediction, 
                                                        player_context=None, 
                                                        stat_category=stat_category, 
                                                        minutes=expected_minutes
                                                    )
                                                    predictions["Ensemble Model"] = enhanced_prediction
                                            
                                            if predictions:
                                                st.subheader(f"Predicted {stat_category} for {selected_player} in {target_season_year} season")
                                                
                                                results_list = []
                                                
                                                for model_name, prediction in predictions.items():
                                                    if isinstance(prediction, dict):
                                                        base_result = {
                                                            'Model': model_name,
                                                            'Predicted Value': prediction.get('value', 0)
                                                        }
                                                        
                                                        if prediction.get('confidence_interval'):
                                                            low, high = prediction['confidence_interval']
                                                            base_result['Lower Bound'] = low
                                                            base_result['Upper Bound'] = high
                                                        
                                                        if prediction.get('adjustment_factors'):
                                                            for factor, value in prediction['adjustment_factors'].items():
                                                                base_result[f'Factor: {factor}'] = value
                                                                
                                                        results_list.append(base_result)
                                                    else:
                                                        results_list.append({
                                                            'Model': model_name,
                                                            'Predicted Value': prediction
                                                        })
                                                
                                                results_df = pd.DataFrame(results_list)
                                                st.dataframe(results_df)
                                                
                                                fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
                                                
                                                bar_colors = {
                                                    'Random Forest': 'blue',
                                                    'Reinforcement Learning': 'green',
                                                    'Gradient Boosting': 'red',
                                                    'Ensemble Model': 'purple'
                                                }
                                                
                                                for model_name, prediction in predictions.items():
                                                    if isinstance(prediction, dict):
                                                        value = prediction.get('value', 0)
                                                        ax.bar(model_name, value, color=bar_colors.get(model_name, 'gray'))
                                                        
                                                        if prediction.get('confidence_interval'):
                                                            low, high = prediction['confidence_interval']
                                                            ax.errorbar(model_name, value, yerr=[[value-low], [high-value]], 
                                                                        fmt='o', color='black', capsize=5)
                                                    else:
                                                        ax.bar(model_name, prediction, color=bar_colors.get(model_name, 'gray'))
                                                
                                                ax.set_ylabel(stat_category)
                                                ax.set_title(f'Predicted {stat_category} for {selected_player} in {target_season_year}')
                                                plt.tight_layout()
                                                st.pyplot(fig)
                                                
                                                if 'Random Forest' in predictions and isinstance(predictions['Random Forest'], dict):
                                                    rf_prediction = predictions['Random Forest']
                                                    
                                                    if 'strategy_recommendations' in rf_prediction:
                                                        st.subheader("Strategy Recommendations")
                                                        recommendations = rf_prediction['strategy_recommendations']
                                                        
                                                        for category, items in recommendations.items():
                                                            st.write(f"**{category}**")
                                                            for item in items:
                                                                st.write(f"- {item}")
                                                        
                                                        if 'feature_explanations' in rf_prediction:
                                                            st.info("Strategy recommendations are based on feature analysis:")
                                                            for feature, explanation in rf_prediction['feature_explanations'].items():
                                                                st.write(f"- **{feature}**: {explanation}")
                                            else:
                                                st.warning("No predictions generated. Try training the models first.")
                                        else:
                                            st.error("Could not create prediction input. Make sure the models are trained properly.")
                                    except Exception as e:
                                        st.error(f"Error making predictions: {e}")
                else:
                    st.warning("Insufficient data for training models. Please ensure you have data for multiple seasons.")

elif page == "Team Win Prediction":
    st.header("Team Win Prediction")
    
    teams_df = st.session_state.data_service.get_teams()
    if not teams_df.empty:
        team_options = teams_df.to_dict('records')
        team_dict = {team['full_name']: team['id'] for team in team_options}
        
        selected_team = st.selectbox("Select Team", options=list(team_dict.keys()))
        selected_team_id = team_dict[selected_team]
        
        season = st.text_input("Season", st.session_state.data_service.get_current_season())
        
        if st.button("Train Team Win Prediction Model"):
            with st.spinner(f"Fetching game logs for {selected_team}..."):
                team_games_df = st.session_state.data_service.get_team_game_logs(
                    team_id=selected_team_id,
                    season=season,
                    refresh=False
                )
                
                if not team_games_df.empty:
                    st.success(f"Successfully fetched {len(team_games_df)} games!")
                    st.dataframe(team_games_df)
                    
                    X, y, feature_names = st.session_state.data_processor.prepare_team_prediction_data(team_games_df)
                    
                    if len(X) > 0 and len(y) > 0:
                        st.success(f"Data prepared with {len(X)} samples and {len(feature_names)} features")
                        
                        rf_model = RandomForestModel(model_type='classification')
                        rf_metrics = rf_model.train(X, y, feature_names=feature_names)
                        
                        st.subheader("Model Performance")
                        metrics_df = pd.DataFrame({
                            'Metric': list(rf_metrics.keys()),
                            'Value': list(rf_metrics.values())
                        })
                        st.dataframe(metrics_df)
                        
                        st.session_state.team_win_model = rf_model
                        st.success("Team win prediction model trained successfully!")
                        
                        if hasattr(rf_model, 'get_feature_importances'):
                            st.subheader("Feature Importance")
                            importance_dict = rf_model.get_feature_importances(normalized=True, limit=15)
                            
                            if importance_dict:
                                importance_df = pd.DataFrame({
                                    'Feature': list(importance_dict.keys()),
                                    'Importance': list(importance_dict.values())
                                }).sort_values(by='Importance', ascending=False)
                                
                                fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
                                ax.barh(importance_df['Feature'][:10], importance_df['Importance'][:10])
                                ax.set_xlabel('Importance')
                                ax.set_title('Top 10 Important Features for Team Win Prediction')
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                st.subheader("Strategy Recommendations")
                                
                                top_features = importance_df['Feature'].tolist()[:6]
                                
                                offensive_recommendations = [
                                    f"Focus on improving {top_features[0].replace('_', ' ')} as it is the most important factor in winning games",
                                    "Increase three-point attempts in the first half to stretch the defense",
                                    f"Emphasize {top_features[2].replace('_', ' ')} during practice sessions"
                                ]
                                
                                defensive_recommendations = [
                                    "Implement aggressive double-teams on high-usage players",
                                    f"Prioritize {top_features[1].replace('_', ' ')} in defensive schemes",
                                    "Increase defensive pressure on perimeter players"
                                ]
                                
                                lineup_recommendations = [
                                    "Optimize rotation to ensure key players are fresh in the fourth quarter",
                                    f"Select lineups that maximize {top_features[3].replace('_', ' ')}",
                                    "Use data-driven matchup analysis when determining defensive assignments"
                                ]
                                
                                st.write("**Offensive Strategy**")
                                for rec in offensive_recommendations:
                                    st.write(f"- {rec}")
                                    
                                st.write("**Defensive Adjustments**")
                                for rec in defensive_recommendations:
                                    st.write(f"- {rec}")
                                    
                                st.write("**Lineup Optimization**")
                                for rec in lineup_recommendations:
                                    st.write(f"- {rec}")
                    else:
                        st.warning("Insufficient data for training the model. Try a different team or season.")
                else:
                    st.warning(f"No game logs found for {selected_team} in {season} season.")

elif page == "Model Evaluation":
    st.header("Model Evaluation")
    
    if (hasattr(st.session_state, 'rf_model') and st.session_state.rf_model) or \
       (hasattr(st.session_state, 'rl_model') and st.session_state.rl_model) or \
       (hasattr(st.session_state, 'gb_model') and st.session_state.gb_model):
        
        st.success("Models are loaded and ready for evaluation!")
        
        model_list = []
        if hasattr(st.session_state, 'rf_model') and st.session_state.rf_model:
            model_list.append("Random Forest")
        if hasattr(st.session_state, 'rl_model') and st.session_state.rl_model:
            model_list.append("Reinforcement Learning") 
        if hasattr(st.session_state, 'gb_model') and st.session_state.gb_model:
            model_list.append("Gradient Boosting")
            
        selected_model = st.selectbox("Select Model for Detailed Evaluation", options=model_list)
        
        if selected_model == "Random Forest" and hasattr(st.session_state, 'rf_model'):
            model = st.session_state.rf_model
            if hasattr(model, 'metrics'):
                st.subheader("Model Metrics")
                metrics_df = pd.DataFrame({
                    'Metric': list(model.metrics.keys()),
                    'Value': list(model.metrics.values())
                })
                st.dataframe(metrics_df)
            
            if hasattr(model, 'get_feature_importances'):
                st.subheader("Feature Importance")
                importance_dict = model.get_feature_importances(normalized=True)
                
                if importance_dict:
                    importance_df = pd.DataFrame({
                        'Feature': list(importance_dict.keys()),
                        'Importance': list(importance_dict.values())
                    }).sort_values(by='Importance', ascending=False)
                    
                    fig, ax = plt.figure(figsize=(12, 8)), plt.axes()
                    ax.barh(importance_df['Feature'][:15], importance_df['Importance'][:15])
                    ax.set_xlabel('Relative Importance')
                    ax.set_title('Feature Importance (Random Forest)')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.subheader("Feature Insights")
                    
                    for feature, importance in importance_df.head(10).itertuples(index=False):
                        st.write(f"**{feature}** (Importance: {importance:.4f})")
                        
                        if 'AVG' in feature:
                            st.write(f"- This represents the average {feature.split('_')[0]} over recent games")
                            st.write("- Consistent performance in this stat is predictive of future outcomes")
                        
                        if 'PCT' in feature:
                            st.write(f"- This efficiency metric shows how well the player performs in {feature.split('_')[0]}")
                            st.write("- Higher percentages tend to indicate better performance sustainability")
                        
                        if 'DIFF' in feature:
                            st.write("- This differential metric compares performance relative to opponents")
                            st.write("- Positive differentials strongly correlate with player impact")
                        
                        if 'PREV' in feature:
                            st.write("- This feature captures historical performance")
                            st.write("- Strong predictive value for establishing player trends")
        
        elif selected_model == "Reinforcement Learning" and hasattr(st.session_state, 'rl_model'):
            model = st.session_state.rl_model
            if hasattr(model, 'metrics'):
                st.subheader("Model Metrics")
                metrics_df = pd.DataFrame({
                    'Metric': list(model.metrics.keys()),
                    'Value': list(model.metrics.values())
                })
                st.dataframe(metrics_df)
                
            st.subheader("Reinforcement Learning Model Characteristics")
            
            rl_params = {
                "State Dimensions": model.state_size,
                "Action Space": model.action_size,
                "Learning Rate": model.learning_rate,
                "Discount Factor": model.discount_factor,
                "Q-Table Entries": len(model.q_table) if hasattr(model, 'q_table') else 0
            }
            
            params_df = pd.DataFrame({
                'Parameter': list(rl_params.keys()),
                'Value': list(rl_params.values())
            })
            st.dataframe(params_df)
        
        elif selected_model == "Gradient Boosting" and hasattr(st.session_state, 'gb_model'):
            model = st.session_state.gb_model
            if hasattr(model, 'metrics'):
                st.subheader("Model Metrics")
                metrics_df = pd.DataFrame({
                    'Metric': list(model.metrics.keys()),
                    'Value': list(model.metrics.values())
                })
                st.dataframe(metrics_df)
            
            if hasattr(model, 'get_feature_importances'):
                st.subheader("Feature Importance")
                importance_dict = model.get_feature_importances(normalized=True)
                
                if importance_dict:
                    importance_df = pd.DataFrame({
                        'Feature': list(importance_dict.keys()),
                        'Importance': list(importance_dict.values())
                    }).sort_values(by='Importance', ascending=False)
                    
                    fig, ax = plt.figure(figsize=(12, 8)), plt.axes()
                    ax.barh(importance_df['Feature'][:15], importance_df['Importance'][:15])
                    ax.set_xlabel('Relative Importance')
                    ax.set_title('Feature Importance (Gradient Boosting)')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.subheader("Model Parameters")
                    if hasattr(model.model, 'get_params'):
                        params = model.model.get_params()
                        params_df = pd.DataFrame({
                            'Parameter': list(params.keys()),
                            'Value': list(params.values())
                        })
                        st.dataframe(params_df)
        
        st.subheader("Save Models")
        if st.button("Save All Trained Models"):
            models_saved = 0
            
            try:
                os.makedirs("models", exist_ok=True)
                
                if hasattr(st.session_state, 'rf_model') and st.session_state.rf_model:
                    st.session_state.rf_model.save_model("models/random_forest_model.joblib")
                    models_saved += 1
                
                if hasattr(st.session_state, 'rl_model') and st.session_state.rl_model:
                    st.session_state.rl_model.save_model("models/reinforcement_learning_model.joblib")
                    models_saved += 1
                
                if hasattr(st.session_state, 'gb_model') and st.session_state.gb_model:
                    st.session_state.gb_model.save_model("models/gradient_boosting_model.joblib")
                    models_saved += 1
                
                if hasattr(st.session_state, 'team_win_model') and st.session_state.team_win_model:
                    st.session_state.team_win_model.save_model("models/team_win_model.joblib")
                    models_saved += 1
                
                st.success(f"Successfully saved {models_saved} models!")
            except Exception as e:
                st.error(f"Error saving models: {e}")
        
        st.subheader("Load Models")
        if st.button("Load Saved Models"):
            models_loaded = 0
            
            try:
                if os.path.exists("models/random_forest_model.joblib"):
                    rf_model = RandomForestModel()
                    if rf_model.load_model("models/random_forest_model.joblib"):
                        st.session_state.rf_model = rf_model
                        models_loaded += 1
                
                if os.path.exists("models/reinforcement_learning_model.joblib"):
                    rl_model = ReinforcementLearningModel(state_size=1, action_size=1) 
                    if rl_model.load_model("models/reinforcement_learning_model.joblib"):
                        st.session_state.rl_model = rl_model
                        models_loaded += 1
                
                if os.path.exists("models/gradient_boosting_model.joblib"):
                    gb_model = GradientBoostingModel()
                    if gb_model.load_model("models/gradient_boosting_model.joblib"):
                        st.session_state.gb_model = gb_model
                        models_loaded += 1
                
                if os.path.exists("models/team_win_model.joblib"):
                    team_win_model = RandomForestModel(model_type='classification')
                    if team_win_model.load_model("models/team_win_model.joblib"):
                        st.session_state.team_win_model = team_win_model
                        models_loaded += 1
                
                st.success(f"Successfully loaded {models_loaded} models!")
            except Exception as e:
                st.error(f"Error loading models: {e}")
    else:
        st.warning("No models have been trained yet. Please go to the Player Performance Prediction or Team Win Prediction pages to train models first.")