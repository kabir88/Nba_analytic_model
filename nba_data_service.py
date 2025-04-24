from nba_api.stats.endpoints import leaguegamefinder, commonplayerinfo, boxscoretraditionalv2
from nba_api.stats.endpoints import playergamelog, teamgamelog, commonteamroster
from nba_api.stats.static import teams, players
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple, Union, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TEAM_ABBREVIATIONS = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets', 
    'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
    'LAC': 'Los Angeles Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
}

SHOOTING_STATS = ['FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA']
SCORING_STATS = ['PTS']
REBOUND_STATS = ['OREB', 'DREB', 'REB']
PLAYMAKING_STATS = ['AST', 'TOV', 'STL', 'BLK']
PERFORMANCE_STATS = SCORING_STATS + SHOOTING_STATS + REBOUND_STATS + PLAYMAKING_STATS

class NBADataService:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        
    def _handle_rate_limit(self):
        time.sleep(1)
        
    def preprocess_player_game_logs(self, game_logs: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting advanced preprocessing of player game logs")
        
        if game_logs.empty:
            logger.warning("Empty game logs provided for preprocessing")
            return game_logs
            
        try:
            df = game_logs.copy()
            
            logger.info("Handling missing values in player game logs")
            df = self._handle_missing_values(df)
            
            logger.info("Converting data types in player game logs")
            df = self._convert_data_types(df)
            
            logger.info("Extracting matchup information")
            df = self._process_matchup_data(df)
            
            logger.info("Standardizing minutes played format")
            df = self._standardize_minutes_played(df)
            
            logger.info("Calculating advanced basketball statistics")
            df = self._calculate_advanced_stats(df)
            
            logger.info("Adding game context features")
            df = self._add_game_context(df)
            
            logger.info("Creating time-based features")
            df = self._create_time_features(df)
            
            logger.info("Scaling numerical features")
            df = self._apply_feature_scaling(df)
            
            logger.info(f"Completed preprocessing: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error during player game logs preprocessing: {e}")
            return game_logs
            
    def preprocess_team_game_logs(self, game_logs: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting advanced preprocessing of team game logs")
        
        if game_logs.empty:
            logger.warning("Empty team game logs provided for preprocessing")
            return game_logs
            
        try:
            df = game_logs.copy()
            
            logger.info("Handling missing values in team game logs")
            df = self._handle_missing_values(df)
            
            logger.info("Converting data types in team game logs")
            df = self._convert_data_types(df)
            
            logger.info("Extracting matchup information")
            df = self._process_matchup_data(df)
            
            logger.info("Calculating advanced team statistics")
            df = self._calculate_team_advanced_stats(df)
            
            logger.info("Adding streak and momentum indicators")
            df = self._add_streaks_and_momentum(df)
            
            logger.info("Calculating strength of schedule metrics")
            df = self._calculate_schedule_strength(df)
            
            logger.info("Scaling numerical features")
            df = self._apply_feature_scaling(df)
            
            logger.info(f"Completed team data preprocessing: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error during team game logs preprocessing: {e}")
            return game_logs
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        missing_before = result.isna().sum()
        logger.info(f"Missing values before processing: {missing_before[missing_before > 0]}")
        
        pct_columns = [col for col in result.columns if '_PCT' in col]
        for col in pct_columns:
            if col in result:
                result[col] = result[col].fillna(result[col].mean() if not result[col].empty else 0.0)
        
        for stat in PERFORMANCE_STATS:
            if stat in result:
                result[stat] = result[stat].fillna(0)
        
        categorical_cols = result.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in result and result[col].isna().any():
                if any(team_substr in col.upper() for team_substr in ['TEAM', 'MATCHUP', 'OPPONENT']):
                    result[col] = result[col].fillna('UNK')
                else:
                    most_frequent = result[col].mode()[0] if not result[col].empty else 'Unknown'
                    result[col] = result[col].fillna(most_frequent)
        
        date_cols = [col for col in result.columns if 'DATE' in col.upper()]
        for col in date_cols:
            if col in result and result[col].isna().any():
                if result[col].notna().any():
                    temp = result.sort_index()
                    temp[col] = temp[col].fillna(method='ffill')
                    temp[col] = temp[col].fillna(method='bfill')
                    result[col] = temp[col]
                else:
                    result[col] = result[col].fillna(pd.Timestamp.now().date())
        
        missing_after = result.isna().sum()
        logger.info(f"Missing values after processing: {missing_after[missing_after > 0]}")
        
        if missing_after.sum() > 0:
            logger.warning(f"Some missing values could not be handled: {missing_after[missing_after > 0]}")
        
        return result
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        try:
            date_cols = [col for col in result.columns if 'DATE' in col.upper()]
            for col in date_cols:
                if col in result:
                    try:
                        result[col] = pd.to_datetime(result[col])
                    except:
                        logger.warning(f"Failed to convert {col} to datetime")
            
            numeric_candidates = []
            for col in result.columns:
                if result[col].dtype.kind in 'fcib':
                    continue
                
                if col in date_cols:
                    continue
                
                if any(stat in col for stat in PERFORMANCE_STATS):
                    numeric_candidates.append(col)
                    
                if '_PCT' in col or 'PERCENT' in col:
                    numeric_candidates.append(col)
            
            for col in numeric_candidates:
                if col in result:
                    try:
                        result[col] = pd.to_numeric(result[col], errors='coerce')
                    except:
                        logger.warning(f"Failed to convert {col} to numeric")
            
            bool_candidates = [col for col in result.columns if any(substr in col.upper() for substr in ['IS_', 'HAS_', 'FLAG', 'ACTIVE'])]
            for col in bool_candidates:
                if col in result:
                    try:
                        result[col] = result[col].astype(bool)
                    except:
                        pass
                        
            id_cols = [col for col in result.columns if col.endswith('_ID')]
            for col in id_cols:
                if col in result:
                    try:
                        result[col] = result[col].astype('Int64')
                    except:
                        pass
                        
            return result
            
        except Exception as e:
            logger.error(f"Error in data type conversion: {e}")
            return df
    
    def _process_matchup_data(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        try:
            if 'MATCHUP' in result.columns:
                result['HOME_GAME'] = result['MATCHUP'].apply(lambda x: 1 if 'vs.' in str(x) else 0)
                result['AWAY_GAME'] = result['MATCHUP'].apply(lambda x: 1 if '@' in str(x) else 0)
                
                def extract_opponent(matchup_str):
                    if pd.isna(matchup_str):
                        return None
                    
                    matchup_str = str(matchup_str)
                    if 'vs.' in matchup_str:
                        return matchup_str.split('vs. ')[-1]
                    elif '@' in matchup_str:
                        return matchup_str.split('@ ')[-1]
                    else:
                        return matchup_str.split(' ')[-1]
                
                result['OPPONENT_TEAM'] = result['MATCHUP'].apply(extract_opponent)
                
            return result
            
        except Exception as e:
            logger.error(f"Error in matchup data processing: {e}")
            return df
    
    def _standardize_minutes_played(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        try:
            if 'MIN' in result.columns:
                def convert_minutes(min_str):
                    if pd.isna(min_str):
                        return None
                    
                    if isinstance(min_str, (int, float)):
                        return float(min_str)
                    
                    min_str = str(min_str).strip()
                    if not min_str:
                        return None
                    
                    if ':' in min_str:
                        try:
                            minutes, seconds = min_str.split(':')
                            return float(minutes) + float(seconds) / 60
                        except:
                            pass
                    
                    try:
                        return float(min_str)
                    except:
                        match = re.search(r'(\d+)', min_str)
                        if match:
                            return float(match.group(1))
                        return None
                
                result['MIN_FLOAT'] = result['MIN'].apply(convert_minutes)
                
                if 'MIN_FLOAT' in result.columns and not result['MIN_FLOAT'].isna().all():
                    result['MIN'] = result['MIN_FLOAT']
                    result = result.drop('MIN_FLOAT', axis=1)
                
            return result
            
        except Exception as e:
            logger.error(f"Error in minutes standardization: {e}")
            return df
    
    def _calculate_advanced_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        try:

            if all(col in result.columns for col in ['PTS', 'FGA', 'FTA']):
                result['TS_PCT'] = result.apply(
                    lambda row: row['PTS'] / (2 * (row['FGA'] + 0.44 * row['FTA'])) 
                    if row['FGA'] + 0.44 * row['FTA'] > 0 else None,
                    axis=1
                )
            

            if all(col in result.columns for col in ['FGM', 'FG3M', 'FGA']):
                result['EFG_PCT'] = result.apply(
                    lambda row: (row['FGM'] + 0.5 * row['FG3M']) / row['FGA'] 
                    if row['FGA'] > 0 else None,
                    axis=1
                )
            

            if all(col in result.columns for col in ['FGA', 'FTA', 'TOV']):
                result['USG_PCT'] = result.apply(
                    lambda row: (row['FGA'] + 0.44 * row['FTA'] + row['TOV']) 
                    if not pd.isna(row['FGA']) and not pd.isna(row['FTA']) and not pd.isna(row['TOV']) else None,
                    axis=1
                )
            

            if all(col in result.columns for col in ['AST', 'TOV']):
                result['AST_TO_RATIO'] = result.apply(
                    lambda row: row['AST'] / row['TOV'] if row['TOV'] > 0 else row['AST'] if row['AST'] > 0 else None,
                    axis=1
                )
            

            if all(col in result.columns for col in ['PTS', 'FGA', 'FTA', 'OREB', 'TOV']):
                def simple_off_rating(row):
                    scoring_poss = row['FGM'] + (1 - (1 - row['FT_PCT'])**2) * row['FTA'] * 0.5 if 'FGM' in row and 'FT_PCT' in row else 0
                    total_poss = row['FGA'] + 0.44 * row['FTA'] + row['TOV'] - row['OREB']
                    return 100 * scoring_poss / total_poss if total_poss > 0 else None
                
                if all(required_col in result.columns for required_col in ['FGM', 'FT_PCT']):
                    result['OFF_RATING'] = result.apply(simple_off_rating, axis=1)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating advanced stats: {e}")
            return df
    
    def _add_game_context(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        try:
            if 'GAME_DATE' in result.columns:

                result = result.sort_values('GAME_DATE')
                

                result['DAYS_REST'] = (result['GAME_DATE'] - result['GAME_DATE'].shift(1)).dt.days
                result['DAYS_REST'] = result['DAYS_REST'].fillna(3)
                

                result['BACK_TO_BACK'] = (result['DAYS_REST'] <= 1).astype(int)
                

                result['GAME_NUMBER'] = range(1, len(result) + 1)
                

                result['MONTH'] = result['GAME_DATE'].dt.month

                result['EARLY_SEASON'] = ((result['MONTH'] >= 10) | (result['MONTH'] <= 11)).astype(int)
                result['MID_SEASON'] = ((result['MONTH'] >= 12) | (result['MONTH'] <= 2)).astype(int)
                result['LATE_SEASON'] = ((result['MONTH'] >= 3) & (result['MONTH'] <= 4)).astype(int)
                result['PLAYOFFS'] = (result['MONTH'] >= 5).astype(int)
            

            if 'WL' in result.columns:
                result['WIN'] = (result['WL'] == 'W').astype(int)
                

                result['STREAK_TYPE'] = result['WL'].ne(result['WL'].shift()).cumsum()
                win_streaks = result[result['WL'] == 'W'].groupby('STREAK_TYPE').cumcount() + 1
                loss_streaks = result[result['WL'] == 'L'].groupby('STREAK_TYPE').cumcount() + 1
                result['WIN_STREAK'] = win_streaks
                result['LOSS_STREAK'] = loss_streaks
                

                result['WIN_STREAK'] = result['WIN_STREAK'].fillna(0)
                result['LOSS_STREAK'] = result['LOSS_STREAK'].fillna(0)
                

                for window in [5, 10, 20]:
                    if len(result) >= window:
                        result[f'WIN_PCT_{window}'] = result['WIN'].rolling(window=window).mean()
            
            return result
            
        except Exception as e:
            logger.error(f"Error adding game context: {e}")
            return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        try:

            if result.shape[0] > 0:
                for stat in PERFORMANCE_STATS:
                    if stat in result.columns:

                        for window in [3, 5, 10]:
                            if len(result) >= window:
                                result[f'{stat}_AVG_{window}'] = result[stat].rolling(window=window).mean()
                                result[f'{stat}_STD_{window}'] = result[stat].rolling(window=window).std()
                        

                        result[f'{stat}_EMA'] = result[stat].ewm(span=5).mean()
                        

                        result[f'{stat}_CUM_AVG'] = result[stat].expanding().mean()
            

            if 'PLUS_MINUS' in result.columns:
                for window in [3, 5, 10]:
                    if len(result) >= window:
                        result[f'PM_AVG_{window}'] = result['PLUS_MINUS'].rolling(window=window).mean()
            

            advanced_stats = ['TS_PCT', 'EFG_PCT', 'USG_PCT', 'AST_TO_RATIO', 'OFF_RATING']
            for stat in advanced_stats:
                if stat in result.columns:

                    if len(result) >= 2:
                        result[f'{stat}_DIFF'] = result[stat] - result[stat].shift(1)
                    

                    if len(result) >= 5:
                        result[f'{stat}_TREND'] = result[stat].diff().rolling(window=5).mean()
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating time features: {e}")
            return df
    
    def _apply_feature_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        try:

            exclude_patterns = ['_ID', 'DATE', 'SEASON', 'GAME_ID', 'PLAYER_ID', 'TEAM_ID']
            numeric_cols = result.select_dtypes(include=['int64', 'float64']).columns
            

            numeric_cols = [col for col in numeric_cols if not any(pattern in col for pattern in exclude_patterns)]
            

            if numeric_cols:
                for col in numeric_cols:
                    col_zscore = f'{col}_ZSCORE'
                    mean = result[col].mean()
                    std = result[col].std()
                    if std > 0:
                        result[col_zscore] = (result[col] - mean) / std
            
            return result
            
        except Exception as e:
            logger.error(f"Error in feature scaling: {e}")
            return df
    
    def _calculate_team_advanced_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        try:

            if all(col in result.columns for col in ['FGM', 'FG3M', 'FGA']):
                result['EFG_PCT'] = (result['FGM'] + 0.5 * result['FG3M']) / result['FGA'].where(result['FGA'] > 0, 1)
            
            if all(col in result.columns for col in ['TOV', 'FGA', 'FTA']):
                result['TOV_PCT'] = result['TOV'] / (result['FGA'] + 0.44 * result['FTA'] + result['TOV']).where(
                    (result['FGA'] + 0.44 * result['FTA'] + result['TOV']) > 0, 1
                )
            
            if all(col in result.columns for col in ['OREB', 'DREB', 'REB']):

                result['OREB_PCT'] = result['OREB'] / (result['OREB'] + result['DREB']).where(
                    (result['OREB'] + result['DREB']) > 0, 1
                )
            
            if all(col in result.columns for col in ['FTA', 'FGA']):
                result['FTR'] = result['FTA'] / result['FGA'].where(result['FGA'] > 0, 1)
            

            if all(col in result.columns for col in ['FGA', 'FTA', 'TOV', 'OREB']):

                result['POSS'] = result['FGA'] + 0.44 * result['FTA'] - result['OREB'] + result['TOV']
                

                if 'MIN' in result.columns:
                    minutes_per_game = result['MIN'].max() if not result['MIN'].empty else 48
                    result['PACE'] = 48 * (result['POSS'] / minutes_per_game)
                

            if 'PTS' in result.columns and 'POSS' in result.columns:
                result['OFF_RTG'] = 100 * result['PTS'] / result['POSS'].where(result['POSS'] > 0, 1)
                

                if 'PLUS_MINUS' in result.columns:
                    result['OPP_PTS'] = result['PTS'] - result['PLUS_MINUS']
                    result['DEF_RTG'] = 100 * result['OPP_PTS'] / result['POSS'].where(result['POSS'] > 0, 1)
                    result['NET_RTG'] = result['OFF_RTG'] - result['DEF_RTG']
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating team advanced stats: {e}")
            return df
    
    def _add_streaks_and_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        try:
            if 'WL' in result.columns:

                if 'GAME_DATE' in result.columns:
                    result = result.sort_values('GAME_DATE')
                

                result['WIN'] = (result['WL'] == 'W').astype(int)
                

                result['PREV_GAME_WIN'] = result['WIN'].shift(1).fillna(0).astype(int)
                

                for i in range(1, 6):
                    if len(result) > i:
                        result[f'WON_{i}_GAMES_AGO'] = result['WIN'].shift(i).fillna(0).astype(int)
                


                result['CUM_WIN_STREAK'] = 0
                result['CUM_LOSS_STREAK'] = 0
                
                for idx in range(1, len(result)):
                    if result.iloc[idx-1]['WIN'] == 1:

                        result.loc[result.index[idx], 'CUM_WIN_STREAK'] = result.iloc[idx-1]['CUM_WIN_STREAK'] + 1
                        result.loc[result.index[idx], 'CUM_LOSS_STREAK'] = 0
                    else:

                        result.loc[result.index[idx], 'CUM_LOSS_STREAK'] = result.iloc[idx-1]['CUM_LOSS_STREAK'] + 1
                        result.loc[result.index[idx], 'CUM_WIN_STREAK'] = 0
                

                for window in [5, 10, 20]:
                    if len(result) >= window:

                        result[f'PREV_WIN_PCT_{window}'] = result['WIN'].shift(1).rolling(window=window).mean().fillna(0.5)
            

            if 'PLUS_MINUS' in result.columns:

                for window in [3, 5, 10]:
                    if len(result) >= window:
                        result[f'PM_AVG_{window}'] = result['PLUS_MINUS'].shift(1).rolling(window=window).mean().fillna(0)
                

                if len(result) >= 3:
                    result['PM_TREND'] = result['PLUS_MINUS'].shift(1).rolling(window=3).mean().diff().fillna(0)
                    result['POSITIVE_MOMENTUM'] = (result['PM_TREND'] > 5).astype(int)
                    result['NEGATIVE_MOMENTUM'] = (result['PM_TREND'] < -5).astype(int)
            
            return result
            
        except Exception as e:
            logger.error(f"Error adding streaks and momentum: {e}")
            return df
    
    def _calculate_schedule_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        try:

            if 'OPPONENT_TEAM' in result.columns and 'GAME_DATE' in result.columns:

                result = result.sort_values('GAME_DATE')
                


                team_win_pcts = {}
                

                opp_teams = result['OPPONENT_TEAM'].unique()
                

                for team in opp_teams:
                    team_win_pcts[team] = 0.5
                

                result['OPP_STRENGTH'] = result['OPPONENT_TEAM'].map(team_win_pcts)
                

                for window in [5, 10]:
                    if len(result) >= window:
                        result[f'SOS_{window}'] = result['OPP_STRENGTH'].rolling(window=window).mean().fillna(0.5)
                

                if 'DAYS_REST' in result.columns:

                    result['REST_ADVANTAGE'] = result['DAYS_REST'] - 2
                    

                    result['BACK_TO_BACK'] = (result['DAYS_REST'] <= 1).astype(int)
                    

                    result['EXTRA_REST'] = (result['DAYS_REST'] >= 3).astype(int)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating schedule strength: {e}")
            return df
    
    def get_teams(self, refresh=False):
        collection_name = "nba_teams"
        
        if not refresh:
            stored_teams = self.db_manager.retrieve_data(collection_name)
            if not stored_teams.empty:
                logger.info(f"Retrieved {len(stored_teams)} teams from database")
                return stored_teams
        
        try:
            self._handle_rate_limit()
            nba_teams = teams.get_teams()
            teams_df = pd.DataFrame(nba_teams)
            
            if not teams_df.empty:
                self.db_manager.store_data(collection_name, teams_df)
                logger.info(f"Stored {len(teams_df)} teams in database")
            
            return teams_df
        except Exception as e:
            logger.error(f"Error fetching teams: {e}")
            return pd.DataFrame()
            
    def get_players(self, team_id=None, refresh=False):
        collection_name = "nba_players"
        query = {}
        
        if team_id:
            collection_name = f"team_{team_id}_players"
            
        if not refresh:
            stored_players = self.db_manager.retrieve_data(collection_name, query)
            if not stored_players.empty:
                logger.info(f"Retrieved {len(stored_players)} players from database")
                return stored_players
        
        try:
            self._handle_rate_limit()
            
            if team_id:
                roster = commonteamroster.CommonTeamRoster(team_id=team_id)
                players_data = roster.get_data_frames()[0]
            else:
                nba_players = players.get_players()
                players_data = pd.DataFrame(nba_players)
            
            if not players_data.empty:
                self.db_manager.store_data(collection_name, players_data)
                logger.info(f"Stored {len(players_data)} players in database")
            
            return players_data
        except Exception as e:
            logger.error(f"Error fetching players: {e}")
            return pd.DataFrame()
    
    def get_player_info(self, player_id, refresh=False):
        collection_name = f"player_{player_id}_info"
        
        if not refresh:
            stored_info = self.db_manager.retrieve_data(collection_name)
            if not stored_info.empty:
                logger.info(f"Retrieved player info for {player_id} from database")
                return stored_info
        
        try:
            self._handle_rate_limit()
            player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
            info_df = player_info.get_data_frames()[0]
            
            if not info_df.empty:
                self.db_manager.store_data(collection_name, info_df)
                logger.info(f"Stored player info for {player_id} in database")
            
            return info_df
        except Exception as e:
            logger.error(f"Error fetching player info: {e}")
            return pd.DataFrame()
    
    def get_player_game_logs(self, player_id, season=None, last_n_games=0, refresh=False):
        if not season:
            season = self.get_current_season()
        
        collection_name = f"player_{player_id}_games_{season}"
        
        if not refresh:
            stored_games = self.db_manager.retrieve_data(collection_name)
            if not stored_games.empty:
                logger.info(f"Retrieved {len(stored_games)} game logs for player {player_id} from database")
                
                if last_n_games > 0 and len(stored_games) >= last_n_games:
                    return stored_games.sort_values('GAME_DATE', ascending=False).head(last_n_games)
                return stored_games
        
        try:
            self._handle_rate_limit()
            game_logs = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star='Regular Season'
            )
            games_df = game_logs.get_data_frames()[0]
            
            if not games_df.empty:

                processed_df = self.preprocess_player_game_logs(games_df)
                
                self.db_manager.store_data(collection_name, processed_df)
                logger.info(f"Stored {len(processed_df)} game logs for player {player_id} in database")
                
                if last_n_games > 0 and len(processed_df) >= last_n_games:
                    return processed_df.sort_values('GAME_DATE', ascending=False).head(last_n_games)
                return processed_df
            
            return games_df
        except Exception as e:
            logger.error(f"Error fetching player game logs: {e}")
            return pd.DataFrame()
    
    def get_team_game_logs(self, team_id, season=None, last_n_games=0, refresh=False):
        if not season:
            season = self.get_current_season()
        
        collection_name = f"team_{team_id}_games_{season}"
        
        if not refresh:
            stored_games = self.db_manager.retrieve_data(collection_name)
            if not stored_games.empty:
                logger.info(f"Retrieved {len(stored_games)} game logs for team {team_id} from database")
                
                if last_n_games > 0 and len(stored_games) >= last_n_games:
                    return stored_games.sort_values('GAME_DATE', ascending=False).head(last_n_games)
                return stored_games
        
        try:
            self._handle_rate_limit()
            game_logs = teamgamelog.TeamGameLog(
                team_id=team_id,
                season=season,
                season_type_all_star='Regular Season'
            )
            games_df = game_logs.get_data_frames()[0]
            
            if not games_df.empty:

                processed_df = self.preprocess_team_game_logs(games_df)
                
                self.db_manager.store_data(collection_name, processed_df)
                logger.info(f"Stored {len(processed_df)} game logs for team {team_id} in database")
                
                if last_n_games > 0 and len(processed_df) >= last_n_games:
                    return processed_df.sort_values('GAME_DATE', ascending=False).head(last_n_games)
                return processed_df
            
            return games_df
        except Exception as e:
            logger.error(f"Error fetching team game logs: {e}")
            return pd.DataFrame()
    
    def get_recent_games(self, days=7, refresh=False):
        collection_name = f"recent_games_{days}days"
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        if not refresh:
            stored_games = self.db_manager.retrieve_data(collection_name)
            if not stored_games.empty and 'GAME_DATE' in stored_games.columns:

                most_recent = pd.to_datetime(stored_games['GAME_DATE']).max()
                if most_recent >= start_date:
                    logger.info(f"Retrieved {len(stored_games)} recent games from database")
                    return stored_games
        
        try:
            self._handle_rate_limit()
            

            gamefinder = leaguegamefinder.LeagueGameFinder(
                date_from_nullable=start_date.strftime('%m/%d/%Y'),
                date_to_nullable=end_date.strftime('%m/%d/%Y'),
                league_id_nullable='00'
            )
            games_df = gamefinder.get_data_frames()[0]
            
            if not games_df.empty:

                games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
                

                processed_df = self.preprocess_team_game_logs(games_df)
                
                self.db_manager.store_data(collection_name, processed_df)
                logger.info(f"Stored {len(processed_df)} recent games in database")
                
                return processed_df
            
            return games_df
        except Exception as e:
            logger.error(f"Error fetching recent games: {e}")
            return pd.DataFrame()
    
    def get_season_stats(self, player_id, seasons=None, refresh=False):
        if not seasons:
            current_season = self.get_current_season()
            seasons = [current_season]
        elif isinstance(seasons, str):
            seasons = [seasons]
        
        collection_name = f"player_{player_id}_seasons_stats"
        
        if not refresh:
            stored_stats = self.db_manager.retrieve_data(collection_name)
            if not stored_stats.empty:
                logger.info(f"Retrieved season stats for player {player_id} from database")
                return stored_stats
        
        try:
            all_seasons_data = []
            
            for season in seasons:
                self._handle_rate_limit()
                

                game_logs = self.get_player_game_logs(
                    player_id=player_id,
                    season=season,
                    refresh=refresh
                )
                
                if not game_logs.empty:

                    season_stats = {}
                    season_stats['SEASON'] = season
                    season_stats['PLAYER_ID'] = player_id
                    season_stats['GAMES_PLAYED'] = len(game_logs)
                    

                    for stat in ['MIN', 'PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV', 
                               'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA']:
                        if stat in game_logs.columns:
                            season_stats[stat] = game_logs[stat].mean()
                    

                    if 'FGM' in game_logs.columns and 'FGA' in game_logs.columns:
                        fga_total = game_logs['FGA'].sum()
                        if fga_total > 0:
                            season_stats['FG_PCT'] = game_logs['FGM'].sum() / fga_total
                    
                    if 'FG3M' in game_logs.columns and 'FG3A' in game_logs.columns:
                        fg3a_total = game_logs['FG3A'].sum()
                        if fg3a_total > 0:
                            season_stats['FG3_PCT'] = game_logs['FG3M'].sum() / fg3a_total
                    
                    if 'FTM' in game_logs.columns and 'FTA' in game_logs.columns:
                        fta_total = game_logs['FTA'].sum()
                        if fta_total > 0:
                            season_stats['FT_PCT'] = game_logs['FTM'].sum() / fta_total
                    

                    for adv_stat in ['TS_PCT', 'EFG_PCT', 'USG_PCT', 'AST_TO_RATIO', 'OFF_RATING']:
                        if adv_stat in game_logs.columns:
                            season_stats[adv_stat] = game_logs[adv_stat].mean()
                    
                    all_seasons_data.append(season_stats)
            
            if all_seasons_data:
                seasons_df = pd.DataFrame(all_seasons_data)
                
                self.db_manager.store_data(collection_name, seasons_df)
                logger.info(f"Stored season stats for player {player_id} in database")
                
                return seasons_df
            
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching season stats: {e}")
            return pd.DataFrame()
    
    def get_current_season(self):
        now = datetime.now()
        if now.month >= 10:
            return f"{now.year}-{str(now.year + 1)[-2:]}"
        else:
            return f"{now.year-1}-{str(now.year)[-2:]}"
    
    def get_boxscore(self, game_id, refresh=False):
        collection_name = f"boxscore_{game_id}"
        
        if not refresh:
            stored_boxscore = self.db_manager.retrieve_data(collection_name)
            if not stored_boxscore.empty:
                logger.info(f"Retrieved boxscore for game {game_id} from database")
                return stored_boxscore
        
        try:
            self._handle_rate_limit()
            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            boxscore_df = boxscore.get_data_frames()[0]
            
            if not boxscore_df.empty:
                self.db_manager.store_data(collection_name, boxscore_df)
                logger.info(f"Stored boxscore for game {game_id} in database")
            
            return boxscore_df
        except Exception as e:
            logger.error(f"Error fetching boxscore: {e}")
            return pd.DataFrame()