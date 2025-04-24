import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import logging
from datetime import datetime
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataProcessor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False,
                                     handle_unknown="ignore")
        self.poly_features = PolynomialFeatures(degree=2,
                                                include_bias=False,
                                                interaction_only=True)

    def prepare_player_prediction_data(self,
                                       player_game_logs,
                                       stat_category='PTS'):
        try:
            if player_game_logs.empty:
                logger.warning("Empty player game logs data provided")
                return np.array([]), np.array([]), []
            df = player_game_logs.copy()
            if 'GAME_DATE' in df.columns:
                df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            if 'GAME_DATE' in df.columns:
                df = df.sort_values('GAME_DATE')
            logger.info(
                f"Preparing prediction with {len(df)} games data, target stat: {stat_category}"
            )

            if 'MATCHUP' in df.columns:
                df['HOME'] = df['MATCHUP'].apply(lambda x: 1
                                                 if 'vs.' in str(x) else 0)

            new_columns = {}

            key_stats = [
                'MIN', stat_category, 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM',
                'FTA', 'AST', 'REB', 'TOV', 'STL', 'BLK'
            ]
            for stat in key_stats:
                if stat in df.columns:
                    for window in [3, 5, 10]:
                        col_name = f'{stat}_AVG_{window}'
                        new_columns[col_name] = df[stat].rolling(
                            window=window).mean().fillna(df[stat].mean())

                        std_col_name = f'{stat}_STD_{window}'
                        new_columns[std_col_name] = df[stat].rolling(
                            window=window).std().fillna(df[stat].std())

                        ema_col_name = f'{stat}_EMA_{window}'
                        new_columns[ema_col_name] = df[stat].ewm(
                            span=window).mean().fillna(df[stat].mean())
            if all(col in df.columns for col in ['PTS', 'FGA', 'FTA']):
                df['TS_PCT'] = df.apply(
                    lambda row: row['PTS'] / (2 *
                                              (row['FGA'] + 0.44 * row['FTA']))
                    if (row['FGA'] + 0.44 * row['FTA']) > 0 else 0,
                    axis=1)
                for window in [5, 10]:
                    if len(df) >= window:
                        col_name = f'TS_PCT_AVG_{window}'
                        new_columns[col_name] = df['TS_PCT'].rolling(
                            window=window).mean().fillna(df['TS_PCT'].mean())

            if all(col in df.columns for col in ['FGA', 'FTA', 'TOV']):
                df['USAGE_RATE'] = df.apply(
                    lambda row: (row['FGA'] + 0.44 * row['FTA'] + row['TOV'])
                    if all(row[col] >= 0
                           for col in ['FGA', 'FTA', 'TOV']) else 0,
                    axis=1)

                for window in [5, 10]:
                    if len(df) >= window:
                        col_name = f'USAGE_RATE_AVG_{window}'
                        new_columns[col_name] = df['USAGE_RATE'].rolling(
                            window=window).mean().fillna(
                                df['USAGE_RATE'].mean())

            if 'GAME_DATE' in df.columns:
                df['DAYS_REST'] = (df['GAME_DATE'] -
                                   df['GAME_DATE'].shift(1)).dt.days
                df['DAYS_REST'] = df['DAYS_REST'].fillna(3)

                df['DAYS_REST'] = df['DAYS_REST'].astype(int)

                new_columns['NO_REST'] = (df['DAYS_REST'] == 0).astype(int)
                new_columns['SHORT_REST'] = (
                    (df['DAYS_REST'] > 0) & (df['DAYS_REST'] <= 1)).astype(int)
                new_columns['NORMAL_REST'] = (
                    (df['DAYS_REST'] > 1) & (df['DAYS_REST'] <= 3)).astype(int)
                new_columns['EXTENDED_REST'] = (df['DAYS_REST']
                                                > 3).astype(int)

                new_columns['BACK_TO_BACK'] = (df['DAYS_REST']
                                               <= 1).astype(int)

                new_columns['MONTH'] = df['GAME_DATE'].dt.month
                new_columns['EARLY_SEASON'] = df['GAME_DATE'].dt.month.isin(
                    [10, 11, 12]).astype(int)
                new_columns['MID_SEASON'] = df['GAME_DATE'].dt.month.isin(
                    [1, 2, 3]).astype(int)
                new_columns['LATE_SEASON'] = df['GAME_DATE'].dt.month.isin(
                    [4, 5, 6]).astype(int)

            if 'MATCHUP' in df.columns:
                df['OPPONENT'] = df['MATCHUP'].apply(
                    lambda x: str(x).split()[-1].replace('@', '')
                    if '@' in str(x) else str(x).split()[-1])
                team_strength = {
                    'GSW': 5,
                    'LAL': 5,
                    'MIL': 5,
                    'BOS': 5,
                    'PHX': 5,
                    'DAL': 4,
                    'DEN': 4,
                    'PHI': 4,
                    'MIA': 4,
                    'UTA': 4,
                    'BKN': 4,
                    'ATL': 3,
                    'CHI': 3,
                    'TOR': 3,
                    'CLE': 3,
                    'LAC': 3,
                    'MIN': 3,
                    'NOP': 3,
                    'CHA': 3,
                    'NYK': 3,
                    'POR': 3,
                    'SAS': 3,
                    'WAS': 2,
                    'SAC': 2,
                    'IND': 2,
                    'OKC': 2,
                    'ORL': 2,
                    'DET': 1,
                    'HOU': 1,
                    'MEM': 1
                }

                new_columns['OPP_STRENGTH'] = df['OPPONENT'].map(
                    team_strength).fillna(3)

                opp_encoded = self.encoder.fit_transform(df[['OPPONENT']])
                opp_cols = [
                    f'OPP_{team}' for team in self.encoder.categories_[0]
                ]
                opp_df = pd.DataFrame(opp_encoded,
                                      columns=opp_cols,
                                      index=df.index)

                for col in opp_df.columns:
                    new_columns[col] = opp_df[col]
            df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
            feature_cols = []

            for window in [3, 5, 10]:
                avg_col = f'{stat_category}_AVG_{window}'
                if avg_col in df.columns:
                    feature_cols.append(avg_col)

                std_col = f'{stat_category}_STD_{window}'
                if std_col in df.columns:
                    feature_cols.append(std_col)

                ema_col = f'{stat_category}_EMA_{window}'
                if ema_col in df.columns:
                    feature_cols.append(ema_col)

            for metric in ['TS_PCT', 'USAGE_RATE']:
                if metric in df.columns:
                    feature_cols.append(metric)

                for window in [5, 10]:
                    avg_col = f'{metric}_AVG_{window}'
                    if avg_col in df.columns:
                        feature_cols.append(avg_col)

            for col in [
                    'HOME', 'DAYS_REST', 'NO_REST', 'SHORT_REST',
                    'NORMAL_REST', 'EXTENDED_REST', 'BACK_TO_BACK',
                    'EARLY_SEASON', 'MID_SEASON', 'LATE_SEASON'
            ]:
                if col in df.columns:
                    feature_cols.append(col)

            if 'OPP_STRENGTH' in df.columns:
                feature_cols.append('OPP_STRENGTH')

            opp_cols = [col for col in df.columns if col.startswith('OPP_')]
            feature_cols.extend(opp_cols)

            if 'MIN' in df.columns:
                feature_cols.append('MIN')

            other_stats = [
                s for s in key_stats if s != stat_category and s != 'MIN'
            ]
            for stat in other_stats:
                for window in [5, 10]:
                    avg_col = f'{stat}_AVG_{window}'
                    if avg_col in df.columns:
                        feature_cols.append(avg_col)

            df_clean = df.dropna(subset=feature_cols + [stat_category])

            logger.info(
                f"Clean data shape: {df_clean.shape} after removing NaN values"
            )

            if len(df_clean) < 3:
                logger.warning(
                    f"Not enough clean data points for prediction (only {len(df_clean)} available)"
                )
                return np.array([]), np.array([]), feature_cols

            numeric_features = []
            for col in feature_cols:
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col],
                                                  errors='raise')
                    numeric_features.append(col)
                except:
                    logger.warning(f"Dropping non-numeric feature: {col}")

            feature_cols = numeric_features

            X = df_clean[feature_cols].values
            y = df_clean[stat_category].values

            if len(X) > 10 and len(feature_cols) > 10:
                try:
                    selector = SelectKBest(mutual_info_regression,
                                           k=min(20, len(feature_cols)))
                    X_new = selector.fit_transform(X, y)

                    selected_indices = selector.get_support(indices=True)
                    selected_feature_names = [
                        feature_cols[i] for i in selected_indices
                    ]

                    logger.info(
                        f"Selected {len(selected_feature_names)} features using mutual information"
                    )

                    X = X_new
                    feature_cols = selected_feature_names
                except Exception as e:
                    logger.warning(
                        f"Feature selection failed: {e}. Using all features.")

            X = self.scaler.fit_transform(X)

            logger.info(
                f"Prepared enhanced prediction data with {X.shape[0]} samples and {X.shape[1]} features"
            )
            return X, y, feature_cols

        except Exception as e:
            logger.error(
                f"Error preparing enhanced player prediction data: {e}")
            return np.array([]), np.array([]), []

    def _convert_min_to_float(self, min_str):
        if pd.isna(min_str):
            return 0.0

        if isinstance(min_str, (int, float)):
            return float(min_str)

        min_str = str(min_str).strip()

        if not min_str:
            return 0.0

        if ':' in min_str:
            try:
                minutes, seconds = min_str.split(':')
                return float(minutes) + float(seconds) / 60
            except:
                match = re.search(r'(\d+)', min_str)
                if match:
                    return float(match.group(1))
                return 0.0
        else:
            try:
                return float(min_str)
            except:
                return 0.0

    def create_prediction_input(self, opponent_team, home_away, rest_days,
                                prev_games_avg, minutes, back_to_back,
                                feature_names):
        try:
            input_data = {}

            for stat, avg_value in prev_games_avg.items():
                for window in [3, 5, 10]:
                    feature_name = f"{stat}_AVG_{window}"
                    if feature_name in feature_names:
                        input_data[feature_name] = avg_value

                    std_feature = f"{stat}_STD_{window}"
                    if std_feature in feature_names:
                        input_data[std_feature] = avg_value * 0.1

                    ema_feature = f"{stat}_EMA_{window}"
                    if ema_feature in feature_names:
                        input_data[ema_feature] = avg_value

            if 'HOME' in feature_names:
                input_data['HOME'] = 1 if home_away.lower() == 'home' else 0

            if 'DAYS_REST' in feature_names:
                input_data['DAYS_REST'] = rest_days

            rest_features = {
                'NO_REST': rest_days == 0,
                'SHORT_REST': rest_days > 0 and rest_days <= 1,
                'NORMAL_REST': rest_days > 1 and rest_days <= 3,
                'EXTENDED_REST': rest_days > 3,
                'BACK_TO_BACK': rest_days <= 1
            }

            for feature, value in rest_features.items():
                if feature in feature_names:
                    input_data[feature] = 1 if value else 0

            current_month = datetime.now().month
            season_features = {
                'EARLY_SEASON': current_month in [10, 11, 12],
                'MID_SEASON': current_month in [1, 2, 3],
                'LATE_SEASON': current_month in [4, 5, 6],
                'MONTH': current_month
            }

            for feature, value in season_features.items():
                if feature in feature_names:
                    input_data[
                        feature] = current_month if feature == 'MONTH' else (
                            1 if value else 0)

            team_strength = {
                'GSW': 5,
                'LAL': 5,
                'MIL': 5,
                'BOS': 5,
                'PHX': 5,
                'DAL': 4,
                'DEN': 4,
                'PHI': 4,
                'MIA': 4,
                'UTA': 4,
                'BKN': 4,
                'ATL': 3,
                'CHI': 3,
                'TOR': 3,
                'CLE': 3,
                'LAC': 3,
                'MIN': 3,
                'NOP': 3,
                'CHA': 3,
                'NYK': 3,
                'POR': 3,
                'SAS': 3,
                'WAS': 2,
                'SAC': 2,
                'IND': 2,
                'OKC': 2,
                'ORL': 2,
                'DET': 1,
                'HOU': 1,
                'MEM': 1
            }

            if 'OPP_STRENGTH' in feature_names and opponent_team in team_strength:
                input_data['OPP_STRENGTH'] = team_strength.get(
                    opponent_team, 3)

            opp_feature = f"OPP_{opponent_team}"
            for feature in feature_names:
                if feature.startswith('OPP_'):
                    input_data[feature] = 1 if feature == opp_feature else 0

            if 'MIN' in feature_names:
                input_data['MIN'] = minutes

            if 'TS_PCT' in feature_names and 'PTS' in prev_games_avg and 'FGA' in prev_games_avg and 'FTA' in prev_games_avg:
                pts = prev_games_avg['PTS']
                fga = prev_games_avg['FGA']
                fta = prev_games_avg['FTA']

                if fga + 0.44 * fta > 0:
                    ts_pct = pts / (2 * (fga + 0.44 * fta))
                    input_data['TS_PCT'] = ts_pct
                else:
                    input_data['TS_PCT'] = 0

            if 'USAGE_RATE' in feature_names and 'FGA' in prev_games_avg and 'FTA' in prev_games_avg and 'TOV' in prev_games_avg:
                fga = prev_games_avg['FGA']
                fta = prev_games_avg['FTA']
                tov = prev_games_avg['TOV']

                input_data['USAGE_RATE'] = fga + 0.44 * fta + tov

            for window in [5, 10]:
                ts_pct_feature = f'TS_PCT_AVG_{window}'
                if ts_pct_feature in feature_names and 'TS_PCT' in input_data:
                    input_data[ts_pct_feature] = input_data['TS_PCT']

                usage_feature = f'USAGE_RATE_AVG_{window}'
                if usage_feature in feature_names and 'USAGE_RATE' in input_data:
                    input_data[usage_feature] = input_data['USAGE_RATE']

            final_input = []
            for feature in feature_names:
                if feature in input_data:
                    final_input.append(input_data[feature])
                else:
                    logger.warning(
                        f"Missing feature in prediction input: {feature}")
                    final_input.append(0)

            input_vector = np.array(final_input).reshape(1, -1)

            logger.info(
                f"Created prediction input with {len(feature_names)} features")
            return self.scaler.transform(input_vector)

        except Exception as e:
            logger.error(f"Error creating prediction input: {e}")
            return None

    def prepare_season_prediction_data(self,
                                       player_game_logs,
                                       stat_category='PTS',
                                       group_by_season=True):
        try:
            if player_game_logs.empty:
                logger.warning("Empty player game logs data provided")
                return np.array([]), np.array([]), [], pd.DataFrame()

            df = player_game_logs.copy()

            if 'GAME_DATE' in df.columns:
                df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

            if 'SEASON_ID' in df.columns:
                df['SEASON_YEAR'] = df['SEASON_ID'].apply(
                    lambda x: f"{str(x)[:4]}-{str(int(str(x)[:4])+1)[2:]}"
                    if len(str(x)) >= 4 else "Unknown")
            elif 'SEASON' in df.columns:
                df['SEASON_YEAR'] = df['SEASON']
            else:
                if 'GAME_DATE' in df.columns:
                    df['SEASON_YEAR'] = df['GAME_DATE'].apply(
                        lambda x: f"{x.year-1}-{str(x.year)[2:]}"
                        if x.month >= 7 else f"{x.year-2}-{str(x.year-1)[2:]}")
                else:
                    logger.warning("No season information found in data")
                    return np.array([]), np.array([]), [], pd.DataFrame()

            logger.info(
                f"Preparing season prediction data with {len(df)} games data, target: {stat_category}"
            )

            if 'MIN' in df.columns and not pd.api.types.is_numeric_dtype(
                    df['MIN']):
                df['MIN'] = df['MIN'].apply(self._convert_min_to_float)

            key_stats = [
                'MIN', 'PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV', 'FGM', 'FGA',
                'FG3M', 'FG3A', 'FTM', 'FTA'
            ]

            for stat in key_stats:
                if stat in df.columns and not pd.api.types.is_numeric_dtype(
                        df[stat]):
                    try:
                        df[stat] = pd.to_numeric(df[stat], errors='coerce')
                        df[stat] = df[stat].fillna(0)
                    except:
                        logger.warning(f"Could not convert {stat} to numeric")

            if all(col in df.columns for col in ['FGM', 'FGA']):
                df['FG_PCT'] = df.apply(lambda row: row['FGM'] / row['FGA']
                                        if row['FGA'] > 0 else 0,
                                        axis=1)

            if all(col in df.columns for col in ['FG3M', 'FG3A']):
                df['FG3_PCT'] = df.apply(lambda row: row['FG3M'] / row['FG3A']
                                         if row['FG3A'] > 0 else 0,
                                         axis=1)

            if all(col in df.columns for col in ['FTM', 'FTA']):
                df['FT_PCT'] = df.apply(lambda row: row['FTM'] / row['FTA']
                                        if row['FTA'] > 0 else 0,
                                        axis=1)

            if group_by_season:
                logger.info("Grouping data by season")

                season_stats = []

                for season, season_df in df.groupby('SEASON_YEAR'):
                    season_dict = {'SEASON_YEAR': season}

                    season_dict['GAMES_PLAYED'] = len(season_df)

                    for stat in key_stats + ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
                        if stat in season_df.columns:
                            season_dict[stat] = season_df[stat].mean()

                    season_stats.append(season_dict)

                if not season_stats:
                    logger.warning("No season stats generated")
                    return np.array([]), np.array([]), [], pd.DataFrame()

                df_seasons = pd.DataFrame(season_stats)

                df_seasons = df_seasons.sort_values('SEASON_YEAR')

                required_cols = ['SEASON_YEAR', 'GAMES_PLAYED', stat_category]
                if not all(col in df_seasons.columns for col in required_cols):
                    missing = [
                        col for col in required_cols
                        if col not in df_seasons.columns
                    ]
                    logger.warning(f"Missing required columns: {missing}")
                    return np.array([]), np.array([]), [], pd.DataFrame()

                logger.info(f"Created {len(df_seasons)} season records")

                df_features = pd.DataFrame()

                for i in range(1, len(df_seasons)):
                    features = {}

                    current_season = df_seasons.iloc[i]
                    prev_season = df_seasons.iloc[i - 1]

                    features['SEASON_YEAR'] = current_season['SEASON_YEAR']
                    features['PREV_SEASON'] = prev_season['SEASON_YEAR']

                    features['AGE'] = i + 20
                    features['EXPERIENCE'] = i

                    for stat in key_stats + ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
                        if stat in prev_season and not pd.isna(
                                prev_season[stat]):
                            features[f'PREV_SEASON_{stat}'] = prev_season[stat]

                    features['PREV_SEASON_GAMES'] = prev_season['GAMES_PLAYED']

                    if i >= 2:
                        prev_prev_season = df_seasons.iloc[i - 2]
                        for stat in [stat_category]:
                            if stat in prev_season and stat in prev_prev_season:
                                features['SEASON_TREND'] = prev_season[
                                    stat] - prev_prev_season[stat]
                    else:
                        features['SEASON_TREND'] = 0

                    stats_2yr_avg = {}
                    count = 0

                    for j in range(1, min(3, i + 1)):
                        if i - j >= 0:
                            past_season = df_seasons.iloc[i - j]
                            for stat in [stat_category]:
                                if stat in past_season:
                                    if f'PREV_{stat}_2YR_AVG' not in stats_2yr_avg:
                                        stats_2yr_avg[
                                            f'PREV_{stat}_2YR_AVG'] = 0
                                    stats_2yr_avg[
                                        f'PREV_{stat}_2YR_AVG'] += past_season[
                                            stat]
                            count += 1

                    for stat_key, total in stats_2yr_avg.items():
                        if count > 0:
                            features[stat_key] = total / count
                        else:
                            features[stat_key] = 0

                    df_features = pd.concat(
                        [df_features, pd.DataFrame([features])],
                        ignore_index=True)

                if df_features.empty:
                    logger.warning("No features generated from season data")
                    return np.array([]), np.array([]), [], df_seasons

                logger.info(
                    f"Created features dataframe with {len(df_features)} rows")

                feature_cols = [
                    col for col in df_features.columns if col not in
                    ['SEASON_YEAR', 'PREV_SEASON', stat_category]
                ]

                for i in range(len(df_features)):
                    current_season = df_features.iloc[i]['SEASON_YEAR']
                    season_stats = df_seasons[df_seasons['SEASON_YEAR'] ==
                                              current_season]

                    if not season_stats.empty and stat_category in season_stats.columns:
                        df_features.loc[i, stat_category] = season_stats.iloc[
                            0][stat_category]

                df_features = df_features.dropna(subset=[stat_category] +
                                                 feature_cols)

                if len(df_features) < 1:
                    logger.warning("Insufficient data after cleaning")
                    return np.array([]), np.array([]), [], df_seasons

                logger.info(
                    f"Final feature dataframe has {len(df_features)} rows")

                X = df_features[feature_cols].values
                y = df_features[stat_category].values

                return X, y, feature_cols, df_seasons
            else:
                logger.info("Using individual game data for season prediction")

                df = df.sort_values('GAME_DATE')

                df['GAME_NUMBER'] = range(1, len(df) + 1)
                df['CAREER_GAMES'] = df['GAME_NUMBER']

                for stat in key_stats:
                    if stat in df.columns:
                        df[f'CAREER_{stat}_AVG'] = df[stat].expanding().mean()

                feature_cols = []

                for col in df.columns:
                    if col.startswith(
                            'CAREER_'
                    ) and col != f'CAREER_{stat_category}_AVG':
                        feature_cols.append(col)

                feature_cols.extend(['GAME_NUMBER', 'CAREER_GAMES'])

                df_clean = df.dropna(subset=feature_cols + [stat_category])

                if len(df_clean) < 10:
                    logger.warning(
                        f"Insufficient clean data points: {len(df_clean)}")
                    return np.array([]), np.array([]), [], df

                X = df_clean[feature_cols].values
                y = df_clean[stat_category].values

                return X, y, feature_cols, df_clean
        except Exception as e:
            logger.error(f"Error preparing season prediction data: {e}")
            return np.array([]), np.array([]), [], pd.DataFrame()

    def create_season_prediction_input(self,
                                       player_seasons_df,
                                       feature_names,
                                       player_info=None):
        try:
            if player_seasons_df.empty:
                logger.warning("Empty player seasons data provided")
                return None

            df = player_seasons_df.copy()

            df = df.sort_values('SEASON_YEAR')

            if len(df) < 1:
                logger.warning("Insufficient season data for prediction")
                return None

            latest_season = df.iloc[-1]

            input_features = {}

            age = player_info.get(
                'AGE'
            ) if player_info and 'AGE' in player_info else latest_season.get(
                'AGE', 25) + 1
            input_features['AGE'] = age

            experience = len(df)
            input_features['EXPERIENCE'] = experience

            for stat in [
                    'MIN', 'PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV', 'FGM',
                    'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'FG_PCT', 'FG3_PCT',
                    'FT_PCT'
            ]:
                if stat in latest_season:
                    input_features[f'PREV_SEASON_{stat}'] = latest_season[stat]

            if 'GAMES_PLAYED' in latest_season:
                games = latest_season['GAMES_PLAYED']
                input_features['PREV_SEASON_GAMES'] = games

            if len(df) >= 2:
                prev_season = df.iloc[-2]
                for stat in ['PTS', 'AST', 'REB']:
                    if stat in latest_season and stat in prev_season:
                        input_features['SEASON_TREND'] = latest_season[
                            stat] - prev_season[stat]
            else:
                input_features['SEASON_TREND'] = 0

            stats_2yr_avg = {}
            count = 0

            for i in range(min(2, len(df))):
                past_season = df.iloc[-(i + 1)]
                for stat in ['PTS', 'AST', 'REB']:
                    if stat in past_season:
                        key = f'PREV_{stat}_2YR_AVG'
                        if key not in stats_2yr_avg:
                            stats_2yr_avg[key] = 0
                        stats_2yr_avg[key] += past_season[stat]
                count += 1

            for key, total in stats_2yr_avg.items():
                if count > 0:
                    input_features[key] = total / count
                else:
                    input_features[key] = 0

            if player_info:
                for key in player_info:
                    if key != 'NEXT_SEASON' and key in feature_names:
                        input_features[key] = player_info[key]

            input_vector = []
            missing_features = []

            for feature in feature_names:
                if feature in input_features:
                    input_vector.append(input_features[feature])
                else:
                    missing_features.append(feature)
                    input_vector.append(0)

            if missing_features:
                logger.warning(
                    f"Missing features in prediction input: {missing_features}"
                )

            logger.info(
                f"Created season prediction input with {len(feature_names)} features"
            )
            return np.array(input_vector).reshape(1, -1)

        except Exception as e:
            logger.error(f"Error creating season prediction input: {e}")
            return None

    def get_player_context(self,
                           player_id=None,
                           opponent_team_id=None,
                           stat_category='PTS'):
        try:
            context = {}

            if not player_id:
                return context

            context['player_id'] = player_id
            context['stat_category'] = stat_category
            context['opponent_team_id'] = opponent_team_id

            player_data = {}

            if stat_category == 'PTS':
                player_data['scoring_type'] = 'scorer'
                player_data['scoring_emphasis'] = 'mid_range'
            elif stat_category == 'AST':
                player_data['scoring_type'] = 'playmaker'
                player_data['scoring_emphasis'] = 'passing'
            elif stat_category == 'REB':
                player_data['scoring_type'] = 'rebounder'
                player_data['scoring_emphasis'] = 'inside'
            elif stat_category == 'BLK':
                player_data['scoring_type'] = 'defender'
                player_data['scoring_emphasis'] = 'paint_defense'
            elif stat_category == 'STL':
                player_data['scoring_type'] = 'defender'
                player_data['scoring_emphasis'] = 'perimeter_defense'
            elif stat_category == 'FG3M':
                player_data['scoring_type'] = 'shooter'
                player_data['scoring_emphasis'] = 'three_point'
            else:
                player_data['scoring_type'] = 'all_around'
                player_data['scoring_emphasis'] = 'balanced'

            context['player_data'] = player_data

            return context

        except Exception as e:
            logger.error(f"Error getting player context: {e}")
            return {'player_id': player_id, 'stat_category': stat_category}

    def get_safe_prediction(self, model, X_input, default_value=None):
        try:
            if X_input is None or (hasattr(X_input, 'size')
                                   and X_input.size == 0):
                logger.warning("Empty prediction input provided")
                return default_value

            raw_prediction = model.predict(X_input)

            if raw_prediction is None or (hasattr(raw_prediction, 'size')
                                          and raw_prediction.size == 0):
                logger.warning("Model returned empty prediction")
                return default_value

            prediction_value = raw_prediction[0]

            return max(0, prediction_value)

        except Exception as e:
            logger.error(f"Error in get_safe_prediction: {e}")
            return default_value

    def enhance_prediction(self,
                           raw_prediction,
                           player_context=None,
                           stat_category='PTS',
                           minutes=None,
                           confidence_interval=False):
        try:
            if raw_prediction is None:
                return {"value": 0, "confidence": "low"}

            enhanced = {}

            prediction_value = float(raw_prediction)
            enhanced["value"] = max(0, prediction_value)

            if confidence_interval:
                error_margin = prediction_value * 0.15
                enhanced["confidence_interval"] = (max(
                    0, prediction_value - error_margin), prediction_value +
                                                   error_margin)

            if player_context:
                player_data = player_context.get('player_data', {})
                scoring_type = player_data.get('scoring_type', 'all_around')

                adjustments = {}

                if minutes is not None:
                    if minutes < 20:
                        adjustments['low_minutes'] = -0.15
                    elif minutes > 35:
                        adjustments['high_minutes'] = 0.1

                if scoring_type == 'scorer' and stat_category == 'PTS':
                    adjustments['scorer_bonus'] = 0.05
                elif scoring_type == 'playmaker' and stat_category == 'AST':
                    adjustments['playmaker_bonus'] = 0.05
                elif scoring_type == 'rebounder' and stat_category == 'REB':
                    adjustments['rebounder_bonus'] = 0.05

                total_adjustment = 1.0
                for factor, value in adjustments.items():
                    total_adjustment += value

                enhanced["adjustment_factors"] = adjustments
                adjusted_value = prediction_value * total_adjustment
                enhanced["value"] = max(0, round(adjusted_value, 1))

                enhanced[
                    "strategy_recommendations"] = self._generate_strategy_recommendations(
                        player_context, stat_category, enhanced["value"])

                enhanced["feature_explanations"] = {
                    "PTS_AVG_5":
                    "Points averaged over last 5 games",
                    "FG_PCT_AVG_5":
                    "Field goal percentage averaged over last 5 games",
                    "MIN_AVG_5":
                    "Minutes played averaged over last 5 games",
                    "TS_PCT":
                    "True Shooting Percentage - efficiency metric that accounts for 2-point, 3-point, and free throws",
                    "BACK_TO_BACK":
                    "Whether the game is on the second night of a back-to-back",
                    "PREV_WIN_PCT_5":
                    "Team's win percentage over the last 5 games",
                    "OPP_PTS_DIFF":
                    "Difference between opponent's points scored vs. their season average"
                }

            enhanced["confidence"] = "high" if prediction_value > 0 else "low"

            return enhanced

        except Exception as e:
            logger.error(f"Error enhancing prediction: {e}")
            return {
                "value": raw_prediction if raw_prediction is not None else 0,
                "confidence": "low"
            }

    def _generate_strategy_recommendations(self, player_context, stat_category,
                                           predicted_value):
        player_data = player_context.get('player_data',
                                         {}) if player_context else {}
        scoring_type = player_data.get('scoring_type', 'all_around')
        scoring_emphasis = player_data.get('scoring_emphasis', 'balanced')

        recommendations = {
            "offensive_strategy": [],
            "defensive_focus": [],
            "lineup_recommendations": []
        }

        if stat_category == 'PTS':
            recommendations["offensive_strategy"] = [
                "Increase pick-and-roll actions with primary ball handler",
                f"Run isolation plays when defensive mismatches occur",
                f"Focus on {scoring_emphasis} shots to maximize efficiency"
            ]

            recommendations["defensive_focus"] = [
                "Limit transition opportunities where player excels",
                "Force to use non-dominant hand on drives",
                "Strategic double-teams when catching in rhythm spots"
            ]

        elif stat_category == 'AST':
            recommendations["offensive_strategy"] = [
                "Position shooters in optimal spacing around playmaker",
                "Run off-ball screens to create passing lanes",
                "Utilize dribble hand-offs to create movement"
            ]

            recommendations["defensive_focus"] = [
                "Aggressive defense at point of attack",
                "Switch screens only with defensive-minded players",
                "Focus on denying passing lanes rather than steal attempts"
            ]

        elif stat_category == 'REB':
            recommendations["offensive_strategy"] = [
                "Ensure proper box-out positioning on shot attempts",
                "Create mismatches through screening action",
                "Attack offensive glass from weak side"
            ]

            recommendations["defensive_focus"] = [
                "Box out aggressively on all shot attempts",
                "Limit offensive rebounding positions with help defense",
                "Prioritize defensive rebounding over transition offense"
            ]

        recommendations["lineup_recommendations"] = [
            "Pair with complementary players who enhance production",
            "Adjust minutes distribution based on game flow and matchups",
            "Strategic substitution patterns to maximize efficiency"
        ]

        return recommendations

    def prepare_team_prediction_data(self, team_game_logs):
        try:
            if team_game_logs.empty:
                logger.warning("Empty team game logs data provided")
                return np.array([]), np.array([]), []

            df = team_game_logs.copy()

            if 'GAME_DATE' in df.columns:
                df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
                df = df.sort_values('GAME_DATE')

            logger.info(f"Preparing team prediction data with {len(df)} games")

            df['WIN'] = df['WL'].apply(lambda x: 1 if x == 'W' else 0)

            df['GAME_NUMBER'] = range(1, len(df) + 1)

            for window in [3, 5, 10]:
                if len(df) >= window:
                    df[f'WIN_STREAK_{window}'] = df['WIN'].rolling(
                        window=window).sum()
                    df[f'WIN_PCT_{window}'] = df['WIN'].rolling(
                        window=window).mean()

                    for stat in [
                            'PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
                            'AST', 'REB', 'STL', 'BLK', 'TOV', 'PLUS_MINUS'
                    ]:
                        if stat in df.columns:
                            df[f'{stat}_AVG_{window}'] = df[stat].rolling(
                                window=window).mean()
                            df[f'{stat}_STD_{window}'] = df[stat].rolling(
                                window=window).std()

            df['PREV_GAME_WIN'] = df['WIN'].shift(1).fillna(0)

            for n in range(1, 4):
                df[f'WON_{n}_AGO'] = df['WIN'].shift(n).fillna(0)

            if 'MATCHUP' in df.columns:
                df['HOME'] = df['MATCHUP'].apply(lambda x: 1
                                                 if 'vs.' in str(x) else 0)

            if 'GAME_DATE' in df.columns:
                df['DAYS_REST'] = (df['GAME_DATE'] -
                                   df['GAME_DATE'].shift(1)).dt.days
                df['DAYS_REST'] = df['DAYS_REST'].fillna(3)

                df['BACK_TO_BACK'] = (df['DAYS_REST'] < 2).astype(int)

            if 'PLUS_MINUS' in df.columns:
                df['BLOWOUT_WIN'] = ((df['PLUS_MINUS'] > 15) &
                                     (df['WIN'] == 1)).astype(int)
                df['CLOSE_GAME'] = (abs(df['PLUS_MINUS']) < 5).astype(int)

            if all(col in df.columns for col in ['PTS', 'FGA', 'FTA']):
                df['TS_PCT'] = df.apply(
                    lambda row: row['PTS'] / (2 *
                                              (row['FGA'] + 0.44 * row['FTA']))
                    if (row['FGA'] + 0.44 * row['FTA']) > 0 else 0,
                    axis=1)

            df = df.iloc[max(10, int(len(df) * 0.1)):]

            feature_cols = []

            for window in [3, 5, 10]:
                for stat in [
                        'WIN_STREAK', 'WIN_PCT', 'PTS_AVG', 'FGM_AVG',
                        'FG3M_AVG', 'AST_AVG', 'REB_AVG', 'STL_AVG', 'BLK_AVG',
                        'TOV_AVG', 'PLUS_MINUS_AVG'
                ]:
                    col = f'{stat}_{window}'
                    if col in df.columns:
                        feature_cols.append(col)

            feature_cols.extend([
                'PREV_GAME_WIN', 'WON_1_AGO', 'WON_2_AGO', 'WON_3_AGO', 'HOME',
                'DAYS_REST', 'BACK_TO_BACK', 'GAME_NUMBER'
            ])

            if 'TS_PCT' in df.columns:
                feature_cols.append('TS_PCT')

            if 'BLOWOUT_WIN' in df.columns and 'CLOSE_GAME' in df.columns:
                feature_cols.extend(['BLOWOUT_WIN', 'CLOSE_GAME'])

            df_clean = df.dropna(subset=feature_cols + ['WIN'])

            logger.info(
                f"Clean data shape after preprocessing: {df_clean.shape}")

            if len(df_clean) < 10:
                logger.warning(
                    f"Not enough clean data points for team prediction (only {len(df_clean)} available)"
                )
                return np.array([]), np.array([]), []

            numeric_features = []
            for col in feature_cols:
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col],
                                                  errors='raise')
                    numeric_features.append(col)
                except:
                    logger.warning(
                        f"Dropping non-numeric feature for team prediction: {col}"
                    )

            feature_cols = numeric_features

            X = df_clean[feature_cols].values
            y = df_clean['WIN'].values

            logger.info(
                f"Prepared team prediction data with {X.shape[0]} samples and {X.shape[1]} features"
            )
            return X, y, feature_cols

        except Exception as e:
            logger.error(f"Error preparing team prediction data: {e}")
            return np.array([]), np.array([]), []
