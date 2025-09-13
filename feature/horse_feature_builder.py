from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime 
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
from metadata.columns import horse_feature_columns

from utils.logger import setup_logger
logger = setup_logger('horse_feature_builder')

YEARLY_HORSE_DIR = Path ('data/yearly_parquet/horse')
SAVE_DIR = Path('data/prcessed_parquet')
SAVE_DIR.mkdir(exist_ok=True)
PREDICT_FEATURE_DIR = Path('data/predict_features')
PREDICT_FEATURE_DIR.mkdir(exist_ok=True)
CSV_DIR = Path('data/csv')
CSV_DIR.mkdir(exist_ok=True)

def process_horse_data():
    logger.info('Starting horse feature building pipeline...')
    horse_df = load_and_merge_data()
    horse_df = add_basic_features(horse_df)
    horse_df = process_time_features(horse_df)
    horse_df = add_speed_index_features(horse_df)
    horse_df = add_gi_features(horse_df)
    horse_df = add_positional_features(horse_df)   
    horse_df = add_kde_features(horse_df)
    finalize_and_save(horse_df)
    

def load_and_merge_data():
    horse_files = list(YEARLY_HORSE_DIR.glob('horse_*.parquet'))
    horse_df = pd.concat([pd.read_parquet(f) for f in horse_files], axis=0)
    cols_for_merge = ['race_id','location', 'ground_type', 'distance', 'is_outer_course', 'race_class_level', 'datetime', 'total_horse_number', 'field_size_bin', 'corners_num','norm_tough_gi', 'time_decay_weight', 'high_pace_proba_log', 'race_age']
    race_df_for_merge = pd.read_parquet(SAVE_DIR / 'race_features.parquet')[cols_for_merge]
    
    columns_to_convert = ['race_id', 'horse_id', 'tamer_id', 'owner_id', 'rider_id']
    horse_df[columns_to_convert] = horse_df[columns_to_convert].apply(lambda x: x.astype(str))

    horse_df['race_id'] = horse_df['race_id'].str.extract(r'(\d{12})')[0]
    horse_df = horse_df[horse_df['race_id'].notnull()]
    horse_df = pd.merge(horse_df, race_df_for_merge, on='race_id', how='left')
    logger.info('Finished loading horse data. Shape: %s', horse_df.shape)
    return horse_df


def add_basic_features(horse_df):
    # 基本的な特徴量の追加
    logger.info('Start adding basic features')
    horse_df['datetime'] = horse_df['datetime'].replace(['0', 0], np.nan)
    horse_df['datetime'] = pd.to_datetime(horse_df['datetime'], errors='coerce')
    horse_df = horse_df.dropna(subset=['datetime'])  
    horse_df['race_year'] = horse_df['datetime'].dt.year.astype(int)  

    # 順位に関する特徴量
    horse_df['total_horse_number'] = horse_df['total_horse_number'].fillna(18)
    horse_df['rank'] = horse_df['rank'].str.replace(r'\(?(降|再)\)?', '', regex=True)
    horse_df = horse_df[~horse_df['rank'].isin(['取', '除', '失', '中'])]
    horse_df = horse_df.dropna(subset=['rank']) # type: ignore
    horse_df['rank'] = horse_df['rank'].astype(int)
    horse_df['norm_rank'] = 1 - (horse_df['rank'] - 1) / (horse_df['total_horse_number'] - 1)
    horse_df['rank_cat'] =  pd.cut(horse_df['norm_rank'],   bins=list(np.linspace(0, 1, 19)), labels=False, include_lowest=True)
    
    # スタート位置に関する特徴量
    horse_df['norm_horse_number'] = (horse_df['horse_number'] - 1) / (horse_df['total_horse_number'] - 1)
    horse_df['horse_number_cat'] = pd.cut(horse_df['norm_horse_number'],   bins=list(np.linspace(0, 1, 19)), labels=False, include_lowest=True)
    horse_df['is_inner'] = (horse_df['norm_horse_number'] <= 0.3).astype(int)
    horse_df['is_middle'] = ((horse_df['norm_horse_number'] > 0.3) & (horse_df['norm_horse_number'] <= 0.7)).astype(int)
    horse_df['is_outer'] = (horse_df['norm_horse_number'] > 0.7).astype(int)

    # 性齢に関する特徴量
    def extract_and_replace(df, column, pattern, new_column, replace_value):
        extracted = df[column].str.extract(f'({pattern})', expand=False)
        df[new_column] = extracted.fillna(0).replace(pattern, replace_value)
        return df

    sex_patterns = {'is_gelding': 'セ', 'is_mare': '牝', 'is_stallion': '牡'}
    for new_column, pattern in sex_patterns.items():
        horse_df = extract_and_replace(horse_df, 'sex_and_age', pattern, new_column, 1)
    horse_df['sex_and_age'] = horse_df['sex_and_age'].str.replace(r'[牝牡セ]', '', regex=True).astype(int)
    horse_df = horse_df.rename(columns={'sex_and_age': 'age'})
    horse_df['age_cat'] = pd.cut(horse_df['age'], bins=[0, 2, 3, 4, 5, 6, 7, np.inf], labels=[0, 1, 2, 3, 4, 5, 6],include_lowest=True, right=True).astype(int)
    horse_df.loc[horse_df['age'] > 3, 'race_age'] = 4

    # 人気に関する特徴量
    horse_df['win_odds'] = horse_df['win_odds'].astype(float)
    horse_df['log_win_odds'] = -np.log1p(horse_df['win_odds'])
    implied_proba = 1 / (horse_df['win_odds'] + 1e-6)
    horse_df['implied_proba_logit'] = np.log(implied_proba/ (1 - implied_proba))
    horse_df['popular'] = horse_df['popular'].fillna(horse_df['total_horse_number']).astype(int)
    horse_df['norm_popularity'] = 1 - (horse_df['popular'] - 1) / (horse_df['total_horse_number'] - 1)
    horse_df['rank_vs_popular'] = (horse_df['norm_rank'] - horse_df['norm_popularity']) 
    
    # 不利に関する特徴量
    remark_mapping = {'出遅れ': 'is_slow_break', '出脚鈍い': 'is_slow_break', '躓く': 'is_slow_break', 'アオル': 'is_slow_break',
                      'S不利': 'is_trouble', 'S接触':'is_trouble', 'Sヨレル': 'is_trouble', '直線不利': 'is_trouble'}
    for key, value in remark_mapping.items():
        horse_df.loc[horse_df['remark'].str.contains(key, na=False), value] = 1

    horse_df['slow_break_rate5'] = (
        horse_df.groupby('horse_id')['is_slow_break']
          .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum()) / 5
    )

    horse_df['trouble_rate5'] = (
        horse_df.groupby('horse_id')['is_trouble']
          .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum()) / 5
    ).fillna(0)

    # 馬体重に関する特徴量
    horse_df['horse_weight'] = horse_df['horse_weight'].str.replace(r'\(([-|+]?\d*)\)', '', regex=True)\
        .replace('計不', np.nan).astype(float)
    horse_weight_rolling = horse_df.sort_values('datetime').groupby('horse_id')['horse_weight']
    horse_df['horse_wight_mean5'] = horse_weight_rolling.transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    horse_df['horse_weight_std5'] = np.log1p(horse_weight_rolling.transform(lambda x: x.shift(1).rolling(5, min_periods=2).std()))
    horse_df['horse_weight'] = horse_df.groupby('horse_id')['horse_weight'].transform(lambda x: x.fillna(x.mean()))
    horse_df['burden_ratio'] = horse_df['burden_weight'] / horse_df['horse_weight']
    horse_df['burden_ratio_to_mean5'] = horse_df['burden_weight'] / horse_df['horse_weight_mean5']
    horse_df['horse_weight_diff_mean5'] = horse_df['horse_weight'] - horse_df['horse_weight_mean5']
    burden_weight_mean5 = horse_df.groupby('horse_id')['burden_weight']\
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    horse_df['burden_weight_diff_mean5'] = (horse_df['burden_weight'] - burden_weight_mean5).fillna(0) 
    
    # 厩舎に関する特徴量
    affiliation_mapping = {'西': 'is_Ritto', '東':'is_Miho', '地': 'is_local_or_foreign', '外': 'is_local_or_foreign'}
    for key, value in affiliation_mapping.items():
        horse_df.loc[horse_df['affiliation'].str.contains(key, na=False), value] = 1
    racecourse_distance_from_training_center = {
        'Tokyo': {'is_Ritto': 430, 'is_Miho': 120},
        'Nakayama': {'is_Ritto': 450, 'is_Miho': 75},
        'Kyoto': {'is_Ritto': 45, 'is_Miho': 430},
        'Hanshin': {'is_Ritto': 85, 'is_Miho': 450},
        'Chukyo': {'is_Ritto': 150, 'is_Miho': 370},
        'Fukushima': {'is_Ritto': 530, 'is_Miho': 290},
        'Niigata': {'is_Ritto': 390, 'is_Miho': 290},
        'Sapporo': {'is_Ritto': 1040, 'is_Miho': 850},
        'Hakodate': {'is_Ritto': 880, 'is_Miho': 720},
        'Kokura': {'is_Ritto': 500, 'is_Miho': 1000},
    }

    def compute_distance(row: pd.Series) -> float:
        loc = row.get('location')

        if pd.isna(loc) or loc not in racecourse_distance_from_training_center:
            return float('nan')

        dist_map = racecourse_distance_from_training_center[loc]

        if row.get('is_Ritto', 0) == 1:
            return float(dist_map.get('is_Ritto', float('nan')))
        elif row.get('is_Miho', 0) == 1:
            return float(dist_map.get('is_Miho', float('nan')))
        elif row.get('is_local_or_foreign', 0) == 1:
            return float('nan')
        else:
            return float('nan')

    horse_df['center_dist'] = horse_df.apply(compute_distance, axis=1)
    horse_df['center_dist'] = horse_df['center_dist'].fillna(horse_df['center_dist'].mean())

    return horse_df


def compute_time_causal_group_stats(df, value_col, group_cols):
    # 時間に関する特徴量のグループごとの平均と標準偏差を計算
    df = df.copy()
    grouped = df.groupby(group_cols + ['race_year'])[value_col].agg(['mean', 'std', 'count']).reset_index()
    grouped = grouped.rename(columns={'mean': f'{value_col}_mean', 'std': f'{value_col}_std', 'count': 'count'})

    result_df = []
    for year in sorted(df['race_year'].unique()):
        df_this_year = df[df['race_year'] == year]
        df_past = grouped[grouped['race_year'] < year]
        merged = df_this_year.merge(
            df_past.groupby(group_cols).agg({f'{value_col}_mean': 'mean', f'{value_col}_std': 'mean', 'count': 'sum'}).reset_index(),
            on=group_cols,
            how='left'
        )
        result_df.append(merged[[f'{value_col}_mean', f'{value_col}_std']])
    
    result_df = pd.concat(result_df, axis=0, ignore_index=True)
    return result_df


def process_time_features(df):
    # 時間に関する特徴量の追加
    logger.info('Start processing time features')
    df['goal_time'] = (
        pd.to_datetime(df['goal_time'], format='%M:%S.%f') -
        pd.to_datetime('00:00.0', format='%M:%S.%f')
    ).dt.total_seconds()
    df['last3f_time'].rename()
    winner_goal_time = df.groupby('race_id')['goal_time'].transform('min')
    df['goal_time_diff_winner'] = df['goal_time'] - winner_goal_time

    time_diff_mask = df['goal_time_diff_winner'] > 25
    for col in ['goal_time', 'last3f_time']:
        df.loc[time_diff_mask, col] = (
            df.groupby('race_id')[col]
            .transform(lambda x: x.dropna().sort_values().iloc[1] if len(x.dropna()) > 1 else x.dropna().iloc[0])
        )
    
    df['goal_time'] = df['goal_time'].fillna(df.groupby('race_id')['goal_time'].transform('max'))
    df['last3f_time'] = df['last3f_time'].fillna(df.groupby('race_id')['last3f_time'].transform('max'))

    winner_3f = df[df['rank'] == 1].groupby('race_id')['last3f_time'].first()
    winner_last3f_time = df['race_id'].map(winner_3f)
    top_last3f_time = df.groupby('race_id')['last3f_time'].transform('min')
    df = df.drop(columns=['goal_time_diff_winner'], axis=1)
    df['is_high_level'] = (df['race_class_level'] > 2).astype(int)
    df['is_tough_track'] = (df['norm_tough_gi'] - 0.2 > 0).astype(int).fillna(0)
    group_cols = ['location', 'ground_type', 'distance', 'is_outer_course', 'race_class_level', 'is_tough_track', 'race_age']
    group_cols_level =  ['location', 'ground_type', 'distance', 'is_outer_course', 'is_high_level']
    group_cols_dist = ['ground_type', 'distance', 'is_high_level']

    group_counts = df.groupby(group_cols)['is_tough_track'].transform('count')
    df.loc[group_counts < 5, 'is_tough_track'] = 0


    df[['goal_time_mean', 'goal_time_std']] = compute_time_causal_group_stats(df, 'goal_time', group_cols)
    df[['last3f_time_mean', 'last3f_time_std']] = compute_time_causal_group_stats(df, 'last3f_time', group_cols)
    df[['goal_time_level_mean', 'goal_time_level_std']] = compute_time_causal_group_stats(df, 'goal_time', group_cols_level)
    df[['last3f_time_level_mean', 'last3f_time_level_std']] = compute_time_causal_group_stats(df, 'last3f_time', group_cols_level)
    df[['goal_time_dist_mean', 'goal_time_dist_std']] = compute_time_causal_group_stats(df, 'goal_time', group_cols_dist)
    df[['last3f_time_dist_mean', 'last3f_time_dist_std']] = compute_time_causal_group_stats(df, 'last3f_time', group_cols_dist)

    df['goal_time_zscore_mean'] = - ((df['goal_time'] - df['goal_time_mean']) / df['goal_time_std']).fillna(0)
    df['last3f_time_zscore_mean'] = - ((df['last3f_time'] - df['last3f_time_mean']) / df['last3f_time_std']).fillna(0)
    df['goal_time_zscore_winner'] = - ((df['goal_time'] - winner_goal_time) / df['goal_time_std']).fillna(0)
    df['last3f_time_zscore_winner'] = - ((df['last3f_time'] - winner_last3f_time) / df['last3f_time_std']).fillna(0)

    df['winner_goal_time_zscore_vs_courseclass'] = - ((winner_goal_time - df['goal_time_mean']) / (df['goal_time_std'] + 1e-6)).fillna(0)
    df['winner_last3f_time_zscore_vs_courseclass'] = - ((winner_last3f_time - df['last3f_time_mean']) / (df['last3f_time_std'] + 1e-6)).fillna(0)
    df['top_last3f_time_zscore_vs_courseclass'] = - ((top_last3f_time - df['last3f_time_mean']) / (df['last3f_time_std'] + 1e-6)).fillna(0)

    grouped = df.groupby('race_id')
    df['goal_time_zscore_mean5'] = (
        df.groupby('horse_id')['goal_time_zscore_mean']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    goal_mean5 = grouped['goal_time_zscore_mean5'].transform('mean')
    df['goal_time_zscore_mean5'] = df['goal_time_zscore_mean5'].fillna(goal_mean5)

    df['last3f_time_zscore_mean5'] = (
        grouped['last3f_time_zscore_mean']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    last3f_mean5 = grouped['last3f_time_zscore_mean5'].transform('mean')
    df['last3f_time_zscore_mean5'] = df['last3f_time_zscore_mean5'].fillna(last3f_mean5)

    pred_goal_time_rank = grouped['goal_time_zscore_mean5'].rank(method='max')
    pred_last3f_time_rank = grouped['last3f_time_zscore_mean5'].rank(method='max')
    df['norm_goal_rank_pred'] = (1 - (pred_goal_time_rank - 1) / (df['total_horse_number'] - 1)).fillna(0.5)
    df['norm_last3f_rank_pred'] = (1 - (pred_last3f_time_rank - 1) / (df['total_horse_number'] - 1)).fillna(0.5)
    df['goal_rank_pred_zscore'] = - (
        (df['goal_time_zscore_mean5'] - goal_mean5) /
        (grouped['goal_time_zscore_mean5'].transform('std') + 1e-3)
    ).fillna(0)
    df['last3f_rank_pred_zscore'] = - (
        (df['last3f_time_zscore_mean5'] - last3f_mean5) /
        (grouped['last3f_time_zscore_mean5'].transform('std') + 1e-3)
    ).fillna(0)
    df['goal_time_zscore_mean5'] = df['goal_time_zscore_mean5'].fillna(0)
    df['last3f_time_zscore_mean5'] = df['last3f_time_zscore_mean5'].fillna(0)

    df['goal_time_course_zscore'] = -(((df['goal_time_level_mean'])- df['goal_time_dist_mean']) / df['goal_time_dist_std']).fillna(0)
    df['last3f_time_course_zscore'] = -(((df['last3f_time_level_mean'])- df['last3f_time_dist_mean']) / df['last3f_time_dist_std']).fillna(0)
    df[group_cols_level + ['high_pace_proba_log', 'goal_time_course_zscore', 'last3f_time_course_zscore']].drop_duplicates().to_csv(PREDICT_FEATURE_DIR / 'time_course_zscore.csv', encoding='utf-8', index=False)
    drop_cols = ['goal_time', 'last3f_time','goal_time_mean', 'last3f_time_mean', 'goal_time_std',  'last3f_time_std', 'goal_time_level_std', 'last3f_time_level_std', 'goal_time_level_mean', 'last3f_time_level_mean', 'goal_time_dist_mean','last3f_time_dist_mean',  'goal_time_dist_std', 'last3f_time_dist_std', 'is_high_level',  'is_tough_track', 'race_age', 'goal_time_zscore_mean5', 'last3f_time_zscore_mean5']
    return df.drop(columns=drop_cols)


def add_gi_features(horse_df):
    # 馬場指数に関する特徴量の追加
    logger.info('Start adding ground index features')
    tough_mask = horse_df['norm_tough_gi'] > 0

    horse_df['norm_rank_weighted'] = horse_df['norm_rank'] * horse_df['norm_tough_gi'] * tough_mask
    horse_df['goal_time_weighted'] = horse_df['goal_time_zscore_mean'] * horse_df['norm_tough_gi'] * tough_mask
    horse_df = horse_df.sort_values('datetime')
    horse_df['ground_pref_rank'] = np.arcsinh(
        horse_df.groupby('horse_id')['norm_rank_weighted']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    horse_df['tough_ground_pref_rank'] = np.arcsinh(
        horse_df.groupby('horse_id')
        .apply(lambda g: g['norm_rank_weighted'].shift(1).rolling(5, min_periods=1).sum() /
                        g['norm_tough_gi'].shift(1).where(tough_mask).rolling(5, min_periods=1).sum().clip(lower=1))
        .reset_index(level=0, drop=True)
    )

    horse_df['ground_pref_time'] = np.arcsinh(
        horse_df.groupby('horse_id')['goal_time_weighted']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    horse_df['tough_ground_pref_time'] = np.arcsinh(
        horse_df.groupby('horse_id')
        .apply(lambda g: g['goal_time_weighted'].shift(1).rolling(5, min_periods=1).sum() /
                        g['norm_tough_gi'].shift(1).where(tough_mask).rolling(5, min_periods=1).sum().clip(lower=1))
        .reset_index(level=0, drop=True)
    )
    fill_race_mean_cols = ['ground_pref_rank', 'tough_ground_pref_rank', 'ground_pref_time', 'tough_ground_pref_time']
    for col in fill_race_mean_cols:
        horse_df[col] = horse_df.groupby('race_id')[col].transform(lambda x: x.fillna(x.mean()))

    horse_df = horse_df.drop(columns=['norm_rank_weighted', 'goal_time_weighted'])
    return  horse_df


def add_speed_index_features(horse_df):
    # スピード指数に関する特徴量の追加
    logger.info('Start adding speed index features')
    horse_df['speed_index'] = horse_df['speed_index'].astype(str).str.strip().replace('', '0')
    horse_df['speed_index'] = horse_df['speed_index'].astype(int).clip(lower=-100)

    grouped = horse_df.groupby('race_id')
    horse_df['speed_index_mean5'] = (
        horse_df.sort_values('datetime')
        .groupby('horse_id')['speed_index']
        .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    )

    horse_df['speed_index_std5'] = np.log1p(
        horse_df.sort_values('datetime')
        .groupby('horse_id')['speed_index']
        .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).std())
    )
    horse_df['speed_index_std5'] = grouped['speed_index_std5'].transform(lambda x: x.fillna(x.mean()))
    mean_std_2yo = horse_df.loc[horse_df['age'] == 2, 'speed_index_std5'].mean()
    horse_df['speed_index_mean5'] = horse_df['speed_index_mean5'].fillna(mean_std_2yo)


    horse_df['speed_index_momentum5'] = (
        horse_df.sort_values(['horse_id', 'datetime'])
        .groupby('horse_id')['speed_index']
        .transform(lambda x: x.diff().shift(1).rolling(window=5, min_periods=1).mean())
    )
    horse_df['speed_index_momentum5'] = grouped['speed_index_momentum5'].transform(lambda x: x.fillna(x.mean()))

    fill_race_mean_cols = ['speed_index_mean5', 'speed_index_std5', 'speed_index_momentum5']
    for col in fill_race_mean_cols:
        horse_df[col] = horse_df.groupby('race_id')[col].transform(lambda x: x.fillna(x.mean()))

    pred_speed_rank = grouped['speed_index_mean5'].rank(method='max')
    horse_df['norm_speed_rank_pred'] = (1 - (pred_speed_rank - 1) / (horse_df['total_horse_number'] - 1)).fillna(0.5)
    horse_df['speed_index_zscore'] = (
            (horse_df['speed_index_mean5'] - grouped['speed_index_mean5'].transform('mean')) /
            grouped['speed_index_mean5'].transform('std')
        ).fillna(0)
    horse_df = horse_df.drop(columns=['speed_index_mean5'])

    return horse_df


def add_positional_features(horse_df):
    # 位置取りに関する特徴量の追加
    logger.info('Start adding positional features')
    norm_pos_array = (
        horse_df['rank_path']
        .str.split('-', expand=True)
        .astype(float)
        .div(horse_df['total_horse_number'], axis=0)
    )

    pos_std = norm_pos_array.std(axis=1)
    horse_df['pos_mean'] = norm_pos_array.mean(axis=1).fillna(0.5)
    horse_df['pos_volatility_index'] = np.arcsinh(pos_std / (horse_df['pos_mean'] + 0.01))

    pos_diff = norm_pos_array.diff(axis=1)
    horse_df['pos_diff_mean'] = pos_diff.mean(axis=1).fillna(0)
    horse_df['pos_diff_stdlog'] = np.log1p(pos_diff.std(axis=1))

    horse_df['norm_early_pos'] = norm_pos_array.iloc[:, 0]
    horse_df['norm_late_pos'] = norm_pos_array.ffill(axis=1).iloc[:, -1]
    horse_df['pos_gain_ratio'] = np.arcsinh((horse_df['norm_early_pos'] - horse_df['norm_late_pos']) / (horse_df['norm_early_pos'] + 0.01)).clip(-1.5, 0.8).fillna(0)

    horse_df['pos_trend'] = np.arcsinh(horse_df['norm_early_pos'] - horse_df['norm_late_pos']).fillna(0)
    horse_df['pos_trend_per_corner'] = horse_df['pos_trend'] / (horse_df['corners_num'] + 0.1)

    horse_df = horse_df.sort_values('datetime')
    horse_df['norm_early_pos5'] = (
        horse_df.groupby('horse_id')['norm_early_pos']
        .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    )
    horse_df['norm_late_pos5'] = (
        horse_df.groupby('horse_id')['norm_late_pos']
        .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    )
    grouped = horse_df.groupby('race_id')
    horse_df['norm_early_pos5'] = grouped['norm_early_pos5'].transform(lambda x: x.fillna(x.mean()))
    horse_df['norm_late_pos5'] = grouped['norm_late_pos5'].transform(lambda x: x.fillna(x.mean()))

    pred_early_rank = grouped['norm_early_pos5'].rank(method='min')
    pred_late_rank = grouped['norm_late_pos5'].rank(method='min')
    horse_df['norm_e_rank_pred'] = ((pred_early_rank - 1) / (horse_df['total_horse_number'] - 1)).fillna(0.5)
    horse_df['norm_l_rank_pred'] = ((pred_late_rank - 1) / (horse_df['total_horse_number'] - 1)).fillna(0.5)
    horse_df['relative_rank_gain'] = (horse_df['norm_l_rank_pred'] - horse_df['norm_e_rank_pred']).fillna(0)

    horse_df['e_rank_pred_zscore'] = (
        (horse_df['norm_early_pos5'] - grouped['norm_early_pos5'].transform('mean')) /
        grouped['norm_early_pos5'].transform('std')
    ).fillna(0)

    horse_df['l_rank_pred_zscore'] = (
        (horse_df['norm_late_pos5'] - grouped['norm_late_pos5'].transform('mean')) /
        grouped['norm_late_pos5'].transform('std')
    ).fillna(0)

    horse_df['rank_path_count'] = (horse_df['rank_path'].str.count('-') + 1).fillna(0)

    def normalize_and_bin(series):
        return pd.cut(
            (series.clip(0.07, 1) - 0.07) / 0.93,
            bins=list(np.linspace(0, 1, 19)),
            labels=False,
            include_lowest=True
        )

    horse_df['early_pos_cat'] = normalize_and_bin(horse_df['norm_early_pos'])
    horse_df['late_pos_cat'] = normalize_and_bin(horse_df['norm_late_pos'])
    horse_df['early_pos5_cat'] = normalize_and_bin(horse_df['norm_early_pos5'])
    horse_df['late_pos5_cat'] = normalize_and_bin(horse_df['norm_late_pos5'])

    horse_df['norm_early_pos'] = grouped['norm_early_pos'].fillna(0.5)
    horse_df['norm_late_pos'] = grouped['norm_late_pos'].fillna(0.5)

    return horse_df.drop(['rank_path','pos_mean'], axis=1)


def add_kde_features(horse_df):
    # KDEに基づく特徴量の追加
    logger.info('Start adding KDE features')
    horse_df = (
        horse_df.sort_values(['horse_id', 'datetime'], ascending=[True, False])
        .groupby('horse_id')
        .apply(lambda g: g.assign(distance_diff=g['distance'].diff(-1)))
        .reset_index(drop=True)
        .sort_values('datetime')

    )
    horse_df['distance_diff'] = horse_df['distance_diff'].clip(-800, 800)
    horse_df['norm_distance_diff'] = (horse_df['distance_diff'] + 800) / 1600
    horse_df['distance_diff_cat'] = (horse_df['distance_diff'].clip(-800, 800) + 800) // 100
    
    group_cols = ['location', 'ground_type', 'distance', 'is_outer_course', 'field_size_bin']
    group_cols_dd = ['location', 'ground_type', 'distance', 'is_outer_course'] 

    kde_early_df = train_and_predict_kde(horse_df, 'norm_early_pos', group_cols)
    kde_early_pred_df = train_and_predict_kde(horse_df, 'norm_early_pos5', group_cols)
    kde_late_df = train_and_predict_kde(horse_df, 'norm_late_pos', group_cols)
    kde_late_pred_df = train_and_predict_kde(horse_df, 'norm_late_pos5', group_cols)
    kde_num_df = train_and_predict_kde(horse_df, 'norm_horse_number', group_cols)
    kde_dd_df = train_and_predict_kde(horse_df, 'norm_distance_diff', group_cols_dd)
    
    horse_df = horse_df.merge(kde_early_df,  on=group_cols + ['norm_early_pos'], how='left')
    horse_df = horse_df.merge(kde_early_pred_df, on=group_cols + ['norm_early_pos5',], how='left')
    horse_df = horse_df.merge(kde_late_df,   on=group_cols + ['norm_late_pos5'], how='left')
    horse_df = horse_df.merge(kde_late_pred_df,  on=group_cols + ['norm_late_pos5'], how='left')
    horse_df = horse_df.merge(kde_num_df,    on=group_cols + ['norm_horse_number'], how='left')
    horse_df = horse_df.merge(kde_dd_df, on=group_cols_dd + ['norm_distance_diff'], how='left')
    horse_df['distance_diff'] = horse_df['distance_diff'].fillna(0)
    

    kde_cols = [col for col in horse_df.columns if col.endswith('_kde')]
    for col in kde_cols:
        grouped = horse_df.groupby('race_id')
        mean = grouped[col].transform('mean')
        std = grouped[col].transform('std') + 1e-3  
        horse_df[col] = horse_df[col].fillna(mean)
        horse_df[col] = ((horse_df[col] - mean) / std).clip(-3, 3)
        horse_df[col] = horse_df[col].fillna(0.0)
    return horse_df

def train_and_predict_kde(horse_df, feature_col, group_cols, save_model=True, min_samples=10, decay_weight=0.95):
    # KDEモデルの訓練と予測
    result_rows = []
    horse_df = horse_df.copy()
    prefix = feature_col.replace('norm_', '')
    this_year = datetime.now().year
    for year in tqdm(range(2009, this_year + 1), desc=f'KDE {feature_col} (causal)'):
        win_kde_dict = {}
        show_kde_dict = {}
        df_train = horse_df[horse_df['race_year'] < year]
        df_eval = horse_df[horse_df['race_year'] == year]
        eval_keys = set(tuple(row) for _, row in df_eval[group_cols].drop_duplicates().iterrows())

        for key, group in df_train.groupby(group_cols):
            if key not in eval_keys and this_year != year:
                continue
            x = group[feature_col].dropna()
            if len(x) < min_samples:
                continue

            X = x.values.reshape(-1, 1)
            group = group.loc[x.index]

            decay = np.power(decay_weight, (year - group['race_year']).clip(0, 100))

            is_win = (group['rank'] == 1).astype(float)
            is_show = (group['rank'] <= 3).astype(float)

            win_weights = decay * is_win
            show_weights = decay * is_show

            if win_weights.sum() == 0 or show_weights.sum() == 0:
                continue

            x_vals = X.squeeze()
            mean = np.mean(x_vals)
            var = np.var(x_vals)
            bw = np.clip(1.06 * np.sqrt(var) * len(X) ** (-1 / 5), 0.05, 1.5)


            win_kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(X, sample_weight=win_weights)
            show_kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(X, sample_weight=show_weights)
            win_kde_dict[key] = win_kde
            show_kde_dict[key] = show_kde
        
        for key, group in df_eval.groupby(group_cols):
            win_kde = win_kde_dict.get(key)
            show_kde = show_kde_dict.get(key)
            if win_kde is None or show_kde is None:
                continue

            x_vals = group[feature_col].dropna().unique()
            for x_val in x_vals:
                row = dict(zip(group_cols, key))
                row['race_year'] = year
                row[feature_col] = x_val

                try:
                    row[f'{prefix}_win_kde'] = win_kde.score_samples([[x_val]])[0]
                except:
                    row[f'{prefix}_win_kde'] = 0.0

                try:
                    row[f'{prefix}_show_kde'] = show_kde.score_samples([[x_val]])[0]
                except:
                    row[f'{prefix}_show_kde'] = 0.0

                result_rows.append(row)

        if save_model and (year == this_year):
            pd.to_pickle(win_kde_dict, PREDICT_FEATURE_DIR / f'kde_{prefix}_win.pkl')
            pd.to_pickle(show_kde_dict, PREDICT_FEATURE_DIR / f'kde_{prefix}_show.pkl')


    kde_lookup_df = pd.DataFrame(result_rows)
    return kde_lookup_df
   
    
def finalize_and_save(horse_df): 
    # 最終的なデータフレームの整形と保存
    logger.info('Finalizing and saving')
    horse_df = horse_df[horse_feature_columns]
    logger.info('Columns selected: %d columns, final shape: %s', horse_df.shape[1], horse_df.shape)

    save_path = SAVE_DIR / 'horse_features.parquet'
    horse_df.to_parquet(save_path, index=False)
    logger.info('Saved full horse features to %s', save_path)

    last_year = datetime.now().year - 1
    mask = (horse_df['race_id'].astype(int) < (last_year+1) * 100_000_000) & (horse_df['race_id'].astype(int) > last_year * 100_000_000) 
    horse_df_last_year = horse_df[mask]
    csv_path = CSV_DIR / f'horse_features_{last_year}.csv'
    horse_df_last_year.to_csv(csv_path, encoding='utf-8', index=False)
    logger.info('Saved %d horses from year %d to %s', len(horse_df_last_year), last_year, csv_path)

    sample_path = CSV_DIR / 'horse_features_sampled.csv'
    horse_df.sample(n=500, random_state=42).to_csv(sample_path, encoding='utf-8', index=False)
    logger.info('Saved 500 sampled horses to %s', sample_path)

if __name__ == '__main__':
    process_horse_data()
