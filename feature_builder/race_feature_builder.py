from pathlib import Path
import numpy as np
import pandas as pd
import re
from datetime import datetime
from metadata.course_map import turf_by_loc, turf_by_dist, dirt_by_loc, dirt_by_dist
from metadata.columns import race_feature_columns

from utils.logger import setup_logger
logger = setup_logger('race_feature_builder')

YEARLY_RACE_DIR = Path ('data/yearly_parquet/race')
SAVE_DIR = Path('data/prcessed_parquet')
SAVE_DIR.mkdir(exist_ok=True)
CSV_DIR = Path('data/csv')
CSV_DIR.mkdir(exist_ok=True)

def process_race_data():
    logger.info('Starting race feature builder pipeline...')
    race_df = load_race_data()
    race_df = add_basic_features(race_df)
    race_df = add_course_features(race_df)
    race_df = add_lap_time_features(race_df)
    race_df = apply_time_group_statistics(race_df)
    finalize_and_save(race_df)

def load_race_data():
    # レースデータの読み込みと前処理
    logger.info('Start loading')
    race_files = list(YEARLY_RACE_DIR.glob('race_*.parquet'))
    race_df = pd.concat([pd.read_parquet(f) for f in race_files], axis=0)
    
    race_df['race_id'] = race_df['race_id'].astype(str).str.extract(r'(\d{12})')[0]
    race_df = race_df[race_df['race_id'].notnull()]

    race_df = race_df[~race_df['race_course'].str.contains('障', na=False)]
    logger.info('Finished loading race data. Shape: %s', race_df.shape)
    return race_df

def add_basic_features(race_df):
    # 基本的な特徴量の追加
    logger.info('Start adding basic features')
    # レース番号
    race_df['race_round'] = race_df['race_round'].str.strip('R \n').astype(int)

    # レースのグレード
    grade_mapping = {'L': 1, 'GIII': 2, 'GII': 3, 'GI': 4}
    
    race_df['race_grade'] = race_df['race_grade'].apply(
        lambda title: (
            grade_mapping.get(match.group(1), 0)
            if isinstance(title, str) and (match := re.search(r'\((GI{1,2,3}|GIII|GII|GI|L)\)', title))
            else 0
        )
    )

    # コース情報
    race_df['ground_type'] = race_df['race_course'].str.extract('(ダ|芝)', expand=False)
    race_df['distance'] = race_df['race_course'].str.extract(r'(\d+)m', expand=False).astype(int)
    race_df['distance_cat'] = ((race_df['distance'].clip(upper=2700) - 1000) // 100).astype(int)
    race_df['is_outer_course'] = race_df['race_course'].str.contains('外').astype(int)
    race_df['turn_direction'] = race_df['course_layout'].map({'左': -1, '右': +1}).fillna(0).astype(int)
    race_df['is_short_race'] = (race_df['distance'] <= 1400).astype(int)
    race_df['is_middle_race'] = ((race_df['distance'] > 1400) & (race_df['distance'] <= 2000)).astype(int)
    race_df['is_long_race'] = (race_df['distance'] > 2000).astype(int)

    # 芝・ダート
    race_df['ground_type'] = race_df['ground_type'].map({'ダ': 'dirt', '芝': 'turf'})
    race_df['is_turf'] = (race_df['ground_type'] == 'turf').astype(int)
    race_df['is_dirt'] = (race_df['ground_type'] == 'dirt').astype(int)

    # 日付と時間
    race_df['time'] = race_df['time'].str.replace(r'発走 : (\d\d):(\d\d)(.|\n)*', r'\1:\2', regex=True)
    race_df['date'] = pd.to_datetime(race_df['date'] + ' ' + race_df['time'], format='%Y年%m月%d日 %H:%M')
    race_df = race_df.rename(columns={'date': 'datetime'}).drop(columns=['time'])

    race_df['date_sin'] = np.sin(2 * np.pi * race_df['datetime'].dt.dayofyear / 365)
    race_df['date_cos'] = np.cos(2 * np.pi * race_df['datetime'].dt.dayofyear / 365)

    race_df['race_year'] = race_df['datetime'].dt.year

    
    # 開催場所
    race_df[['location', 'race_meeting_day']] = race_df['location'].str.extract(r'\d*回(..)(\d+)日目')
    race_df['race_meeting_day'] = pd.to_numeric(race_df['race_meeting_day'], errors='coerce').fillna(0).astype(int)
    location_english = {
        '札幌': 'Sapporo', '函館': 'Hakodate', '福島': 'Fukushima', '東京': 'Tokyo', '中山': 'Nakayama',
        '新潟': 'Niigata', '中京': 'Chukyo', '京都': 'Kyoto', '阪神': 'Hanshin', '小倉': 'Kokura'
    }
    race_df['location'] = race_df['location'].map(location_english)
    for value in location_english.values():
        race_df[f'is_{value}'] = (race_df['location'] == value).astype(int)

    location_order = [
        'Tokyo', 'Nakayama', 'Hanshin', 'Kyoto', 'Niigata',
        'Chukyo', 'Kokura', 'Fukushima', 'Sapporo', 'Hakodate'
    ]
    race_df['location_cat'] = (
        pd.Categorical(race_df['location'], categories=location_order, ordered=True).codes
        + race_df['is_turf'] * 10
    )

    # 天気
    race_df['weather'] = race_df['weather'].str.strip('天候 :')
    weather_mapping = {'晴': 'is_sunny', '曇': 'is_cloudy','雪': 'is_snowy', '小雪': 'is_snowy'}
    for jp, col in weather_mapping.items():
        race_df[col] = race_df['weather'].str.contains(jp, na=False).astype(int)

    weather_conditions = race_df['weather'].str.extract('(小雨|雨)', expand=False)
    rainy_map = {'小雨': 1, '雨': 2}
    race_df['rain_intensity'] = weather_conditions.map(rainy_map).fillna(0)
    race_df['weather_cat'] = (0 * race_df['is_sunny'] + 1 * race_df['is_cloudy'] + 2 * race_df['is_snowy'])
    race_df.loc[race_df['rain_intensity'] == 1, 'weather_cat'] = 3
    race_df.loc[race_df['rain_intensity'] == 2, 'weather_cat'] = 4

    # 馬場状態
    track_condition_mapping = {
        '.*(良).*': 1,
        '.*(稍重).*': 2,
        '.*(重).*': 3,
        '.*(不良).*': 4
    }
    race_df['track_condition'] = race_df['track_condition'].replace(track_condition_mapping, regex=True)
    race_df['is_bad_track'] = race_df['track_condition'].isin([2, 3, 4]).astype(int).fillna(0)

    # レース
    category_mapping = {'ハンデ': 'is_handicap', '牝': 'is_only_mare', '別定': 'is_allowance'}
    for key, value in category_mapping.items():
        race_df.loc[race_df['race_class'].str.contains(key, na=False), value] = 1

    # レースクラス
    race_class_mapping = {
        '新馬': 1, '未勝利': 2, '1勝クラス': 3, '500万下': 3,
        '2勝クラス': 4, '1000万下': 4, '3勝': 5, '1600万下': 5, 'オープン': 6, '2歳オープン': 3,
    }
    
    for key, value in race_class_mapping.items():
        race_df.loc[race_df['race_class'].str.contains(key, na=False), 'race_class_level'] = value
    race_df.loc[race_df['race_grade'] > 0, 'race_class_level'] = 7
    race_df['is_maiden'] = (race_df['race_class_level'] == 1).astype(int)

    # レースの年齢
    race_age_mapping = race_age_mapping = {'2歳':2, '3歳':3, '4歳':4}
    for key, value in race_age_mapping.items():
        race_df.loc[race_df['race_class'].str.contains(key, na=False), 'race_age'] = value
    race_df['race_age'] = race_df['race_age'].fillna(0)
    
    # 出走頭数
    race_df['field_size_bin'] = np.select(
        [race_df['total_horse_number'] <= 10, race_df['total_horse_number'] <= 14],
        ['small', 'medium'],
        default='large'
    )
    race_df['is_small_field_size']  = (race_df['field_size_bin'] == 'small').astype(int)
    race_df['is_medium_field_size'] = (race_df['field_size_bin'] == 'medium').astype(int)
    race_df['is_large_field_size']  = (race_df['field_size_bin'] == 'large').astype(int)
    race_df['field_size_cat'] = race_df['is_medium_field_size'] * 1 + race_df['is_large_field_size'] * 2


    # 馬場指数
    race_df['ground_index'] = race_df['ground_index'].astype(int)
    race_df['norm_tough_gi'] = -race_df['ground_index'].clip(-20, 20) / 20

    logger.info('Finished basic features. Columns: %s', list(race_df.columns))
    return race_df

def add_course_features(race_df):
    # コースに関する特徴量の追加
    logger.info('Start adding course features')

    # コース特徴量
    new_columns = ['start_slope', 'after_1f_slope', 'start_stretch_length', 'corners_num',
        'big_turn', 'tight_turn', 'last_stretch_length', 'slope_before_goal']

    # 芝コース
    turf_outer_by_loc = {'Niigata': 659.0, 'Kyoto': 404.0, 'Hanshin': 473.6}
    turf_outer_by_dist = {('Niigata', 2000) : [0, 0.5, 960, 2],('Kyoto',  1600): [0, 0, 730, 2]}

    for loc, values in turf_by_loc.items():
        race_df.loc[(race_df['ground_type'] == 'turf') & (race_df['location'] == loc), new_columns[4:]] = values
    for loc, values in turf_outer_by_loc.items():
        race_df.loc[(race_df['ground_type'] == 'turf') & (race_df['is_outer_course'] == 1) & (race_df['location'] == loc), [new_columns[6]]] = values

    for loc, dist_dict in turf_by_dist.items():
        for dist, values in dist_dict.items():
            race_df.loc[(race_df['is_turf'] == 1) & (race_df['location'] == loc) & (race_df['distance'] == dist), new_columns[:4]] = values
    for keys, values in turf_outer_by_dist.items():
        race_df.loc[(race_df['ground_type'] == 'turf') & (race_df['is_outer_course'] == 1) & (race_df['location'] == keys[0]) & (race_df['distance'] == keys[1]), new_columns[:4]] = values

    # ダートコース
    for loc, values in dirt_by_loc.items():
        race_df.loc[(race_df['ground_type'] == 'dirt') & (race_df['location'] == loc), new_columns[4:]] = values

    for loc, dist_dict in dirt_by_dist.items():
        for dist, values in dist_dict.items():
            race_df.loc[(race_df['ground_type'] == 'dirt') & (race_df['location'] == loc) & (race_df['distance'] == dist), new_columns[:4]] = values

    # 追加の特徴量
    race_df['corners_density'] = race_df['corners_num'] / race_df['distance']
    race_df['stretch_difficulty'] = race_df['last_stretch_length'] / race_df['distance']
    return race_df

def build_lap_time_features(lap_list):
    # ラップタイムの特徴量を計算
    if not lap_list or not isinstance(lap_list, list):
        return {
            'race_lap_time_mean': np.nan,
            'race_lap_time_std': np.nan,
            'race_lap_time_first_half': np.nan,
            'race_lap_time_last_half': np.nan,
            'race_lap_early_vs_late': np.nan,
            'early_last_half_pct_diff': np.nan,
            'lap_acceleration_mean': np.nan,
            'lap_acceleration_std': np.nan,
            'race_lap_count': 0
        }

    lap_array = np.array(lap_list, dtype=float)
    lap_count = len(lap_array)
    race_lap_time_mean = np.mean(lap_array)
    race_lap_time_std = np.std(lap_array)

    race_lap_time_first_half = np.mean(lap_array[:3])
    race_lap_time_last_half = np.mean(lap_array[-3:])
    race_lap_early_vs_late = race_lap_time_first_half - race_lap_time_last_half
    first_last_half_pct_diff = race_lap_early_vs_late / race_lap_time_mean

    lap_diffs = np.diff(lap_array)
    lap_acceleration_mean = np.mean(lap_diffs)
    lap_acceleration_std = np.std(lap_diffs)

    return {
        'race_lap_time_mean': race_lap_time_mean,
        'race_lap_time_std': race_lap_time_std,
        'race_lap_time_first_half': race_lap_time_first_half,
        'race_lap_time_last_half': race_lap_time_last_half,
        'race_lap_early_vs_late': race_lap_early_vs_late,
        'first_last_half_pct_diff': first_last_half_pct_diff,
        'lap_acceleration_mean': lap_acceleration_mean,
        'lap_acceleration_std': lap_acceleration_std,
        'race_lap_count': lap_count
    }

def add_lap_time_features(df, lap_col='lap_time'):
    # ラップタイム特徴量の追加
    logger.info('Start adding lap time features')
    lap_list_series = df[lap_col].str.split(' - ').map(lambda x: list(map(float, x)) if x else [])
    features_df = lap_list_series.apply(build_lap_time_features).apply(pd.Series)
    logger.info('Finished adding lap time features. New columns: %s', list(features_df.columns))
    return pd.concat([df, features_df], axis=1)


def compute_time_decay_group_stats(df, group_cols, target_col, decay_rate=0.95):
    # 時間減衰を考慮したグループ統計量の計算
    stats = []
    years = sorted(df['race_year'].unique())
    
    for year in years:
        df_train = df[df['race_year'] < year]
        df_eval = df[df['race_year'] == year]

        if df_train.empty or df_eval.empty:
            continue

        df_train = df_train.copy()
        df_train['weight'] = decay_rate ** (year - df_train['race_year'])

        group_stat = (
            df_train.groupby(group_cols)
            .apply(lambda g: np.average(g[target_col], weights=g['weight']))
            .rename('group_' + target_col)
        )

        df_eval = df_eval.merge(group_stat, left_on=group_cols, right_index=True, how='left')
        stats.append(df_eval)

    df_stats = pd.concat(stats, axis=0)
    return df_stats


def apply_time_group_statistics(df):
    # グループ統計量の計算
    logger.info('Start applying time decay group statistics')
    group_cols = ['location', 'ground_type', 'distance', 'is_outer_course', 'race_class_level', 'race_age']
    
    df = df.sort_values('datetime').copy()
    
    df_with_stats = compute_time_decay_group_stats(df, group_cols, 'race_lap_time_mean')
    df_with_stats_std = compute_time_decay_group_stats(df, group_cols, 'race_lap_time_std')
    df_with_early_vs_late = compute_time_decay_group_stats(df, group_cols, 'race_lap_early_vs_late')

    df = df.merge(df_with_stats[['race_id', 'group_race_lap_time_mean']], on='race_id', how='left')
    df = df.merge(df_with_stats_std[['race_id', 'group_race_lap_time_std']], on='race_id', how='left')
    df = df.merge(df_with_early_vs_late[['race_id', 'group_early_vs_late_mean']], on='race_id', how='left')

    df['race_lap_time_mean_zscore'] = -(
        (df['race_lap_time_mean'] - df['group_race_lap_time_mean']) / 
        (df['group_race_lap_time_std'] + 1e-6)
    ).fillna(0)

    df['race_lap_early_vs_late_relative'] = (
        df['race_lap_early_vs_late'] - df['group_early_vs_late_mean']
    ).fillna(0)
    
    df = df.drop(columns=[
        'race_lap_time_mean',
        'race_lap_time_std',
        'race_lap_time_first_half',
        'race_lap_time_last_half',
        'group_race_lap_time_std',
        'group_race_lap_time_mean',
        'group_race_lap_time_mean_std',
        'group_early_vs_late_mean'
    ])
    logger.info('Finished applying time decay group statistics. Shape: %s', df.shape)
    return df

def finalize_and_save(race_df):
    # 最終的なデータフレームの整形と保存
    logger.info('Finalizing and saving')
    race_df = race_df[race_feature_columns]
    logger.info('Columns selected: %d columns, final shape: %s', race_df.shape[1], race_df.shape)
    
    save_path = SAVE_DIR / 'race_features.parquet'
    race_df.to_parquet(save_path, index=False)
    logger.info('Saved full race features to %s', save_path)

    last_year = datetime.now().year - 1
    mask = (race_df['race_id'].astype(int) < (last_year+1) * 100_000_000) & (race_df['race_id'].astype(int) > last_year * 100_000_000) 
    race_df_last_year = race_df[mask]
    csv_path = CSV_DIR / f'race_features_{last_year}.csv'
    race_df_last_year.to_csv(csv_path, encoding='utf-8', index=False)
    logger.info('Saved %d races from year %d to %s', len(race_df_last_year), last_year, csv_path)

    sample_path = CSV_DIR / 'race_features_sampled.csv'
    race_df.sample(n=500, random_state=42).to_csv(sample_path, encoding='utf-8', index=False)
    logger.info('Saved 500 sampled races to %s', sample_path)

    

if __name__ == '__main__':
    process_race_data()