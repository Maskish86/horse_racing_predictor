from pathlib import Path
from datetime import datetime
import glob
import pandas as pd
import pytz
import sys
import numpy as np

from utils.logger import setup_logger
logger = setup_logger('horse_feature_builder')

now_datetime = datetime.now(pytz.timezone('Asia/Tokyo'))
PARQUET_DIR = Path('data/processed_parquet')
PREDICT_FEATURE_DIR = Path('data/predict_features')
SAVE_DIR = Path('data/train_data')
CSV_DIR = Path('data/csv')

def build_mlp_dataset():
    df = load_and_merge_data()
    df = add_categorical_columns(df)
    df = fill_2yro_mean_values(df)
    df = convert_dtypes(df)
    df = make_lagged_features(df, n_shifts=5)
    finalize_and_save(df)




def load_and_merge_data():    
    race_df = pd.read_parquet(PARQUET_DIR / 'race_feaure.parquet')
    horse_df = pd.read_parquet(PARQUET_DIR / 'horse_feature.parquet')
    horse_training_df = pd.read_parquet(PARQUET_DIR / 'training_feature.parquet')
    race_col_to_drop = ['race_round', 'datetime', 'is_long_race',  'is_middle_race',  'is_short_race',
                        'is_sunny',  'is_cloudy',  'is_snowy', 'rain_intensity', 'is_bad_track', 'race_lap_count',
                        'is_Sapporo',  'is_Hakodate',  'is_Fukushima',  'is_Tokyo',  'is_Nakayama',  'is_Niigata',  'is_Chukyo',  'is_Kyoto',  'is_Hanshin',  'is_Kokura',
                        'is_small_field_size',  'is_medium_field_size',  'is_large_field_size', 'time_decay_weight']
    horse_col_to_drop = ['ground_type', 'distance',  'is_outer_course', 'race_class_level',  'total_horse_number', 'field_size_bin', 'corners_num', 'frame_number', 'norm_tough_gi',  'is_inner',  'is_middle',  'is_outer', 'rank_path_count', 'rider_id', 'is_trouble', 'is_slow_break']
    training_col_to_drop = ['tlap_time_3f', 'tlap_time_1f', 'tsec_time_mean',   'tsec_time_stdlog', 'tsec_early_vs_late', 'course_type_wood',  'course_type_slope',  'course_type_poly',  'course_type_dirt',  'course_type_turf', 'top_lap_time_1',  'top_lap_time_2',  'top_lap_time_3',  'top_lap_time_4',  'top_lap_time_5']
    race_df = race_df.drop(race_col_to_drop, axis=1)
    horse_df = horse_df.drop(horse_col_to_drop, axis=1)
    horse_training_df = horse_training_df.drop(training_col_to_drop, axis=1)

    merged_df = pd.merge(horse_df, race_df, on='race_id')
    merged_df = pd.merge(merged_df, horse_training_df, on=['race_id', 'horse_id'])
    merged_df['datetime'] = pd.to_datetime(merged_df['datetime'], errors='coerce')
    return merged_df

def add_categorical_columns(merged_df):

    def compress_number(n):
        return int(n) // 2 if pd.notna(n) and n < 18 else np.nan


    def compress_distdiff_number(n):
        bins = [2, 3, 5, 7, 8, 10, 12, 14, 16]
        if pd.isna(n) or n < 0:
            return -1
        for i, b in enumerate(bins):
            if n <= b:
                return i
        return np.nan



    merged_df['loc_dist_cat'] = (merged_df['location_cat'] + merged_df['is_outer_course'])* 18 + merged_df['distance_cat']
    merged_df = merged_df.drop('is_outer_course', axis=1)

    merged_df['loc_dist_horsenum_cat'] = merged_df['loc_dist_cat'] * 10 + merged_df['horse_number_cat'].apply(compress_number)
    merged_df['loc_dist_field_cat'] = merged_df['loc_dist_cat'] * 10 + merged_df['field_size_cat']
    merged_df['loc_dist_distdiff_cat'] = merged_df['loc_dist_cat'] * 10 + merged_df['distance_diff_cat'].apply(compress_distdiff_number)
    merged_df['loc_dist_level_cat'] = merged_df['loc_dist_cat'] * 10 + merged_df['race_class_level']

    merged_df['loc_dist_epos_cat'] = merged_df['loc_dist_cat'] * 10 + merged_df['early_pos_cat'].apply(compress_number)
    merged_df['loc_dist_lpos_cat'] = merged_df['loc_dist_cat'] * 10 + merged_df['late_pos_cat'].apply(compress_number)
    merged_df['loc_dist_rank_cat'] = merged_df['loc_dist_cat'] * 10 + merged_df['rank_cat'].apply(compress_number)

    merged_df['horsenum_epos_cat'] = merged_df['horse_number_cat'] * 20 + merged_df['early_pos_cat']
    merged_df['epos_lpos_cat'] = merged_df['early_pos_cat'] * 20 + merged_df['late_pos_cat']
    merged_df['epos_lpos_horsenum_cat'] = merged_df['early_pos_cat'].apply(compress_number) * 100 + merged_df['late_pos_cat'].apply(compress_number) * 10 + merged_df['horse_number_cat'].apply(compress_number)
    merged_df['epos_lpos_rank_cat'] = merged_df['early_pos_cat'].apply(compress_number) * 100 + merged_df['late_pos_cat'].apply(compress_number) * 10 + merged_df['rank_cat'].apply(compress_number)

    merged_df['is_insufficient_history'] = 0
    return merged_df


def fill_2yro_mean_values(merged_df):
    cols_fill_2yo = [
        'speed_index_std5', 'speed_index_momentum5',
        'ground_pref_rank', 'tough_ground_pref_rank',
        'ground_pref_time', 'tough_ground_pref_time',
        'pos_volatility_index', 'pos_diff_stdlog',
        'tsec_acceleration_mean', 'tsec_acceleration_stdlog',
        'horse_weight'
    ]

    year_last = datetime.now().year - 1

    means_2yo_by_year = (
        merged_df[merged_df['age'] == 2]
        .replace(0, np.nan)
        .groupby('race_year')[cols_fill_2yo]
        .mean()
    )

    means_2yo_last = means_2yo_by_year.loc[year_last].to_dict()

    df_means_2yo_last = (
        pd.DataFrame.from_dict(means_2yo_last, orient='index', columns=['fill_value'])
        .reset_index()
        .rename(columns={'index': 'column'})
    )
    df_means_2yo_last.to_csv(PREDICT_FEATURE_DIR / 'means_2yo_last.csv', index=False)
    for col in cols_fill_2yo:
        merged_df[col] = merged_df[col].fillna(means_2yo_last.get(col, np.nan))
    return merged_df



def convert_dtypes(merged_df):    
    cat_cols = [col for col in merged_df.columns if col.endswith('_cat')]
    merged_df[cat_cols] = merged_df[cat_cols].fillna(-1)
    merged_df[cat_cols] = merged_df[cat_cols].astype('int32')
    cont_cols = [c for c in merged_df.columns if c not in cat_cols + ['race_id', 'horse_id', 'datetime', 'location']]
    merged_df[cont_cols] = merged_df[cont_cols].astype('float32')
    return merged_df

def make_lagged_features(df, n_shifts=5):
    df = df.sort_values(['horse_id', 'datetime'], ascending=[True, False])
    result_df = df.copy()
    shifted_feature_cols = [
        'speed_index',
        'log_win_odds',
        'implied_proba_logit',
        'norm_popularity',
        'rank_vs_popular',
        'goal_time_zscore_mean',
        'last3f_time_zscore_mean',
        'goal_time_zscore_winner',
        'last3f_time_zscore_winner',
        'winner_goal_time_zscore_vs_courseclass',
        'winner_last3f_time_zscore_vs_courseclass',
        'top_last3f_time_zscore_vs_courseclass',
        'early_pos5_win_kde',
        'early_pos5_show_kde',
        'late_pos5_win_kde',
        'late_pos_show_kde',
    ]

    for i in range(1, n_shifts + 1):
        shifted = df.groupby('horse_id').shift(-i)
        shifted.columns = [f'{col}-{i}' for col in shifted_feature_cols]
        result_df = pd.concat([result_df, shifted], axis=1)

    return result_df

def finalize_and_save(train_df):
    # 最終的なデータフレームの整形と保存
    logger.info('Finalizing and saving')
    
    train_df = train_df.dropna(subset=['rank'])

    train_df['is_win'] = (train_df['rank'] <= 1.0) * 1
    train_df['is_place'] = (train_df['rank'] <= 2.0) * 1
    train_df['is_show'] = (train_df['rank'] <= 3.0) * 1

    train_df = train_df.drop(
        ['rank','norm_rank',  'rank_cat', 'horse_id','horse_weight_diff','speed_index',
        'log_win_odds', 'implied_proba_logit',    'norm_popularity',  'rank_vs_popular',
        'goal_time_zscore_mean',    'last3f_time_zscore_mean', 'goal_time_zscore_winner',  'last3f_time_zscore_winner', 'winner_goal_time_zscore_vs_courseclass',
        'winner_last3f_time_zscore_vs_courseclass', 'top_last3f_time_zscore_vs_courseclass', 'pos_volatility_index',  'pos_diff_mean',  'pos_diff_stdlog',  'norm_early_pos',  'norm_late_pos',
        'pos_gain_ratio',  'pos_trend',  'pos_trend_per_corner', 'early_pos_cat',  'late_pos_cat',
        'early_pos5_win_kde', 'early_pos5_show_kde','late_pos5_win_kde', 'late_pos_show_kde',
        'first_last_half_pct_diff',  'lap_acceleration_mean',  'lap_acceleration_std', 'norm_tough_gi',
        'race_lap_time_mean_zscore',  'race_lap_time_std_relative',  'race_lap_early_vs_late_relative',
        'loc_dist_epos_cat','loc_dist_rank_cat', 'horsenum_epos_cat',  'epos_lpos_cat',  'epos_lpos_horsenum_cat',  'epos_lpos_rank_cat' ], axis=1)
    
    
    last_year = datetime.now().year - 1 
    mask = (train_df['race_id'].astype(int) < (last_year+1) * 100_000_000) & (train_df['race_id'].astype(int) > 2009 * 100_000_000)
    train_df = train_df[mask]
    logger.info('Columns selected: %d columns, final shape: %s', train_df.shape[1], train_df.shape)
    
    save_path = SAVE_DIR / 'mlp_train_dataset.parquet'
    train_df.to_parquet(save_path, index=False)
    logger.info('Saved full horse features to %s', save_path)

    
    last_year_mask = (train_df['race_id'].astype(int) < (last_year+1) * 100_000_000) & (train_df['race_id'].astype(int) > last_year * 100_000_000) 
    train_df_last_year = train_df[last_year_mask]
    csv_path = CSV_DIR / f'mlp_train_dataset_{last_year}.csv'
    train_df_last_year.to_csv(csv_path, encoding='utf-8', index=False)
    logger.info('Saved %d rows from year %d to %s', len(train_df_last_year), last_year, csv_path)

    sample_path = CSV_DIR / 'mlp_train_dataset_sampled.csv'
    train_df.sample(n=500, random_state=42).to_csv(sample_path, encoding='utf-8', index=False)
    logger.info('Saved 500 sampled train rows to %s', sample_path)

    

