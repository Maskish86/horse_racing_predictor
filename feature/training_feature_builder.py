from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from metadata.columns_feature import training_feature_columns

from utils.logger import setup_logger
logger = setup_logger('horse_feature_builder')

YEARLY_TRAINING_DIR = Path('data/yearly_parquet/training')
SAVE_DIR = Path('data/processed_parquet')
PREDICT_FEATURE_DIR = Path('data/predict_features')
CSV_DIR = Path('data/csv')

def process_training_data():
    df = load_and_clean_data()
    df = compute_training_section_times(df)
    df = compute_tsec_features(df)
    df = set_min_group_count(df, threshold=30)
    df = add_group_time_features(df)
    df = add_past_time_features(df)
    df = course_and_style_mapping(df)
    finalize_and_save(df)

def load_and_clean_data():
    # データの読み込みと基本的な処理
    training_files = list(YEARLY_TRAINING_DIR.glob('training_*.parquet'))
    df = pd.concat([pd.read_parquet(f) for f in training_files], axis=0)
    race_df_for_merge = pd.read_parquet(SAVE_DIR / 'race_features.parquet')
    df['race_id'] = df['race_id'].astype(str).str.extract(r'(\d{12})')
    df = df.dropna(subset=['race_id'])
    df['race_id'] = df['race_id'].astype(int)
    df = df.merge(race_df_for_merge[['race_id', 'datetime', 'race_class_level']], on='race_id', how='left')
    df = df.dropna(subset=['datetime'])
    course_type_correction_mapping = {'北Ｂ':'栗Ｂ','美Ｗ': 'ＢＷ', '南Ｗ': 'ＢＷ', '美ポ':'美Ｐ', '南Ｐ': '美Ｐ', '南ポ': '美Ｐ', '美北':'北Ｃ', '美ダ': '南ダ'}
    df['course_type'] = df['course_type'].replace(course_type_correction_mapping)
    df.loc[(df['race_id'].astype(int) < 202000000000) & (df['course_type'] == '南Ｂ'), 'course_type'] = 'ＢＷ'
    df.loc[(df['race_id'].astype(int) < 202000000000) & (df['course_type'] == '南ダ'), 'course_type'] = '南Ｄ'
    df.loc[(df['race_id'].astype(int) > 202000000000) & (df['course_type'] == '南ダ'), 'course_type'] = '南Ｂ'
    df.loc[(df['race_id'].astype(int) > 202000000000) & (df['course_type'] == 'ＢＷ'), 'course_type'] = 'ＤＷ'
    df.loc[(df['race_id'].astype(int) > 202000000000) & (df['course_type'] == '南Ｄ'), 'course_type'] = 'ＤＷ'
    running_style_correction_mapping = {'Ｇ強': '仕掛', 'Ｇ一' : '仕掛', '直一': '仕掛', '直強':'仕掛'}
    df['running_style'] = df['running_style'].replace(running_style_correction_mapping)
    return df


def compute_training_section_times(df):
    # ラップタイムから区間タイムを計算
    lap_cols = [f'lap_time_{i}' for i in range(1, 6)]
    clip_range = [(13.5, 20.5), (12.5, 17.5), (11.5, 17), (11.5, 15.5), (11.5, 15.5), (11, 15.5) ]
    small_course_types = df['course_type'].value_counts()
    small_course_types = small_course_types[small_course_types < 500].index
    df.loc[df['course_type'].isin(small_course_types), 'course_type'] = np.nan
    df['course_type'] = df['course_type'].replace('栗飛', np.nan)

    df[lap_cols] = df[lap_cols].apply(pd.to_numeric, errors='coerce').replace(0, np.nan)
    df.loc[df[lap_cols].notna().any(axis=1) & df['running_style'].isna(), 'running_style'] = '馬也'
    mean_times = df.groupby(['course_type', 'running_style'])[lap_cols[2:]].mean()

    df_filled = df.copy()
    group_keys = df[['course_type', 'running_style']].values

    for lap_col in lap_cols[2:]:
        nan_counts = df_filled[lap_cols[2:]].isna().sum(axis=1)
        single_nan_mask = (df_filled[lap_col].isna()) & (nan_counts == 1)

        mean_values = np.array([
            mean_times.loc[(course_type, running_style), lap_col]
            if (course_type, running_style) in mean_times.index
            else np.nan
            for course_type, running_style in group_keys[single_nan_mask]
        ])

        df_filled.loc[single_nan_mask, lap_col] = mean_values

    is_hill_course = df['course_type'].str.contains('坂', na=False)
    for i in range(1, 7):
        tsec_col = f'tsec_time_{7 - i}f'

        if (i > 1) and (i < 6):
            df.loc[is_hill_course, tsec_col] = (
                df.loc[is_hill_course, f'lap_time_{i - 1}'] -
                df.loc[is_hill_course, f'lap_time_{i}']
            )
        elif i == 6:  
            df.loc[is_hill_course, tsec_col] = df.loc[is_hill_course, 'lap_time_5']

        if i == 1:  # 6F→5F
            df.loc[~is_hill_course, tsec_col] = df['lap_time_1'] - df['lap_time_2']
        elif i == 2:  # 5F→4F
            df.loc[~is_hill_course, tsec_col] = df['lap_time_2'] - df['lap_time_3']
        elif i == 3:  # 4F→3F
            df.loc[~is_hill_course, tsec_col] = df['lap_time_3'] - df['lap_time_4']
        elif i == 4:  # 3F→2F
            df.loc[~is_hill_course, tsec_col] = (df['lap_time_4'] - df['lap_time_5']) / 2
        elif i == 5:  # 2F→1F
            df.loc[~is_hill_course, tsec_col] = (df['lap_time_4'] - df['lap_time_5']) / 2
        elif i == 6:  # 1F→Finish
            df.loc[~is_hill_course, tsec_col] = df['lap_time_5']
        if i <= 5:
            df.loc[df[f'lap_time_{i}'].isna(), tsec_col] = np.nan
        df.loc[df[tsec_col] < 8, tsec_col] = np.nan
        df[tsec_col] = df[tsec_col].clip(*clip_range[i-1])
    df = df.drop(columns=lap_cols)
    return df


def compute_tsec_features(df):
    # 区間タイムから特徴量を計算
    tsec_col = [f'tsec_time_{i}f' for i in range(6, 0, -1)]
    tsec_values = df[tsec_col].values
    rightmost_nan_index = np.where(np.isnan(tsec_values[:, ::-1]), np.arange(6), -1).max(axis=1)

    for i, row in enumerate(tsec_values):
        if rightmost_nan_index[i] != -1:  # -1 means no NaN found
            nan_cutoff = 6 - rightmost_nan_index[i]
            row[:nan_cutoff] = np.nan


    df[tsec_col] = tsec_values
    df['valid_tsec_count'] = df[tsec_col].notna().sum(axis=1)
    df['tlap_time_3f'] = df[tsec_col[-3:]].sum(axis=1)
    df['tlap_time_1f'] = df[tsec_col[-1]]
    df['tsec_time_mean'] = df[tsec_col].mean(axis=1)
    df['tsec_time_stdlog'] = np.log1p(df[tsec_col].std(axis=1))

    tsec_time_last_half = df[tsec_col[-3:]].mean(axis=1)
    tsec_values = df[tsec_col].values

    half_counts = (df['valid_tsec_count'] // 2).values
    tsec_time_first_half = np.array([
        np.nanmean(row[:count]) if count > 0 else np.nan
        for row, count in zip(tsec_values, half_counts)
    ])

    df['tsec_early_vs_late'] = tsec_time_first_half - tsec_time_last_half
    df['first_last_tsec_half_pct_diff'] = (df['tsec_early_vs_late'] / df['tsec_time_mean']).fillna(0)

    tsec_diffs = df[tsec_col].diff(axis=1)
    df['tsec_acceleration_mean'] = tsec_diffs.mean(axis=1)
    df['tsec_acceleration_stdlog'] = np.log1p(tsec_diffs.std(axis=1))
    df['tsec_acceleration_mean'] = df.groupby('race_id')['tsec_acceleration_mean'].transform(lambda x: x.fillna(x.mean()))
    df['tsec_acceleration_stdlog'] = df.groupby('race_id')['tsec_acceleration_stdlog'].transform(lambda x: x.fillna(x.mean()))

    df = df.drop(columns=tsec_col)
    return df


def set_min_group_count(df, threshold=30):
    # グループごとのデータ数が閾値未満の場合の処理
    group_cols = ['course_type', 'running_style', 'is_high_level']
    df['valid_tsec_count_for_group'] = df['valid_tsec_count'].replace(0, np.nan)
    df['is_high_level'] = (df['race_class_level'] > 2).astype(int)

    group_counts = df.groupby(group_cols).size().reset_index(name='group_size')

    group_counts['is_level_down'] = 0
    group_counts['is_style_change'] = 0

    high_level_mask = (group_counts['is_high_level'] == 1) & (group_counts['group_size'] < threshold)
    if high_level_mask.any():
        df.loc[
            df.set_index(group_cols).index.isin(group_counts.loc[high_level_mask, group_cols].set_index(group_cols).index),
            'is_high_level'
        ] = 0
        group_counts.loc[high_level_mask, 'is_level_down'] = 1

    for course_type in group_counts['course_type'].unique():
        full_effort_mask = (group_counts['course_type'] == course_type) & (group_counts['running_style'] == '一杯')
        strong_effort_mask = (group_counts['course_type'] == course_type) & (group_counts['running_style'] == '強め')

        full_effort_count = group_counts.loc[full_effort_mask, 'group_size'].sum()
        strong_effort_count = group_counts.loc[strong_effort_mask, 'group_size'].sum()

        if full_effort_count < threshold and strong_effort_count >= threshold:
            df.loc[
                (df['course_type'] == course_type) & (df['running_style'] == '一杯'),
                'running_style'
            ] = '強め'
            group_counts.loc[full_effort_mask, 'is_style_change'] = 1

        if strong_effort_count < threshold and full_effort_count >= threshold:
            df.loc[
                (df['course_type'] == course_type) & (df['running_style'] == '強め'),
                'running_style'
            ] = '一杯'
            group_counts.loc[strong_effort_mask, 'is_style_change'] = 1

    preserved_flags = group_counts[['course_type', 'running_style', 'is_high_level', 'is_level_down', 'is_style_change']]

    group_counts = df.groupby(group_cols).size().reset_index(name='group_size')

    group_counts = group_counts.merge(preserved_flags, on=group_cols, how='left').fillna(0)

    low_level_mask = (group_counts['is_high_level'] == 0) & (group_counts['group_size'] < threshold)
    if low_level_mask.any():
        df.loc[
            df.set_index(group_cols).index.isin(group_counts.loc[low_level_mask, group_cols].set_index(group_cols).index),
            'valid_tsec_count_for_group'
        ] = np.nan

    filtered_groups = group_counts[
        (group_counts['is_level_down'] == 1) |
        (group_counts['is_style_change'] == 1)
    ]

    filtered_groups.to_csv(PREDICT_FEATURE_DIR / 'group_level_map_filtered.csv', index=False)


    return df


def add_group_time_features(df):
    # グループごとの時間特徴量を追加
    group_col = ['course_type', 'running_style', 'valid_tsec_count_for_group', 'is_high_level']

    group_stats = df.groupby(group_col).agg({
        'tlap_time_3f': ['mean', 'std'],
        'tlap_time_1f': ['mean', 'std'],
        'tsec_time_mean': ['mean', 'std'],
        'tsec_time_stdlog': 'mean',
        'tsec_early_vs_late': 'mean'
    }).reset_index()

    group_stats.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col for col in group_stats.columns
    ]

    df = df.merge(group_stats, on=group_col, how='left')

    df['tlap_time_3f_zscore'] =  - ((df['tlap_time_3f'] - df['tlap_time_3f_mean']) / df['tlap_time_3f_std']).fillna(0)
    df['tlap_time_1f_zscore'] =  - ((df['tlap_time_1f'] - df['tlap_time_1f_mean']) / df['tlap_time_1f_std']).fillna(0)
    df['tsec_time_mean_zscore'] = - ((df['tsec_time_mean'] - df['tsec_time_mean_mean']) / df['tsec_time_mean_std']).fillna(0)
    df['tsec_time_stdlog_relative'] = (df['tsec_time_stdlog'] - df['tsec_time_stdlog_mean']).fillna(0)
    df['tsec_early_vs_late_relative'] = (df['tsec_early_vs_late'] - df['tsec_early_vs_late_mean']).fillna(0)
    col_to_save = ['tlap_time_3f_mean','tlap_time_3f_std', 'tlap_time_1f_mean', 'tlap_time_1f_std', 'tsec_time_mean_mean', 'tsec_time_mean_std', 'tsec_time_stdlog_mean', 'tsec_early_vs_late_mean']
    df[col_to_save+group_col].drop_duplicates().to_csv( 'group_tsec_features.csv', index=False)
    df = df.drop(columns=col_to_save+['valid_tsec_count_for_group', 'is_high_level'])
    return df


def add_past_time_features(df):
    mean_col = ['tlap_time_3f', 'tlap_time_1f', 'tsec_time_mean', 'tsec_time_stdlog', 'tsec_early_vs_late']
    std_col = ['tlap_time_3f', 'tlap_time_1f', 'tsec_time_mean']
    group_col = ['horse_id', 'course_type', 'valid_tsec_count']

    df = df.sort_values(['horse_id', 'course_type', 'valid_tsec_count', 'datetime'])
    df[group_col+mean_col+['datetime']].to_csv(PREDICT_FEATURE_DIR / 'tsec_horse_df.csv', index=False)

    df = df.sort_values(group_col + ['datetime'])

    past5_features = []

    for _, group in df.groupby(group_col):
        shifted_group = group.copy()
        shifted_group[mean_col] = shifted_group[mean_col].shift(1)

        shifted_group['tsec_past5_count'] = shifted_group['tlap_time_1f'].rolling(window=5, min_periods=2).count().fillna(0)

        for col in mean_col:
            shifted_group[f'{col}_avg5'] = shifted_group[col].rolling(window=5, min_periods=2).mean()

        for col in std_col:
            shifted_group[f'{col}_std5'] = shifted_group[col].rolling(window=5, min_periods=2).std()

        past5_features.append(shifted_group)

    past5_df = pd.concat(past5_features)

    df = df.merge(past5_df[group_col + [f'{col}_avg5' for col in mean_col] + [f'{col}_std5' for col in std_col] + ['tsec_past5_count']],
                            on=group_col,
                            how='left')
    df['tsec_past5_count'] = df['tsec_past5_count'].fillna(0)
    df['tlap_time_3f_zscore5'] = - ((df['tlap_time_3f'] - df['tlap_time_3f_avg5']) / df['tlap_time_3f_std5']).clip(-4, 4).fillna(0)
    df['tlap_time_1f_zscore5'] = - ((df['tlap_time_1f'] - df['tlap_time_1f_avg5']) / df['tlap_time_1f_std5']).clip(-3, 3).fillna(0)
    df['tsec_time_mean_zscore5'] = - ((df['tsec_time_mean'] - df['tsec_time_mean_avg5']) / df['tsec_time_mean_std5']).clip(-4, 4).fillna(0)
    df['tsec_time_stdlog_relative5'] = (df['tsec_time_stdlog'] - df['tsec_time_stdlog_avg5']).fillna(0)
    df['tsec_early_vs_late_relative5'] = (df['tsec_early_vs_late'] - df['tsec_early_vs_late_avg5']).fillna(0)
    col_to_drop =[f'{col}_avg5' for col in mean_col] + [f'{col}_std5' for col in std_col]
    df = df.drop(columns=col_to_drop)
    return df

def course_and_style_mapping(df):
    # コースタイプと走法のマッピング
    course_order = ['栗坂', '美坂', 'ＤＷ', 'ＣＷ', 'ＢＷ', '美Ｐ',  '北Ｃ', '函Ｗ', 'ＤＰ', '南Ｄ', '栗Ｂ', '札ダ', '小ダ', '函ダ', '栗芝', '南芝', '南Ｂ', '札芝', '栗Ｅ', '函芝']
    df['course_cat'] = pd.Categorical(df['course_type'], categories=course_order, ordered=True).codes
    df.insert(df.columns.get_loc('course_type') + 1, 'course_cat', df.pop('course_cat'))
    course_type_mapping = {
            'ＣＷ': 'wood', '栗坂': 'slope', 'ＤＰ':'poly', '栗Ｂ':'dirt', '栗芝': 'turf',
            'ＢＷ': 'wood', 'ＤＷ': 'wood','美坂': 'slope', '美Ｐ': 'poly','北Ｃ': 'dirt', '南芝': 'turf',
            }
    for key, value in course_type_mapping.items():
        if f'course_type_{value}' not in df.columns:
            df.insert(df.columns.get_loc('course_type'), f'course_type_{value}', 0)
        df.loc[df['course_type'] == key, f'course_type_{value}'] = 1


    course_condition_mapping = {'良': 1, '稍': 2, '重': 3, '不良': 4}
    df['course_condition'] = df['course_condition'].map(course_condition_mapping).fillna(1)

    running_style_mapping = { '馬也': 'own_pace', '一杯': 'full_effort', '強め': 'strong_effort', '仕掛': 'urged'}
    for key, value in running_style_mapping.items():
        df[value] = (df['running_style'] == key).astype(int)
    df['running_style_cat'] = pd.Categorical(df['running_style'], categories=list(running_style_mapping.keys()), ordered=True).codes
    evaluation_grade_mapping = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
    df['evaluation_grade'] = df['evaluation_grade'].map(evaluation_grade_mapping).fillna(2)

def finalize_and_save(df):
    # 最終的なデータフレームの整形と保存
    logger.info('Finalizing and saving')
    df = df[training_feature_columns]
    logger.info('Columns selected: %d columns, final shape: %s', df.shape[1], df.shape)

    save_path = SAVE_DIR / 'training_features.parquet'
    df.to_parquet(save_path, index=False)
    logger.info('Saved full training features to %s', save_path)

    last_year = datetime.now().year - 1
    mask = (df['race_id'].astype(int) < (last_year+1) * 100_000_000) & (df['race_id'].astype(int) > last_year * 100_000_000) 
    df_last_year = df[mask]
    csv_path = CSV_DIR / f'training_features_{last_year}.csv'
    df_last_year.to_csv(csv_path, encoding='utf-8', index=False)
    logger.info('Saved %d training rows from year %d to %s', len(df_last_year), last_year, csv_path)

    sample_path = CSV_DIR / 'training_features_sampled.csv'
    df.sample(n=500, random_state=42).to_csv(sample_path, encoding='utf-8', index=False)
    logger.info('Saved 500 sampled training rows to %s', sample_path)
