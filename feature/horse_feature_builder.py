from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime 
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
from metadata.columns_feature import horse_feature_columns

from utils.logger import setup_logger
logger = setup_logger("horse_feature_builder")

YEARLY_HORSE_DIR = Path("data/yearly_parquet/horse")
SAVE_DIR = Path("data/processed_parquet")
SAVE_DIR.mkdir(exist_ok=True)
PREDICT_FEATURE_DIR = Path("data/predict_features")
CSV_DIR = Path("data/csv")
CSV_DIR.mkdir(exist_ok=True)

def process_horse_data():
    logger.info("Starting horse feature building pipeline...")
    horse_df = load_and_merge_data()
    horse_df = add_basic_features(horse_df)
    horse_df = process_time_features(horse_df)
    horse_df = add_positional_features(horse_df)   
    horse_df = add_kde_features(horse_df)
    finalize_and_save(horse_df)
    

def load_and_merge_data():
    horse_files = list(YEARLY_HORSE_DIR.glob("horse_*.parquet"))
    horse_df = pd.concat([pd.read_parquet(f) for f in horse_files], axis=0)
    cols_for_merge = ["race_id","location", "ground_type", "distance", "is_outer_course", "race_class_level", "datetime", "total_horse_number", "field_size_bin", "corners_num", "pace_early_vs_late", "norm_tough_gi",  "race_age", "turn_direction"]
    race_df_for_merge = pd.read_parquet(SAVE_DIR / "race_features.parquet")[cols_for_merge]
    
    columns_to_convert = ["race_id", "horse_id", "tamer_id", "owner_id", "rider_id"]
    horse_df[columns_to_convert] = horse_df[columns_to_convert].apply(lambda x: x.astype(str))

    horse_df["race_id"] = horse_df["race_id"].str.extract(r"(\d{12})")[0]
    horse_df = horse_df[horse_df["race_id"].notnull()]
    horse_df = pd.merge(horse_df, race_df_for_merge, on="race_id", how="left")
    logger.info("Finished loading horse data. Shape: %s", horse_df.shape)
    return horse_df


def add_horse_feature_summaries(horse_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    rolling = horse_df.sort_values("datetime").groupby("horse_id")

    for col in feature_cols:
        horse_df[f"{col}_mean5"] = rolling[col].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
        horse_df[f"{col}_stdlog5"] = np.log1p(
            rolling[col].transform(lambda x: x.shift(1).rolling(5, min_periods=3).std())
        )

        horse_df[f"{col}_mom_tr5"] = rolling[col].transform(
            lambda x: x.diff().shift(1).rolling(5, min_periods=3).mean()
        )
        horse_df[f"{col}_mom_ema5"] = rolling[col].transform(
            lambda x: x.shift(1).ewm(span=5, adjust=False).mean()
        )
    return horse_df


def add_race_zscore(horse_df: pd.DataFrame, feature_cols: list, with_rank: bool = False) -> pd.DataFrame:
    race_groups = horse_df.groupby("race_id")

    for col in feature_cols:
        group_mean = race_groups[col].transform("mean")
        group_std = race_groups[col].transform("std")

        horse_df[f"{col}_race_zscore"] = (horse_df[col] - group_mean) / group_std

        if with_rank:
            group_rank = race_groups[col].rank(method="average")
            denom = (horse_df["total_horse_number"] - 1).replace(0, 1)
            horse_df[f"{col}_race_rank"] = (1 - (group_rank - 1) / denom)

    return horse_df


def add_basic_features(horse_df):
    # 基本的な特徴量の追加
    logger.info("Start adding basic features")
    horse_df = add_horse_feature_summaries(horse_df, ["race_class_level", "distance"])
    # 日付に関する特徴量
    horse_df["datetime"] = horse_df["datetime"].replace(["0", 0], np.nan)
    horse_df["datetime"] = pd.to_datetime(horse_df["datetime"], errors="coerce")
    horse_df = horse_df.dropna(subset=["datetime"])  
    horse_df["race_year"] = horse_df["datetime"].dt.year.astype(int)  

    # 順位に関する特徴量
    horse_df["total_horse_number"] = horse_df["total_horse_number"].fillna(18)
    horse_df["rank"] = horse_df["rank"].str.replace(r"\(?(降|再)\)?", "", regex=True)
    horse_df = horse_df[~horse_df["rank"].isin(["取", "除", "失", "中"])]
    horse_df = horse_df.dropna(subset=["rank"]) # type: ignore
    horse_df["rank"] = horse_df["rank"].astype(int)
    horse_df["norm_rank"] = 1 - (horse_df["rank"] - 1) / (horse_df["total_horse_number"] - 1)
    
    # スタート位置に関する特徴量
    horse_df["norm_horse_number"] = (horse_df["horse_number"] - 1) / (horse_df["total_horse_number"] - 1)
    horse_df["horse_number_cat"] = pd.cut(horse_df["norm_horse_number"],   bins=list(np.linspace(0, 1, 19)), labels=False, include_lowest=True)
    horse_df["is_inner"] = (horse_df["norm_horse_number"] <= 0.3).astype(int)
    horse_df["is_middle"] = ((horse_df["norm_horse_number"] > 0.3) & (horse_df["norm_horse_number"] <= 0.7)).astype(int)
    horse_df["is_outer"] = (horse_df["norm_horse_number"] > 0.7).astype(int)

    # 性齢に関する特徴量
    def extract_and_replace(df, column, pattern, new_column, replace_value):
        extracted = df[column].str.extract(f"({pattern})", expand=False)
        df[new_column] = extracted.fillna(0).replace(pattern, replace_value)
        return df

    sex_patterns = {"is_gelding": "セ", "is_mare": "牝", "is_stallion": "牡"}
    for new_column, pattern in sex_patterns.items():
        horse_df = extract_and_replace(horse_df, "sex_and_age", pattern, new_column, 1)
    horse_df["sex_and_age"] = horse_df["sex_and_age"].str.replace(r"[牝牡セ]", "", regex=True).astype(int)
    horse_df = horse_df.rename(columns={"sex_and_age": "age"})
    horse_df["age_cat"] = pd.cut(horse_df["age"], bins=[0, 2, 3, 4, 5, 6, 7, np.inf], labels=[0, 1, 2, 3, 4, 5, 6],include_lowest=True, right=True).astype(int)
    horse_df.loc[horse_df["age"] > 3, "race_age"] = 4

    # 人気に関する特徴量
    horse_df["win_odds"] = horse_df["win_odds"].astype(float)
    horse_df["log_win_odds"] = -np.log1p(horse_df["win_odds"])
    implied_proba = 1 / (horse_df["win_odds"] + 1e-3)
    horse_df["implied_proba_logit"] = np.log(implied_proba/ (1 - implied_proba))
    horse_df["popular"] = horse_df["popular"].fillna(horse_df["total_horse_number"]).astype(int)
    horse_df["norm_popularity"] = 1 - (horse_df["popular"] - 1) / (horse_df["total_horse_number"] - 1)
    horse_df["rank_vs_popular"] = horse_df["norm_rank"] - horse_df["norm_popularity"]
    
    market_feat_cols = ["norm_rank", "log_win_odds", "implied_proba_logit", "norm_popularity", "rank_vs_popular"]
    horse_df = add_horse_feature_summaries(horse_df, market_feat_cols)

    # 不利に関する特徴量
    remark_mapping = {"出遅れ": "is_slow_break", "出脚鈍い": "is_slow_break", "躓く": "is_slow_break", "アオル": "is_slow_break",
                      "S不利": "is_trouble", "S接触":"is_trouble", "Sヨレル": "is_trouble", "直線不利": "is_trouble"}
    for key, col in remark_mapping.items():
        mask = horse_df["remark"].str.contains(key, na=False)
        horse_df.loc[mask, col] = 1
    for col in set(remark_mapping.values()):
        horse_df[col] = horse_df[col].fillna(0).astype(int)

    horse_df["slow_break_rate5"] = (
        horse_df.groupby("horse_id")["is_slow_break"]
          .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum()) / 5
    ).fillna(0)

    horse_df["trouble_rate5"] = (
        horse_df.groupby("horse_id")["is_trouble"]
          .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum()) / 5
    ).fillna(0)


    # 馬体重に関する特徴量
    horse_df["horse_weight"] = horse_df["horse_weight"].str.replace(r"\(([-|+]?\d*)\)", "", regex=True)\
        .replace("計不", np.nan).astype(float)
    horse_df["horse_weight"] = horse_df.groupby("horse_id")["horse_weight"].transform(lambda x: x.fillna(x.mean()))
    horse_df["burden_weight"] = horse_df["burden_weight"].astype(float)
    horse_df["burden_ratio"] = horse_df["burden_weight"] / horse_df["horse_weight"]
    weight_feat_cols = ["horse_weight", "burden_weight", "burden_ratio"]
    horse_df = add_horse_feature_summaries(horse_df, weight_feat_cols)
    horse_df = add_race_zscore(horse_df, ["horse_weight", "burden_weight", "burden_ratio"], with_rank=False)

      
    # 厩舎に関する特徴量
    affiliation_mapping = {"西": "is_Ritto", "東":"is_Miho", "地": "is_local_or_foreign", "外": "is_local_or_foreign"}
    for key, value in affiliation_mapping.items():
        horse_df.loc[horse_df["affiliation"].str.contains(key, na=False), value] = 1
    racecourse_distance_from_training_center = {
        "Tokyo": {"is_Ritto": 430, "is_Miho": 120},
        "Nakayama": {"is_Ritto": 450, "is_Miho": 75},
        "Kyoto": {"is_Ritto": 45, "is_Miho": 430},
        "Hanshin": {"is_Ritto": 85, "is_Miho": 450},
        "Chukyo": {"is_Ritto": 150, "is_Miho": 370},
        "Fukushima": {"is_Ritto": 530, "is_Miho": 290},
        "Niigata": {"is_Ritto": 390, "is_Miho": 290},
        "Sapporo": {"is_Ritto": 1040, "is_Miho": 850},
        "Hakodate": {"is_Ritto": 880, "is_Miho": 720},
        "Kokura": {"is_Ritto": 500, "is_Miho": 1000},
    }

    def compute_distance(row: pd.Series) -> float:
        loc = row.get("location")

        if pd.isna(loc) or loc not in racecourse_distance_from_training_center:
            return float("nan")

        dist_map = racecourse_distance_from_training_center[loc]

        if row.get("is_Ritto", 0) == 1:
            return float(dist_map.get("is_Ritto", float("nan")))
        elif row.get("is_Miho", 0) == 1:
            return float(dist_map.get("is_Miho", float("nan")))
        elif row.get("is_local_or_foreign", 0) == 1:
            return float("nan")
        else:
            return float("nan")

    horse_df["center_dist"] = horse_df.apply(compute_distance, axis=1)
    horse_df["center_dist"] = horse_df["center_dist"].fillna(horse_df["center_dist"].mean())
    
    horse_df = (
        horse_df.sort_values(["horse_id", "datetime"], ascending=[True, False])
        .groupby("horse_id")
        .apply(lambda g: g.assign(distance_diff=g["distance"].diff(-1)))
        .reset_index(drop=True)
        .sort_values("datetime")

    )
    horse_df["distance_diff"] = horse_df["distance_diff"].clip(-800, 800)
    horse_df["norm_distance_diff"] = (horse_df["distance_diff"] + 800) / 1600
    horse_df["distance_diff_cat"] = (horse_df["distance_diff"].clip(-800, 800) + 800) // 100
    
    return horse_df


def compute_time_causal_group_stats(df, add_cols=None, suffix="", decay=0.95, eps=1e-3):
    if add_cols is None:
        add_cols = []

    df = df.copy()
    df["__index__"] = df.index  # preserve original row order

    course_cols = ["location", "ground_type", "distance", "is_outer_course"]
    group_cols = course_cols + add_cols
    val_cols = ["goal_time", "last3f_time"]

    all_results = []
    year_groups = dict(tuple(df.groupby("race_year")))

    for col in val_cols:
        df_group_source = df[group_cols + ["race_year", col]]
        grouped = (
            df_group_source.groupby(group_cols + ["race_year"], observed=True, sort=False)[col]
            .agg(["mean", "std", "count"])
            .rename(columns={
                "mean": f"{col}_{suffix}_mean",
                "std": f"{col}_{suffix}_std",
                "count": f"{col}_{suffix}_count"
            })
            .reset_index()
        )

        result_rows = []
        for year, df_this_year in sorted(year_groups.items()):
            df_this_year = df_this_year.copy()
            df_past = grouped[grouped["race_year"] < year]

            if df_past.empty:
                df_this_year[[f"{col}_{suffix}_mean", f"{col}_{suffix}_std"]] = np.nan
            else:
                df_past["year_diff"] = year - df_past["race_year"]
                df_past["decay_weight"] = np.exp(-decay * df_past["year_diff"])
                df_past["effective_weight"] = df_past[f"{col}_{suffix}_count"] * df_past["decay_weight"]

                weighted_stats = (
                    df_past.groupby(group_cols).apply(
                        lambda g: pd.Series({
                            f"{col}_{suffix}_mean": np.average(
                                g[f"{col}_{suffix}_mean"], weights=g["effective_weight"]
                            ),
                            f"{col}_{suffix}_std": np.average(
                                g[f"{col}_{suffix}_std"], weights=g["effective_weight"]
                            ),
                        })
                    ).reset_index()
                )

                df_this_year = df_this_year.merge(weighted_stats, on=group_cols, how="left")

            result_rows.append(df_this_year[["__index__", f"{col}_{suffix}_mean", f"{col}_{suffix}_std"]])

        final_df = pd.concat(result_rows, axis=0, ignore_index=True)
        all_results.append(final_df)

    combined = all_results[0]
    for next_df in all_results[1:]:
        combined = combined.merge(next_df, on="__index__", how="left")

    merged_df = df.merge(combined, on="__index__", how="left").drop(columns="__index__")

    for col in val_cols:
        base_col = col.replace("_time", "")
        merged_df[f"{base_col}_zscore_{suffix}"] = -(
            (merged_df[col] - merged_df[f"{col}_{suffix}_mean"])
            / (merged_df[f"{col}_{suffix}_std"] + eps)
        )

    merged_df = merged_df.drop(columns=[
        f"{col}_{suffix}_mean" for col in val_cols
    ] + [
        f"{col}_{suffix}_std" for col in val_cols
    ] + [
        f"{col}_{suffix}_count" for col in val_cols
    ])

    return merged_df


def rolling_conditional_mean(df, condition_col, condition_value, group_col="horse_id", window=3, suffix=""):
    df = df.sort_values([group_col, "datetime"]).reset_index(drop=True)
    grouped = df.groupby(group_col, observed=True, sort=False)

    for base in ["goal_zscore_class", "last3f_zscore_class", "speed_index"]:
        base_clean = base.replace("_class", "")
        new_col = f"{base_clean}_{suffix}_mean{window}"

        df[new_col] = grouped[base_clean].transform(
            lambda x: x.where(df.loc[x.index, condition_col] == condition_value)
                       .shift(1)
                       .rolling(window, min_periods=1)
                       .mean()
        )

    return df



def rolling_context_mean(df, context_col, group_col="horse_id", window=3, suffix=""):
    df = df.sort_values([group_col, "datetime"]).reset_index(drop=True)
    grouped = df.groupby([group_col, context_col], observed=True, sort=False)

    for base in ["goal_zscore_class", "last3f_zscore_class", "speed_index"]:
        base_clean = base.replace("_class", "")
        new_col = f"{base_clean}_{suffix}_mean{window}"

        df[new_col] = grouped[base_clean].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    return df


def process_time_features(df):
    # 時間に関する特徴量の追加
    logger.info("Start processing time features")
    df["goal_time"] = (
        pd.to_datetime(df["goal_time"], format="%M:%S.%f") -
        pd.to_datetime("00:00.0", format="%M:%S.%f")
    ).dt.total_seconds()
    winner_goal_time = df.groupby("race_id")["goal_time"].transform("min")
    df["goal_time_diff_winner"] = df["goal_time"] - winner_goal_time

    time_diff_mask = df["goal_time_diff_winner"] > 25
    for col in ["goal_time", "last3f_time"]:
        df.loc[time_diff_mask, col] = (
            df.groupby("race_id")[col]
            .transform(lambda x: x.dropna().sort_values().iloc[1] if len(x.dropna()) > 1 else x.dropna().iloc[0])
        )
    
    df["goal_time"] = df["goal_time"].fillna(df.groupby("race_id")["goal_time"].transform("max"))
    df["last3f_time"] = df["last3f_time"].fillna(df.groupby("race_id")["last3f_time"].transform("max"))
    df = df.drop(columns=["goal_time_diff_winner"], axis=1)
    
    df["is_high_level"] = (df["race_class_level"] > 2).astype(int).fillna(0)
    df["is_tough_track"] = (df["norm_tough_gi"] - 0.2 > 0).astype(int).fillna(0)
    df = compute_time_causal_group_stats(df, [], "course")
    df = compute_time_causal_group_stats(df, ["is_high_level"] + ["race_age"], "level")
    df = compute_time_causal_group_stats(df, ["is_high_level"] + ["race_age"] + ["is_tough_track"], "tough")
    df = compute_time_causal_group_stats(df, ["race_class_level"] + ["race_age"], "class")

    df["speed_index"] = (
        pd.to_numeric(df["speed_index"], errors="coerce")
        .fillna(0)
        .astype(int)
        .clip(lower=-100)
    )
    df = add_horse_feature_summaries(df, ["goal_zscore_course", "last3f_zscore_course",
                                        "goal_zscore_level", "last3f_zscore_level", 
                                        "goal_zscore_tough", "last3f_zscore_tough",
                                        "goal_zscore_class", "last3f_zscore_class",
                                        "speed_index"])
    df["is_fast_pace"] = (df["pace_early_vs_late"] > 0).astype(int).fillna(0)
    df["is_inner_pos"] = (df["norm_horse_number"] <= 0.5).astype(int).fillna(0)
    df = add_race_zscore(df, ["goal_zscore_class", "last3f_zscore_class", "speed_index"], with_rank=True)
    df = rolling_conditional_mean(df, "is_tough_track", 1, suffix="tough")
    df = rolling_conditional_mean(df, "is_tough_track", 0, suffix="soft")
    df = rolling_conditional_mean(df, "is_fast_pace", 1, suffix="fastpace")
    df = rolling_conditional_mean(df, "is_fast_pace", 0, suffix="slowpace")
    df = rolling_conditional_mean(df, "turn_direction", 1, suffix="cw")
    df = rolling_conditional_mean(df, "turn_direction", -1, suffix="ccw")
    df = rolling_conditional_mean(df, "is_inner_pos", 1, suffix="inner")
    df = rolling_conditional_mean(df, "is_outer_pos", 0, suffix="outer")
    df = rolling_context_mean(df, context_col="location", window=3, suffix="location")
    df = rolling_context_mean(df, context_col="distance", window=3, suffix="distance")
    return df


def add_positional_features(horse_df):
    # 位置取りに関する特徴量の追加
    logger.info("Start adding positional features")
    norm_pos_array = 1 - (
        horse_df["rank_path"].str.split("-", expand=True).astype(float) - 1
    ) / (horse_df["total_horse_number"] - 1)

    pos_mean = norm_pos_array.mean(axis=1).fillna(0.5)
    pos_std = norm_pos_array.std(axis=1)
    horse_df["pos_volatility_index"] = np.arcsinh(pos_std / (pos_mean + 0.01))

    horse_df["pos_early"] = norm_pos_array.iloc[:, 0]
    horse_df["pos_late"] = norm_pos_array.ffill(axis=1).iloc[:, -1]
    n = norm_pos_array.shape[1]
    if n % 2 == 1:
        horse_df["pos_mid"] = norm_pos_array.iloc[:, n // 2]
    else:
        horse_df["pos_mid"] = norm_pos_array.iloc[:, [n//2 - 1, n//2]].mean(axis=1)
    horse_df["pos_early_efficiency"] = np.arcsinh(horse_df["pos_early"] - (1 - horse_df["norm_horse_number"]))
    horse_df["pos_gain"] = np.arcsinh(horse_df["pos_late"] - horse_df["pos_early"]).fillna(0)
    horse_df["pos_gain_per_corner"] = horse_df["pos_gain"] / (horse_df["corners_num"] + 0.1)
    horse_df["pos_final_kick"] =  np.arcsinh(norm_pos_array.ffill(axis=1).iloc[:, -2] - horse_df["pos_late"])
    horse_df["pos_finish_gain"] =  np.arcsinh(horse_df["pos_late"]- horse_df["norm_rank"])
    horse_df["pos_finish_efficiency"] = np.arcsinh(
        horse_df["norm_rank"] - (1 - horse_df["norm_horse_number"])
    )
    
    pos_feat_cols = [col for col in horse_df.columns if col.startswith("pos_")]
    horse_df = add_horse_feature_summaries(horse_df, pos_feat_cols)
    horse_df = add_race_zscore(horse_df, ["pos_early", "pos_mid", "pos_late"], with_rank=True)
    horse_df["rank_path_count"] = (horse_df["rank_path"].str.count("-") + 1).fillna(0)


    def normalize_and_bin(series):
        return pd.cut(
            (series.clip(0.07, 1) - 0.07) / 0.93,
            bins=list(np.linspace(0, 1, 19)),
            labels=False,
            include_lowest=True
        )

    horse_df["early_pos5_cat"] = normalize_and_bin(horse_df["norm_early_pos5"])
    horse_df["late_pos5_cat"] = normalize_and_bin(horse_df["norm_late_pos5"])


    return horse_df.drop(["rank_path","pos_mean"], axis=1)


def add_kde_features(horse_df):
    # KDEに基づく特徴量の追加
    logger.info("Start adding KDE features")
    
    group_cols = ["location", "ground_type", "distance", "is_outer_course", "field_size_bin"]
    group_cols_dd = ["location", "ground_type", "distance", "is_outer_course"] 

    kde_early_df = train_and_predict_kde(horse_df, "norm_early_pos", group_cols)
    kde_early_pred_df = train_and_predict_kde(horse_df, "norm_early_pos5", group_cols)
    kde_late_df = train_and_predict_kde(horse_df, "norm_late_pos", group_cols)
    kde_late_pred_df = train_and_predict_kde(horse_df, "norm_late_pos5", group_cols)
    kde_num_df = train_and_predict_kde(horse_df, "norm_horse_number", group_cols)
    kde_dd_df = train_and_predict_kde(horse_df, "norm_distance_diff", group_cols_dd)
    
    horse_df = horse_df.merge(kde_early_df,  on=group_cols + ["norm_early_pos"], how="left")
    horse_df = horse_df.merge(kde_early_pred_df, on=group_cols + ["norm_early_pos5",], how="left")
    horse_df = horse_df.merge(kde_late_df,   on=group_cols + ["norm_late_pos5"], how="left")
    horse_df = horse_df.merge(kde_late_pred_df,  on=group_cols + ["norm_late_pos5"], how="left")
    horse_df = horse_df.merge(kde_num_df,    on=group_cols + ["norm_horse_number"], how="left")
    horse_df = horse_df.merge(kde_dd_df, on=group_cols_dd + ["norm_distance_diff"], how="left")
    horse_df["distance_diff"] = horse_df["distance_diff"].fillna(0)
    

    kde_cols = [col for col in horse_df.columns if col.endswith("_kde")]
    for col in kde_cols:
        grouped = horse_df.groupby("race_id")
        mean = grouped[col].transform("mean")
        std = grouped[col].transform("std") + 1e-3  
        horse_df[col] = horse_df[col].fillna(mean)
        horse_df[col] = ((horse_df[col] - mean) / std).clip(-3, 3)
        horse_df[col] = horse_df[col].fillna(0.0)
    return horse_df

def train_and_predict_kde(horse_df, feature_col, group_cols, save_model=True, min_samples=10, decay_weight=0.95):
    # KDEモデルの訓練と予測
    result_rows = []
    horse_df = horse_df.copy()
    prefix = feature_col.replace("norm_", "")
    this_year = datetime.now().year
    for year in tqdm(range(2009, this_year + 1), desc=f"KDE {feature_col} (causal)"):
        win_kde_dict = {}
        show_kde_dict = {}
        df_train = horse_df[horse_df["race_year"] < year]
        df_eval = horse_df[horse_df["race_year"] == year]
        eval_keys = set(tuple(row) for _, row in df_eval[group_cols].drop_duplicates().iterrows())

        for key, group in df_train.groupby(group_cols):
            if key not in eval_keys and this_year != year:
                continue
            x = group[feature_col].dropna()
            if len(x) < min_samples:
                continue

            X = x.values.reshape(-1, 1)
            group = group.loc[x.index]

            decay = np.power(decay_weight, (year - group["race_year"]).clip(0, 100))

            is_win = (group["rank"] == 1).astype(float)
            is_show = (group["rank"] <= 3).astype(float)

            win_weights = decay * is_win
            show_weights = decay * is_show

            if win_weights.sum() == 0 or show_weights.sum() == 0:
                continue

            x_vals = X.squeeze()
            mean = np.mean(x_vals)
            var = np.var(x_vals)
            bw = np.clip(1.06 * np.sqrt(var) * len(X) ** (-1 / 5), 0.05, 1.5)


            win_kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(X, sample_weight=win_weights)
            show_kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(X, sample_weight=show_weights)
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
                row["race_year"] = year
                row[feature_col] = x_val

                try:
                    row[f"{prefix}_win_kde"] = win_kde.score_samples([[x_val]])[0]
                except:
                    row[f"{prefix}_win_kde"] = 0.0

                try:
                    row[f"{prefix}_show_kde"] = show_kde.score_samples([[x_val]])[0]
                except:
                    row[f"{prefix}_show_kde"] = 0.0

                result_rows.append(row)

        if save_model and (year == this_year):
            pd.to_pickle(win_kde_dict, PREDICT_FEATURE_DIR / f"kde_{prefix}_win.pkl")
            pd.to_pickle(show_kde_dict, PREDICT_FEATURE_DIR / f"kde_{prefix}_show.pkl")


    kde_lookup_df = pd.DataFrame(result_rows)
    return kde_lookup_df
   
    
def finalize_and_save(horse_df): 
    # 最終的なデータフレームの整形と保存
    logger.info("Finalizing and saving")
    horse_df = horse_df[horse_feature_columns]
    logger.info("Columns selected: %d columns, final shape: %s", horse_df.shape[1], horse_df.shape)

    save_path = SAVE_DIR / "horse_features.parquet"
    horse_df.to_parquet(save_path, index=False)
    logger.info("Saved full horse features to %s", save_path)

    last_year = datetime.now().year - 1
    mask = (horse_df["race_id"].astype(int) < (last_year+1) * 100_000_000) & (horse_df["race_id"].astype(int) > last_year * 100_000_000) 
    horse_df_last_year = horse_df[mask]
    csv_path = CSV_DIR / f"horse_features_{last_year}.csv"
    horse_df_last_year.to_csv(csv_path, encoding="utf-8", index=False)
    logger.info("Saved %d horses from year %d to %s", len(horse_df_last_year), last_year, csv_path)

    sample_path = CSV_DIR / "horse_features_sampled.csv"
    horse_df.sample(n=500, random_state=42).to_csv(sample_path, encoding="utf-8", index=False)
    logger.info("Saved 500 sampled horses to %s", sample_path)

if __name__ == "__main__":
    process_horse_data()
