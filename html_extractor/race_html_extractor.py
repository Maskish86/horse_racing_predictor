from pathlib import Path
import datetime
import pytz
from bs4 import BeautifulSoup, Tag
import pandas as pd
import os
from typing import Tuple, Dict, List, Any, cast
from metadata.columns import race_data_columns, horse_data_columns

from utils.logger import setup_logger
logger = setup_logger("race_html_extractor")


# 現在日時（タイムゾーン: 日本）
now_datetime = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

RACE_URL_DIR = Path('data/race_url')     # レースURLを保存するディレクトリ
RACE_HTML_DIR = Path('data/race_html')   # レースHTMLを保存するディレクトリ
PARQUET_DIR = Path('data/yearly_parquet')        # 出力CSVの保存先ディレクトリ
PARQUET_DIR.mkdir(exist_ok=True)


# 複数年分まとめて処理する関数
def extract_and_save_race_data(start_year=2000):
    for year in range(start_year, now_datetime.year + 1):
        extract_and_save_race_data_by_year(year)


# 1年分のHTMLを処理してParquetを生成する関数
def extract_and_save_race_data_by_year(year):
    race_data_path = PARQUET_DIR / f'race_{year}.csv'
    horse_data_path = PARQUET_DIR / f'horse_{year}.csv'
    race_data_dicts = []
    horse_data_dicts = []
    logger.info(f"Extracting race data for {year}")
    total = 0
    # 各月のHTMLを読み込む
    for month in range(1, 13):
        html_dir = os.path.join(RACE_HTML_DIR, str(year), str(month))
        if not os.path.isdir(html_dir):
            logger.warning(f"Missing directory: {html_dir}")
            continue
        file_list = os.listdir(html_dir)
        total += len(file_list)
        logger.info(f"{len(file_list)} HTML files found for {year}-{month:02d}")
        # 各HTMLを処理
        for file_path in file_list:
            try:
                with open(file_path, 'r', encoding='euc-jp', errors='replace') as f:
                    html = f.read()
                race_id = file_path.split('.')[0]
                race_dict, horse_dicts = get_race_and_horse_data_by_html(race_id, html)
                race_data_dicts.append(race_dict)
                horse_data_dicts.extend(horse_dicts)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
    # DataFrameに変換
    race_df = pd.DataFrame(race_data_dicts)
    horse_df = pd.DataFrame(horse_data_dicts)

    # データ中に\xa0（ノーブレークスペース）が残っていないか確認
    for col in race_df.columns:
        matches = race_df[race_df[col].astype(str).str.contains('\xa0', na=False)]
        if not matches.empty:
            logger.warning(f"Column contains \\xa0: {col}")

    # Parquetを保存
    race_df.to_parquet(race_data_path, index=False)
    horse_df.to_parquet(horse_data_path, index=False)

    logger.info(f"Race data saved: {race_data_path} ({race_df.shape})")
    logger.info(f"Horse data saved: {horse_data_path} ({horse_df.shape})")
    logger.info(f"Total HTML files processed: {total}")
    

# HTMLからレース情報と競走馬情報を抽出する関数

def safe_find_text(parent: Tag | None, tag: str, default: str = '') -> str:
    if parent is None:
        return default
    found = parent.find(tag)
    return found.get_text(strip=True) if found else default

def safe_get_text(elem: Tag | None, default: str = '') -> str:
    return elem.get_text(strip=True) if elem else default

def as_tag(elem: Any) -> Tag | None:
    return cast(Tag, elem) if isinstance(elem, Tag) else None

def get_race_and_horse_data_by_html(race_id: str, html: str) -> Tuple[Dict, List[Dict]]:
    soup = BeautifulSoup(html, 'html.parser')

    race_data: Dict = {'race_id': race_id}
    horse_data: List[Dict] = []

    # -------------------- レース情報 --------------------
    try:
        data_intro = soup.find('div', class_='data_intro')
        data_intro_tag = cast(Tag, data_intro)
        if data_intro is None:
            logger.warning(f"[{race_id}] Missing data_intro block")
            return race_data, horse_data

        race_data['race_round'] = safe_find_text(data_intro_tag, 'dt')
        race_data['race_grade'] = safe_find_text(data_intro_tag, 'h1')

        p_tags = data_intro_tag.find_all('p')
        if len(p_tags) >= 1:
            parts = p_tags[0].get_text(strip=True).split('\xa0/\xa0')
            race_data['race_course'] = parts[0] if len(parts) > 0 else 'N/A'
            race_data['weather'] = parts[1] if len(parts) > 1 else 'N/A'
            race_data['track_condition'] = parts[2].replace('\xa0', '') if len(parts) > 2 else 'N/A'
            race_data['time'] = parts[3] if len(parts) > 3 else 'N/A'
        else:
            race_data.update(dict.fromkeys(['race_course', 'weather', 'track_condition', 'time'], 'N/A'))

        if len(p_tags) >= 2:
            smalltxt = p_tags[1].get_text(strip=True).split(' ')
            race_data['date'] = smalltxt[0] if len(smalltxt) > 0 else 'N/A'
            race_data['location'] = smalltxt[1] if len(smalltxt) > 1 else 'N/A'
            race_data['race_class'] = smalltxt[2].replace('\xa0', '') if len(smalltxt) > 2 else 'N/A'
        else:
            race_data.update(dict.fromkeys(['date', 'location', 'race_class'], 'N/A'))
    except Exception as e:
        logger.error(f"[{race_id}] Failed to extract race header: {e}")

    # -------------------- 結果テーブル --------------------
    try:
        result_table = soup.find('table', class_='race_table_01 nk_tb_common')
        result_rows = cast(Tag, result_table).find_all('tr') if result_table else []

        race_data['total_horse_number'] = len(result_rows) - 1

        for i in range(1, 4):
            try:
                row = cast(Tag, result_rows[i]).find_all('td')
                race_data[f'frame_number_{"first" if i==1 else "second" if i==2 else "third"}'] = row[1].get_text()
                race_data[f'horse_number_{"first" if i==1 else "second" if i==2 else "third"}'] = row[2].get_text()
            except Exception:
                race_data[f'frame_number_{"first" if i==1 else "second" if i==2 else "third"}'] = '0'
                race_data[f'horse_number_{"first" if i==1 else "second" if i==2 else "third"}'] = '0'
    except Exception as e:
        logger.warning(f"[{race_id}] Error extracting result table: {e}")

    # -------------------- 払戻金 --------------------
    try:
        pay_tables = soup.find_all('table', class_='pay_table_01')
        race_data['win'] = race_data['show_1'] = race_data['show_2'] = race_data['show_3'] = '0'
        race_data['bracket_quinella'] = race_data['quinella'] = '0'
        race_data['quinella_place_1_2'] = race_data['quinella_place_1_3'] = race_data['quinella_place_2_3'] = '0'
        race_data['exacta'] = race_data['trio'] = race_data['trifecta'] = '0'
    except Exception as e:
        logger.warning(f"[{race_id}] Failed extracting additional race fields: {e}")


    try:
        # -------------------- 払戻金1 (単勝・複勝など) --------------------
        if len(pay_tables) >= 1:
            table1 = cast(Tag, pay_tables[0])
            rows1 = table1.find_all('tr')

            # 単勝
            row0 = as_tag(rows1[0])
            td_win = row0.find('td', class_='txt_r') if row0 else None
            race_data['win'] = safe_get_text(as_tag(td_win))

            # 複勝（複数値）
            row1 = as_tag(rows1[1]) if len(rows1) > 1 else None
            td_show = row1.find('td', class_='txt_r') if row1 else None
            tag_show = as_tag(td_show)
            show = list(tag_show.strings) if tag_show else []

            for i in range(3):
                race_data[f'show_{i+1}'] = show[i] if i < len(show) else '0'

            # 枠連・馬連
            row2 = as_tag(rows1[2]) if len(rows1) > 2 else None
            td_bracket_q = row2.find('td', class_='txt_r') if row2 else None
            race_data['bracket_quinella'] = safe_get_text(as_tag(td_bracket_q))

            row3 = as_tag(rows1[3]) if len(rows1) > 3 else None
            td_quinella = row3.find('td', class_='txt_r') if row3 else None
            race_data['quinella'] = safe_get_text(as_tag(td_quinella))

        # -------------------- 払戻金2 (馬連・三連複など) --------------------
        if len(pay_tables) >= 2:
            table2 = as_tag(pay_tables[1])
            rows2 = table2.find_all('tr') if table2 else []

            # ワイド（複勝）
            row0 = as_tag(rows2[0]) if len(rows2) > 0 else None
            td_wide = row0.find('td', class_='txt_r') if row0 else None
            tag_wide = as_tag(td_wide)
            q_place = list(tag_wide.strings) if tag_wide else []
            race_data['quinella_place_1_2'] = q_place[0] if len(q_place) > 0 else '0'
            race_data['quinella_place_1_3'] = q_place[1] if len(q_place) > 1 else '0'
            race_data['quinella_place_2_3'] = q_place[2] if len(q_place) > 2 else '0'

            # 馬単・三連複・三連単
            row1 = as_tag(rows2[1]) if len(rows2) > 1 else None
            td_exacta = row1.find('td', class_='txt_r') if row1 else None
            race_data['exacta'] = safe_get_text(as_tag(td_exacta))

            row2 = as_tag(rows2[2]) if len(rows2) > 2 else None
            td_trio = row2.find('td', class_='txt_r') if row2 else None
            race_data['trio'] = safe_get_text(as_tag(td_trio))

            row3 = as_tag(rows2[3]) if len(rows2) > 3 else None
            td_trifecta = row3.find('td', class_='txt_r') if row3 else None
            race_data['trifecta'] = safe_get_text(as_tag(td_trifecta))

    except Exception as e:
        logger.warning(f"Pay table parse failed for {race_id}: {e}")

    # -------------------- 馬場指数・ラップ --------------------
    result_table_02 = soup.find_all('table', class_='result_table_02')
    table0 = as_tag(result_table_02[0]) if len(result_table_02) > 0 else None
    table2 = as_tag(result_table_02[2]) if len(result_table_02) > 2 else None

    try:
        tr0 = as_tag(table0.find_all('tr')[0]) if table0 and table0.find_all('tr') else None
        td0 = as_tag(tr0.find('td')) if tr0 else None
        text = td0.get_text(strip=True).split('(')[0] if td0 else '0'
        race_data['ground_index'] = text
    except Exception as e:
        logger.warning(f"ground_index parse failed: {e}")
        race_data['ground_index'] = '0'

    try:
        trs = table2.find_all('tr') if table2 else []
        tr_lap = as_tag(trs[0]) if len(trs) > 0 else None
        tr_pace = as_tag(trs[1]) if len(trs) > 1 else None

        td_lap = as_tag(tr_lap.find('td')) if tr_lap else None
        td_pace = as_tag(tr_pace.find('td')) if tr_pace else None

        race_data['lap_time'] = td_lap.get_text(strip=True) if td_lap else '0'
        race_data['pace_time'] = td_pace.get_text(strip=True) if td_pace else '0'
    except Exception as e:
        logger.warning(f"lap/pace parse failed: {e}")
        race_data['lap_time'] = race_data['pace_time'] = '0'

    # -------------------- 馬情報 --------------------
    for tr in result_rows[1:]:
        tr_tag = as_tag(tr)
        tds = tr_tag.find_all('td') if tr_tag else []
        if len(tds) < 20:
            continue

        try:
            td_horse = as_tag(tds[3])
            td_rider = as_tag(tds[6])
            td_tamer = as_tag(tds[18])
            td_owner = as_tag(tds[19])

            a_horse = as_tag(td_horse.find('a')) if td_horse else None
            a_rider = as_tag(td_rider.find('a')) if td_rider else None
            a_tamer = as_tag(td_tamer.find('a')) if td_tamer else None
            a_owner = as_tag(td_owner.find('a')) if td_owner else None

            td_aff = as_tag(tds[18])
            if td_aff and td_aff.contents and isinstance(td_aff.contents[0], str):
                affiliation = td_aff.contents[0].strip().strip('[]')
            else:
                affiliation = ''

            horse = {
                'race_id': race_id,
                'rank': tds[0].get_text(strip=True),
                'frame_number': tds[1].get_text(strip=True),
                'horse_number': tds[2].get_text(strip=True),
                'horse_id': str(a_horse.get('href')).split('/')[-2] if a_horse and a_horse.has_attr('href') else '0',
                'sex_and_age': tds[4].get_text(strip=True),
                'burden_weight': tds[5].get_text(strip=True),
                'rider_id': str(a_rider.get('href')).split('/')[-2] if a_rider and a_rider.has_attr('href') else '0',
                'goal_time': tds[7].get_text(strip=True),
                'margin': tds[8].get_text(strip=True),
                'speed_index': tds[9].get_text(strip=True),
                'rank_path': tds[10].get_text(strip=True),
                'last3f_time': tds[11].get_text(strip=True),
                'win_odds': tds[12].get_text(strip=True),
                'popular': tds[13].get_text(strip=True),
                'horse_weight': tds[14].get_text(strip=True),
                'remark': tds[17].get_text(strip=True),
                'affiliation': affiliation,
                'tamer_id': str(a_tamer.get('href')).split('/')[-2] if a_tamer and a_tamer.has_attr('href') else '0',
                'owner_id': str(a_owner.get('href')).split('/')[-2] if a_owner and a_owner.has_attr('href') else '0'
            }
            horse_data.append(horse)

        except Exception as e:
            logger.warning(f"Error parsing horse row in race_id={race_id}: {e}")
            continue

    return race_data, horse_data



# メイン処理
if __name__ == '__main__':
    logger.info('Extracting data from HTML and saving to Parquet files...')
    extract_and_save_race_data()
