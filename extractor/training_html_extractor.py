from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup, Tag
import re
from datetime import datetime 
from tqdm import tqdm
from metadata.columns_raw import training_data_columns

from utils.logger import setup_logger
logger = setup_logger('training_html_extractor')

PARQUET_DIR = Path('data/processed_parquet')
HTML_DIR = Path('horse_training_html')
YEARLY_TRAINING_DIR = Path('data/yearly_parquet/training')
YEARLY_TRAINING_DIR.mkdir(exist_ok=True)


def extract_and_save_training_data(start_year=2001, start_month=1):
    logger.info('Starting training HTML extraction...')
    horse_df = pd.read_parquet(PARQUET_DIR / 'horse_features.parquet')
    
    # 指定された年月以降に出走した馬の調教データを更新
    filtered_df = horse_df[horse_df['datetime'] > f'{start_year}-{start_month:02d}-01']
    id_list = sorted(set(filtered_df['horse_id'].dropna().astype(str)))
    logger.info(f'{len(id_list)} horse IDs after {start_year}-{start_month:02d}.')
    min_year = min(int(horse_id[:4]) for horse_id in id_list)

    # 年ごとに処理
    for year in range(min_year, datetime.now().year + 1):
        yearly_ids = [horse_id for horse_id in id_list if int(horse_id[:4]) == year]
        if yearly_ids == []:
            logger.info(f'Skipping year {year} (no horses)')
            continue
        logger.info(f'Processing year {year} with {len(yearly_ids)} horses')
        save_path = YEARLY_TRAINING_DIR / f'training_data_{year}.parquet'
        if save_path.exists():
            existing_df = pd.read_parquet(save_path)
        else:
            existing_df = pd.DataFrame([], columns=training_data_columns)
        new_rows_df = pd.DataFrame([], columns=training_data_columns)
        for horse_id in tqdm(yearly_ids, desc=f'Year {year}'):
            html_path = HTML_DIR /f'{year}/{horse_id}.html'
            try:
                one_horse_df = extract_training_data_from_html(html_path, horse_id)
                new_rows_df = pd.concat([new_rows_df, one_horse_df], ignore_index=True)
            except Exception as e:
                logger.warning(f'Failed to parse {horse_id}: {e}')
        # 更新した馬のデータがある場合、既存のデータを置き換え
        if not new_rows_df.empty:
            existing_df = existing_df[~existing_df['horse_id'].isin(new_rows_df['horse_id'])]
            existing_df = pd.concat([existing_df, new_rows_df], ignore_index=True)
        existing_df.to_parquet(save_path, index=False)
        logger.info(f'Extracted {len(new_rows_df)} horses for year {year}')


def extract_training_data_from_html(html, horse_id=None):
    # HTMLから調教データの抽出
    with open(html, 'r', encoding='euc-jp') as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    one_horse_df = pd.DataFrame([], columns=training_data_columns)
    table_elements = soup.find_all('table', class_='race_table_01 nk_tb_common')
    for table_element in table_elements:
        if isinstance(table_element, Tag):
            a_tag = table_element.find('a')  
            if isinstance(a_tag, Tag) and a_tag.has_attr('href'):
                race_id = str(a_tag['href']).split('/')[-2]
            else:
                race_id = None
            td_elements = table_element.find_all('td')
            course_type = td_elements[1].text
            course_condition = td_elements[2].text or 0
            li_elements = table_element.select('ul.TrainingTimeDataList li')
            def extract_numeric(text):
                match = re.search(r'\d*\.\d+|\d+', text)
                return float(match.group()) if match else 0.0
            lap_time = [extract_numeric(li.text) for li in li_elements]
            while len(lap_time) < 5:
                lap_time.insert(0, 0)
            top_lap_time = []
            for _, element in enumerate(li_elements):
                classes = element.get('class') or []  

                if 'TokeiColor02' in classes:
                    top_lap_time.append(1)
                elif 'TokeiColor01' in classes:
                    top_lap_time.append(2)
                else:
                    top_lap_time.append(0)

            try:
                position = int(td_elements[5].text) if td_elements[5].text else 0
            except Exception as e:
                logger.warning(f'Failed to parse position for {horse_id}: {e}')
                position = 0
            running_style = td_elements[6].text
            evaluation = td_elements[7].text
            evaluation_grade = td_elements[8].text
            new_row = {
                'horse_id': horse_id,
                'race_id': race_id,
                'course_type': course_type,
                'course_condition': course_condition,
                'lap_time_1': lap_time[0],
                'lap_time_2': lap_time[1],
                'lap_time_3': lap_time[2],
                'lap_time_4': lap_time[3],
                'lap_time_5': lap_time[4],
                'top_lap_time_1': top_lap_time[0],
                'top_lap_time_2': top_lap_time[1],
                'top_lap_time_3': top_lap_time[2],
                'top_lap_time_4': top_lap_time[3],
                'top_lap_time_5': top_lap_time[4],
                'position': position,
                'running_style': running_style,
                'evaluation_grade': evaluation_grade
            }

            new_row_df = pd.DataFrame([new_row])
            one_horse_df = pd.concat([one_horse_df, new_row_df], ignore_index=True)
    return one_horse_df