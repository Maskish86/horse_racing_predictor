import datetime
import pytz
from bs4 import BeautifulSoup
import pandas as pd
import os
from os import path

# 現在日時（タイムゾーン: 日本）
now_datetime = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

RACE_URL_DIR = 'race_url'     # レースURLを保存するディレクトリ
RACE_HTML_DIR = 'race_html'   # レースHTMLを保存するディレクトリ
CSV_DIR = 'yearly_csv'        # 出力CSVの保存先ディレクトリ

# レースデータのcolumns
race_data_columns = [
    'race_id',
    'race_round',
    'race_grade',
    'race_course',
    'weather',
    'track_condition',
    'time',
    'date',
    'location',
    'race_class',
    'total_horse_number',
    'frame_number_first',
    'horse_number_first',
    'frame_number_second',
    'horse_number_second',
    'frame_number_third',
    'horse_number_third',
    'win',
    'show_1',
    'show_2',
    'show_3',
    'bracket_quinella',
    'quinella',
    'quinella_place_1_2',
    'quinella_place_1_3',
    'quinella_place_2_3',
    'exacta',
    'trio',
    'trifecta',
    'ground_index',
    'lap_time',
    'pace_time'
]

# 競走馬データのcolumns
horse_data_columns = [
    'race_id',
    'rank',
    'frame_number',
    'horse_number',
    'horse_id',
    'sex_and_age',
    'burden_weight',
    'rider_id',
    'goal_time',
    'margin',
    'speed_index',
    'rank_path',
    'time_for_3_furlongs',
    'win_odds',
    'popular',
    'horse_weight',
    'remark',
    'affiliation',
    'tamer_id',
    'owner_id'
]


# 複数年分まとめて処理する関数
def make_csv_from_html(start_year=2000):
    for year in range(start_year, now_datetime.year + 1):
        make_csv_from_html_by_year(year)


# 1年分のHTMLを処理してCSVを生成する関数
def make_csv_from_html_by_year(year):
    save_race_csv = os.path.join(CSV_DIR, f'race_{year}.csv')
    horse_race_csv = os.path.join(CSV_DIR, f'horse_{year}.csv')
    # 既にCSVがある場合はスキップ
    if not ((os.path.isfile(save_race_csv)) and (os.path.isfile(horse_race_csv))):
        race_data_dicts = []
        horse_data_dicts = []
        print(f'CSV作成中 ({year})')
        total = 0
        # 各月のHTMLを読み込む
        for month in range(1, 13):
            html_dir = os.path.join(RACE_HTML_DIR, str(year), str(month))
            if os.path.isdir(html_dir):
                file_list = os.listdir(html_dir)
                total += len(file_list)
                print(f'{len(file_list)} 件を追加 ({year}年{month}月)')
                # 各HTMLを処理
                for file_name in file_list:
                    with open(os.path.join(html_dir, file_name), 'r', encoding='euc-jp', errors='replace') as f:
                        html = f.read()
                        race_id = file_name.split('.')[-2]
                        race_dict, horse_dicts = get_race_and_horse_data_by_html(race_id, html)
                        race_data_dicts.append(race_dict)
                        horse_data_dicts.extend(horse_dicts)
        # DataFrameに変換
        race_df = pd.DataFrame(race_data_dicts)
        horse_df = pd.DataFrame(horse_data_dicts)

        # データ中に\xa0（ノーブレークスペース）が残っていないか確認
        for col in race_df.columns:
            matches = race_df[race_df[col].astype(str).str.contains('\xa0', na=False)]
            if not matches.empty:
                print(f'Column: {col}')
                print(matches[[col]])

        # CSVを保存
        race_df.to_csv(save_race_csv, encoding='euc-jp', index=False)
        horse_df.to_csv(horse_race_csv, encoding='euc-jp', index=False)
        print(f'レースデータ (行, 列):\t{race_df.shape}')
        print(f'競走馬データ (行, 列):\t{horse_df.shape}')
        print(f'合計 {total} 件のHTMLをCSV化しました ({year})')
    else:
        print(f'既にCSVがあります ({year})')


# HTMLからレース情報と競走馬情報を抽出する関数
def get_race_and_horse_data_by_html(race_id, html):
    soup = BeautifulSoup(html, 'html.parser')

    race_data = {'race_id': race_id}
    horse_data = []

    # レース概要
    data_intro = soup.find('div', class_='data_intro')
    race_data['race_round'] = data_intro.find('dt').get_text().strip()
    race_data['race_grade'] = data_intro.find('h1').get_text().strip()
    race_details1 = data_intro.find('p').get_text().strip().split('\xa0/\xa0')
    race_data['race_course'] = race_details1[0]
    race_data['weather'] = race_details1[1]
    race_data['track_condition'] = race_details1[2].replace('\xa0', '')
    race_data['time'] = race_details1[3]

    race_details2 = data_intro.find('p', class_='smalltxt').get_text().strip().split(' ')
    race_data['date'] = race_details2[0]
    race_data['location'] = race_details2[1]
    race_data['race_class'] = race_details2[2].replace('\xa0', '')

    result_rows = soup.find('table', class_='race_table_01 nk_tb_common').find_all('tr')
    race_data['total_horse_number'] = len(result_rows) - 1

    # 1〜3着の馬番号・枠番号
    for i in range(1, 4):
        row = result_rows[i].find_all('td')
        race_data[f'frame_number_{"first" if i==1 else "second" if i==2 else "third"}'] = row[1].get_text()
        race_data[f'horse_number_{"first" if i==1 else "second" if i==2 else "third"}'] = row[2].get_text()

    # 払戻金
    pay_back_tables = soup.find_all('table', class_='pay_table_01')
    pay_back1 = pay_back_tables[0].find_all('tr')
    race_data['win'] = pay_back1[0].find('td', class_='txt_r').get_text()
    show = [s for s in pay_back1[1].find('td', class_='txt_r').strings]
    for i in range(3):
        race_data[f'show_{i+1}'] = show[i] if i < len(show) else '0'
    try:
        race_data['bracket_quinella'] = pay_back1[2].find('td', class_='txt_r').get_text()
    except:
        race_data['bracket_quinella'] = '0'
    try:
        race_data['quinella'] = pay_back1[3].find('td', class_='txt_r').get_text()
    except:
        race_data['quinella'] = '0'

    pay_back2 = pay_back_tables[1].find_all('tr')
    try:
        quinella = [s for s in pay_back2[0].find('td', class_='txt_r').strings]
        race_data['quinella_place_1_2'] = quinella[0]
        race_data['quinella_place_1_3'] = quinella[1]
        race_data['quinella_place_2_3'] = quinella[2]
    except:
        race_data['quinella_place_1_2'] = '0'
        race_data['quinella_place_1_3'] = '0'
        race_data['quinella_place_2_3'] = '0'
    try:
        race_data['exacta'] = pay_back2[1].find('td', class_='txt_r').get_text()
    except:
        race_data['exacta'] = '0'
    try:
        race_data['trio'] = pay_back2[2].find('td', class_='txt_r').get_text()
    except:
        race_data['trio'] = '0'
    try:
        race_data['trifecta'] = pay_back2[3].find('td', class_='txt_r').get_text()
    except:
        race_data['trifecta'] = '0'

    # 馬場指数・ラップタイム
    condition_td_element = soup.find_all('table', class_='result_table_02')[0].find_all('tr')[0].find('td')
    race_data['ground_index'] = condition_td_element.get_text(strip=True).split('(')[0].replace('\xa0', '') if condition_td_element else '0'

    lap_tr_element = soup.find_all('table', class_='result_table_02')[2].find_all('tr')
    if lap_tr_element:
        race_data['lap_time'] = lap_tr_element[0].find('td').get_text(strip=True)
        race_data['pace_time'] = lap_tr_element[1].find('td').get_text(strip=True).replace('\xa0', '')
    else:
        race_data['lap_time'] = '0'
        race_data['pace_time'] = '0'

    # 各競走馬データ
    for rank in range(1, len(result_rows)):
        result_row = result_rows[rank].find_all('td')
        horse = {
            'race_id': race_id,
            'rank': result_row[0].get_text(),
            'frame_number': result_row[1].get_text(),
            'horse_number': result_row[2].get_text(),
            'horse_id': result_row[3].find('a').get('href').split('/')[-2],
            'sex_and_age': result_row[4].get_text(),
            'burden_weight': result_row[5].get_text(),
            'rider_id': result_row[6].find('a').get('href').split('/')[-2],
            'goal_time': result_row[7].get_text(),
            'margin': result_row[8].get_text(),
            'speed_index': result_row[9].get_text(),
            'rank_path': result_row[10].get_text(),
            'time_for_3_furlongs': result_row[11].get_text(),
            'win_odds': result_row[12].get_text(),
            'popular': result_row[13].get_text(),
            'horse_weight': result_row[14].get_text(),
            'remark': result_row[17].get_text(),
            'affiliation': result_row[18].contents[0].strip().strip('[]'),
            'tamer_id': result_row[18].find('a').get('href').split('/')[-2],
            'owner_id': result_row[19].find('a').get('href').split('/')[-2] if result_row[19].find('a') else '0'
        }
        horse_data.append(horse)

    return race_data, horse_data


# メイン処理
if __name__ == '__main__':
    print('CSV作成を開始します')
    make_csv_from_html()
