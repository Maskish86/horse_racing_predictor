from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import requests
import os
from tqdm import tqdm

from utils.logger import setup_logger
logger = setup_logger('training_html_downloader')

load_dotenv()

ID = os.getenv('NETKEIBA_ID')
PASSWORD = os.getenv('NETKEIBA_PASSWORD')

SAVE_DIR = Path('horse_training_html')
SAVE_DIR.mkdir(exist_ok=True)
PARQUET_DIR = Path('yearly_parquet')

def get_training_html(start_year=2001, start_month=1):
    logger.info('Starting training HTML downloader...')
    horse_df = pd.read_parquet(PARQUET_DIR / 'horse_features.parquet')
    filtered_df = horse_df[horse_df['datetime'] > f'{start_year}-{start_month:02d}-01']
    id_list = sorted(set(filtered_df['horse_id'].dropna().astype(str)))
    logger.info(f'🎯 Found {len(id_list)} unique horse IDs to download.')

    session = requests.Session()
    payload = {
        'pid': 'login',
        'action': 'auth',
        'return_url2': '',
        'mem_tp': '',
        'login_id': ID,
        'pswd': PASSWORD
    }
    login_url = 'https://regist.netkeiba.com/account/?pid=login'
    login_response = session.post(login_url, data=payload)
    if login_response.status_code != 200:
        logger.error(f'Login failed with status: {login_response.status_code}')
        return
    logger.info('Login successful.')

    for horse_id in tqdm(id_list, desc='Downloading horse training pages'):
        save_path = SAVE_DIR / f'{horse_id[:4]}/{horse_id}.html'
        if save_path.exists():
            logger.debug(f'Skipping {horse_id}: already exists.')
            continue

        url = f'https://db.netkeiba.com/?pid=horse_training&id={horse_id}'
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            html = response.text

            if '調教タイムが存在しませんでした' in html:
                logger.debug(f'{horse_id}: No training data.')
                continue

            with open(save_path, 'w', encoding='euc-jp', errors='replace') as file:
                file.write(html)
        except Exception as e:
            logger.warning(f'Failed to fetch {horse_id}: {e}')
            continue

    logger.info('Training HTML download completed.')

if __name__ == '__main__':
    get_training_html()