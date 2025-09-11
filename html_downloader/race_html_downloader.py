from pathlib import Path
import datetime
from dotenv import load_dotenv
import pytz
import requests
import os

from utils.logger import setup_logger

logger = setup_logger("race_url_scraper")
now_datetime = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

now_datetime = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
RACE_URL_DIR = Path("data/race_url")
RACE_HTML_DIR = Path('data/race_html')
RACE_HTML_DIR.mkdir(exist_ok=True)

load_dotenv()

ID = os.getenv("NETKEIBA_ID")
PASSWORD = os.getenv("NETKEIBA_PASSWORD")

if not ID or not PASSWORD:
    logger.warning("NETKEIBA_ID or NETKEIBA_PASSWORD not set in .env")


def get_race_html(start_year=2001, start_month=1):
    logger.info("Start downloading race HTMLs...")
    # セッション開始
    session = requests.Session()

    # ログイン情報
    payload = {
        'pid': 'login',
        'action': 'auth',
        'return_url2': '',
        'mem_tp': '',
        'login_id': ID,
        'pswd': PASSWORD
    }
    login_url = 'https://regist.netkeiba.com/account/?pid=login'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': login_url,
    }

    # ログイン
    try:
        login_response = session.post(login_url, data=payload, headers=headers)
        login_response.raise_for_status()
        logger.info(f"Login success: status={login_response.status_code}")
    except Exception as e:
        logger.error(f"Login failed: {e}")
        return


    # 今月の最初の日
    first_day_of_month = now_datetime.replace(day=1)
    today = now_datetime

    # 今月に日曜があったか確認
    sunday_occurred = False
    for i in range(0, today.day):
        day = first_day_of_month + datetime.timedelta(days=i)
        if day.weekday() == 6:
            sunday_occurred = True
            break

    # 年月リストを作成
    ranges = []

    if start_year < now_datetime.year:
        for year in range(start_year, now_datetime.year):
            for month in range(1, 13):
                if year == start_year and month < start_month:
                    continue
                ranges.append((year, month))
        end_month = now_datetime.month if not sunday_occurred else now_datetime.month + 1
        for month in range(1, end_month):
            ranges.append((now_datetime.year, month))
    elif start_year == now_datetime.year:
        end_month = now_datetime.month if not sunday_occurred else now_datetime.month + 1
        for month in range(start_month, end_month):
            ranges.append((now_datetime.year, month))
    else:
        logger.error("start_year cannot be in the future.")

    # HTML取得
    for year, month in ranges:
        get_race_html_by_year_and_month(year, month, session)

    session.close()

def get_race_html_by_year_and_month(year, month, session):
    login_url = 'https://regist.netkeiba.com/account/?pid=login'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': login_url,
    }

    # URLファイルを読む
    url_file = RACE_URL_DIR / f"{year}-{month}.txt"
    if not os.path.isfile(url_file):
        logger.info(f"URL file not found: {url_file}")
        return

    with open(url_file, 'r') as f:
        urls = f.read().splitlines()

    # 保存ディレクトリ作成
    save_dir = RACE_HTML_DIR / os.path.join(str(year), str(month))
    os.makedirs(save_dir, exist_ok=True)

    # 既に存在するrace_idセット
    existing_files = {fname.split(".")[0] for fname in os.listdir(save_dir) if fname.endswith(".html")}

    # 取得対象を抽出
    urls_to_download = []
    for url in urls:
        race_id = url.strip().split("/")[-2]
        if race_id not in existing_files:
            urls_to_download.append((url, race_id))

    if not urls_to_download:
        logger.info(f"[{year}-{month}] Already have {len(urls)} HTML files. Skipping.")
        return

    logger.info(f"[{year}-{month}] Downloading {len(urls_to_download)} new race HTMLs...")

    # HTMLをダウンロード
    success_count = 0
    for url, race_id in urls_to_download:
        save_file_path = os.path.join(save_dir, f"{race_id}.html")
        try:
            response = session.get(url, headers=headers)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            with open(save_file_path, 'w', encoding='euc-jp', errors='replace') as file:
                file.write(response.text)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")

    logger.info(f"[{year}-{month}] Saved {success_count}/{len(urls_to_download)} race HTMLs.")


if __name__ == '__main__':
    get_race_html()
