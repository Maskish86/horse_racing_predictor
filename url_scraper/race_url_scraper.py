from pathlib import Path
import datetime
import pytz
import re
import os

from selenium import webdriver
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

from utils.logger import setup_logger

logger = setup_logger("race_url_scraper")
now_datetime = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

RACE_URL_DIR = Path("data/race_url")
RACE_URL_DIR.mkdir(parents=True, exist_ok=True) 
URL = 'https://db.netkeiba.com/?pid=race_search_detail'
WAIT_SECOND = 5


def get_race_url(start_year=2001, start_month=1):
    options = Options()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(10)

    def generate_year_month_ranges():
        ranges = []

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

        if start_year < now_datetime.year:
            # 過去の年
            for year in range(start_year, now_datetime.year):
                for month in range(1, 13):
                    if year == start_year and month < start_month:
                        continue
                    ranges.append((year, month))
            # 今年
            end_month = now_datetime.month if not sunday_occurred else now_datetime.month + 1
            for month in range(1, end_month):
                ranges.append((now_datetime.year, month))
        elif start_year == now_datetime.year:
            end_month = now_datetime.month if not sunday_occurred else now_datetime.month + 1
            for month in range(start_month, end_month):
                ranges.append((now_datetime.year, month))
        else:
            raise ValueError("start_year cannot be in the future.")

        return ranges

    for year, month in generate_year_month_ranges():
        logger.info(f"Getting race URLs for {year}-{month}")
        get_race_url_by_year_and_mon(driver, year, month)


    driver.close()
    driver.quit()


def get_race_url_by_year_and_mon(driver, year, month):
    race_url_file =  RACE_URL_DIR / f"{year}-{month}.txt"

    wait = WebDriverWait(driver, 10)
    driver.get(URL)
    wait.until(EC.presence_of_element_located((By.NAME, 'start_year')))

    # 期間を選択
    start_year_select = Select(driver.find_element(By.NAME, 'start_year'))
    start_year_select.select_by_value(str(year))
    start_mon_select = Select(driver.find_element(By.NAME, 'start_mon'))
    start_mon_select.select_by_value(str(month))
    end_year_select = Select(driver.find_element(By.NAME, 'end_year'))
    end_year_select.select_by_value(str(year))
    end_mon_select = Select(driver.find_element(By.NAME, 'end_mon'))
    end_mon_select.select_by_value(str(month))

    for i in range(1, 11):
        terms = driver.find_element(By.ID, f'check_Jyo_{str(i).zfill(2)}')
        driver.execute_script("arguments[0].click();", terms)

    list_select = Select(driver.find_element(By.NAME, 'list'))
    list_select.select_by_value('100')

    frm = driver.find_element(By.CSS_SELECTOR, '#db_search_detail_form > form')
    frm.submit()
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'race_table_01')))

    total_num_and_now_num = driver.find_element(By.XPATH, "//*[@id='contents_liquid']/div[1]/div[2]").text
    match = re.search(r'(.*)件中', total_num_and_now_num)
    if match:
        total_num = int(match.group(1).strip())
    else:
        raise ValueError(f"Could not extract total number from text: {total_num_and_now_num}")

    pre_url_num = 0
    if os.path.isfile(race_url_file):
        with open(race_url_file, mode='r') as f:
            pre_url_num = len(f.readlines())

    if total_num != pre_url_num:
        with open(race_url_file, mode='w') as f:
            total_written = 0
            while True:
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'race_table_01')))
                rows = driver.find_element(By.CLASS_NAME, 'race_table_01').find_elements(By.TAG_NAME, 'tr')
                for row in range(1, len(rows)):
                    race_href = rows[row].find_elements(By.TAG_NAME, 'td')[4]\
                        .find_element(By.TAG_NAME, 'a').get_attribute('href')
                    f.write(race_href+'\n')
                    total_written += 1
                try:
                    target = driver.find_elements(By.LINK_TEXT, '次')[0]
                    driver.execute_script('arguments[0].click();', target)  # javascriptでクリック処理
                except IndexError:
                    break
        logger.info(f"Saved {total_written} race URLs for {year}-{month}")
    else:
        logger.info(f"Already have {pre_url_num} race URLs for {year}-{month}, skipping")


if __name__ == '__main__':
    logger.info("Start scraping race URLs...")
    get_race_url()
