import datetime
import pytz
import requests
import os

now_datetime = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
RACE_URL_DIR = 'race_url'
RACE_HTML_DIR = 'race_html'

ID = ''  # your id on netkeiba 
PASSWORD = ''  # your password on netkeiba

def my_makedirs(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def get_race_html():
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
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': login_url,
    }
    login_response =  session.post(login_url, data=payload, headers=headers)
    print(login_response)
    for year in range(2006, now_datetime.year):
        for month in range(1, 13):
            get_race_html_by_year_and_month(year, month, session)
    for year in range(now_datetime.year, now_datetime.year+1):
        for month in range(1, now_datetime.month+1):
            get_race_html_by_year_and_month(year, month, session)
    session.close()


def get_race_html_by_year_and_month(year, month, session):
    login_url = 'https://regist.netkeiba.com/account/?pid=login'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': login_url,
    }
    with open(f'{RACE_URL_DIR}/{year}-{month}.txt', 'r') as f:
        save_dir = os.path.join(RACE_HTML_DIR, str(year), str(month))
        my_makedirs(save_dir)
        urls = f.read().splitlines()

        file_list = os.listdir(save_dir)

        if len(urls) != len(file_list):
            print(f'getting htmls ({year} {month})')
            for url in urls:
                race_id = url.split('/')[-2]
                save_file_path = os.path.join(save_dir, f'{race_id}.html')
                if not os.path.isfile(save_file_path):
                    response = session.get(url, headers=headers)
                    response.encoding = response.apparent_encoding
                    html = response.text
                    with open(save_file_path, 'w', encoding='euc-jp', errors='replace') as file:
                        file.write(html)
            print(f'saved {len(urls)} htmls ({year} {month})')
        else:
            print(f'already have {len(urls)} htmls ({year} {month})')


if __name__ == '__main__':
    print('start get race html!')
    get_race_html()
