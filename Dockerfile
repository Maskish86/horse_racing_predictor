# Pythonの軽量イメージをベースにする
FROM python:3.10-slim

# 作業ディレクトリを作成
WORKDIR /app

# システムパッケージをインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 必要に応じてpoetryやその他ツールを入れる場合はここに書く

# Python依存ライブラリを先にコピーしてインストール
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install jupyterlab

# JupyterLab用にポートを開放
EXPOSE 8888

# 起動コマンド
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]