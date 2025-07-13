FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
EXPOSE 8888

# Default CMD (Flask server for Cloud Run)
CMD ["python", "app.py"]