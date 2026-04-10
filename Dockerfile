FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends cron tzdata && rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY locales/ ./locales/
COPY crontab /etc/cron.d/news-digest

RUN chmod 0644 /etc/cron.d/news-digest && \
    crontab /etc/cron.d/news-digest && \
    mkdir -p /app/logs && \
    touch /var/log/cron.log

CMD cron && tail -f /var/log/cron.log
