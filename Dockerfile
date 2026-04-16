FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY trading_system /app/trading_system
COPY config /app/config
COPY examples /app/examples
COPY main.py /app/main.py

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .[system]

CMD ["python", "main.py", "--config", "config/trading_system.example.toml"]
