FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY astroimg/ astroimg/

RUN pip install --no-cache-dir .

COPY demo.py .

CMD ["python", "demo.py"]
