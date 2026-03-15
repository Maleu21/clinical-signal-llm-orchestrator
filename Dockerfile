FROM python:3.11-slim

WORKDIR /app

# Installer torch en premier (cache stable, jamais invalidé)
RUN pip install --no-cache-dir --timeout 1000 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2

# Copier et installer le reste
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 1000 -r requirements.txt

COPY . .

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]