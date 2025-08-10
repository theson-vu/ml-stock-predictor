# Dockerfile
FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=.
# default entrypoint runs training (change in prod)
CMD ["python", "src/model.py"]