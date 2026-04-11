FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gradio>=4.0.0

COPY . .

EXPOSE 7860

CMD ["python", "-m", "src.web.app", "--port", "7860"]
