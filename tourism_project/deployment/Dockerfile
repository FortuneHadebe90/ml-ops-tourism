FROM python:3.11-slim

WORKDIR /app

# Writable locations for the non-root user Hugging Face Spaces runs as
ENV HOME=/tmp \
    HF_HOME=/tmp/hf \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false"]
