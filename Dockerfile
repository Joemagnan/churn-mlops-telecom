FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY models ./models
# monitoring/ est créé ici pour appartenir à apiuser : un volume monté dessus
# hérite de ces droits, sinon l'écriture du journal d'inférences est refusée.
RUN useradd -m apiuser && mkdir -p /app/monitoring && chown -R apiuser:apiuser /app
USER apiuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["uvicorn","src.api:app","--host","0.0.0.0","--port","8000"]
