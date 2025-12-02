# ============================================
# CHRONOS FORECASTER - Dockerfile
# ============================================
# Multi-stage build per immagine ottimizzata
#
# Build: docker build -t chronos-forecaster .
# Run Dashboard: docker run -p 8501:8501 chronos-forecaster
# Run API: docker run -p 8000:8000 chronos-forecaster api

FROM python:3.11-slim as base

# Variabili ambiente
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dipendenze di sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Directory di lavoro
WORKDIR /app

# ============================================
# Stage: Dependencies
# ============================================
FROM base as dependencies

# Copia requirements e installa dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# Stage: Production
# ============================================
FROM dependencies as production

# Copia codice applicazione
COPY app.py .
COPY api_server.py .
COPY avvia.sh .

# Permessi
RUN chmod +x avvia.sh

# Pre-scarica il modello di default (opzionale, aumenta dimensione immagine)
# RUN python -c "from chronos import BaseChronosPipeline; BaseChronosPipeline.from_pretrained('amazon/chronos-bolt-small')"

# Esponi porte
EXPOSE 8501 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || curl -f http://localhost:8000/health || exit 1

# Script di avvio
COPY <<EOF /app/entrypoint.sh
#!/bin/bash
set -e

if [ "\$1" = "api" ]; then
    echo "Avvio API Server su porta 8000..."
    exec uvicorn api_server:app --host 0.0.0.0 --port 8000
elif [ "\$1" = "both" ]; then
    echo "Avvio Dashboard e API..."
    uvicorn api_server:app --host 0.0.0.0 --port 8000 &
    exec streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
else
    echo "Avvio Dashboard Streamlit su porta 8501..."
    exec streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
fi
EOF

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["dashboard"]
