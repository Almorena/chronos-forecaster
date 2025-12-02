# Chronos Forecaster - Guida per Sviluppatori

## Panoramica

Questo progetto fornisce una dashboard Streamlit e un API server FastAPI per fare previsioni di serie temporali usando i modelli **Chronos** di Amazon Science.

## Struttura del Progetto

```
chronos-experiment/
├── app.py                    # Dashboard Streamlit principale
├── api_server.py             # API REST FastAPI
├── esempio_previsioni.py     # Script di esempio standalone
├── requirements.txt          # Dipendenze Python
├── Dockerfile                # Container Docker
├── avvia.sh                  # Script di avvio rapido
└── venv/                     # Virtual environment
```

## Modelli Disponibili

### Famiglia Bolt (Veloci - Consigliati)

| Modello | Key API | Parametri | RAM | Velocità |
|---------|---------|-----------|-----|----------|
| Bolt Mini | `bolt-mini` | 9M | ~200MB | Velocissimo |
| Bolt Small | `bolt-small` | 48M | ~400MB | Veloce |
| Bolt Base | `bolt-base` | 205M | ~1GB | Medio |

### Famiglia T5 (Originali)

| Modello | Key API | Parametri | RAM | Velocità |
|---------|---------|-----------|-----|----------|
| T5 Tiny | `t5-tiny` | 8M | ~200MB | Medio |
| T5 Mini | `t5-mini` | 20M | ~400MB | Lento |
| T5 Small | `t5-small` | 46M | ~600MB | Lento |
| T5 Base | `t5-base` | 200M | ~1.5GB | Molto lento |
| T5 Large | `t5-large` | 710M | ~4GB | Molto lento |

## Setup Locale

### 1. Ambiente Virtuale (già creato)

```bash
cd chronos-experiment
source venv/bin/activate
```

### 2. Installare Dipendenze (se necessario)

```bash
pip install -r requirements.txt
```

### 3. Avviare la Dashboard

```bash
streamlit run app.py
# oppure
./avvia.sh
```

Dashboard disponibile su: http://localhost:8501

### 4. Avviare l'API Server

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

API disponibile su: http://localhost:8000
Documentazione Swagger: http://localhost:8000/docs

## API REST

### Endpoints Principali

#### POST /forecast
Genera previsioni per una serie temporale.

**Request:**
```json
{
    "values": [100, 120, 115, 130, 125, 140, 135, 150],
    "prediction_length": 6,
    "model": "bolt-small",
    "num_samples": 20
}
```

**Response:**
```json
{
    "model": "bolt-small",
    "device": "cpu",
    "prediction_length": 6,
    "quantiles": {
        "q10": [145.2, 148.3, ...],
        "q20": [147.1, 150.2, ...],
        "median": [152.3, 155.4, ...],
        "q80": [157.5, 160.6, ...],
        "q90": [159.4, 162.5, ...]
    }
}
```

#### GET /forecast/simple
Endpoint semplificato per query string.

```bash
curl "http://localhost:8000/forecast/simple?values=100,120,115,130&length=6"
```

#### GET /models
Lista tutti i modelli disponibili.

#### GET /health
Health check per monitoring.

### Esempio Chiamata Python

```python
import requests

response = requests.post(
    "http://localhost:8000/forecast",
    json={
        "values": [100, 120, 115, 130, 125, 140],
        "prediction_length": 6,
        "model": "bolt-small"
    }
)

data = response.json()
print(f"Previsione: {data['quantiles']['median']}")
```

## Uso Programmatico

### Uso Base (Bolt)

```python
from chronos import BaseChronosPipeline
import torch

# Carica modello
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-small",
    device_map="cpu"  # o "cuda" o "mps"
)

# Dati storici
values = [100, 120, 115, 130, 125, 140, 135, 150]
context = torch.tensor(values, dtype=torch.float32).unsqueeze(0)

# Previsioni
forecast = pipeline.predict(context, prediction_length=6)

# Estrai quantili (Bolt restituisce 9 quantili)
median = forecast[0, 4, :].numpy()    # Q50
lower_10 = forecast[0, 0, :].numpy()  # Q10
upper_90 = forecast[0, 8, :].numpy()  # Q90
```

### Uso con T5 (Campionamento)

```python
from chronos import BaseChronosPipeline
import torch
import numpy as np

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu"
)

context = torch.tensor([100, 120, 115, 130], dtype=torch.float32).unsqueeze(0)

# T5 genera campioni
forecast = pipeline.predict(context, prediction_length=6, num_samples=20)

# Calcola quantili
samples = forecast[0].numpy()
median = np.percentile(samples, 50, axis=0)
```

## Deploy con Docker

### Build

```bash
docker build -t chronos-forecaster .
```

### Run Dashboard

```bash
docker run -p 8501:8501 chronos-forecaster
```

### Run API Server

```bash
docker run -p 8000:8000 chronos-forecaster api
```

### Run Entrambi

```bash
docker run -p 8501:8501 -p 8000:8000 chronos-forecaster both
```

### Docker Compose

```yaml
version: '3.8'
services:
  dashboard:
    build: .
    ports:
      - "8501:8501"
    command: dashboard

  api:
    build: .
    ports:
      - "8000:8000"
    command: api
```

## GPU Support

### NVIDIA CUDA

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Il sistema rileva automaticamente la GPU.

### Apple Silicon (MPS)

Funziona automaticamente su Mac M1/M2/M3 con PyTorch >= 2.0.

### Verifica Device

```python
import torch

if torch.cuda.is_available():
    print("CUDA disponibile")
elif torch.backends.mps.is_available():
    print("Apple MPS disponibile")
else:
    print("Solo CPU")
```

## Differenze Bolt vs T5

| Aspetto | Bolt | T5 |
|---------|------|-----|
| **Output** | 9 quantili fissi | Campioni probabilistici |
| **Velocità** | Fino a 250x più veloce | Base di confronto |
| **Flessibilità** | Quantili predefiniti | Qualsiasi quantile calcolabile |
| **Uso** | Produzione | Ricerca, analisi |

## Limitazioni Note

1. **Univariato**: Una sola serie temporale alla volta
2. **Frequenza fissa**: Meglio con dati regolari (giornalieri, settimanali, mensili)
3. **Prima esecuzione**: Lenta per download modello
4. **Memoria**: Modelli grandi richiedono molta RAM
5. **Prediction length**: Max consigliato 64 periodi

## Troubleshooting

### "CUDA out of memory"

- Usa un modello più piccolo
- Passa a CPU: `device_map="cpu"`

### "Model not found"

- Verifica connessione internet (primo download)
- Controlla nome modello esatto

### Previsioni poco accurate

- Aumenta dati storici (min 20-30 punti)
- Usa modello più grande (bolt-base o t5-base)

## Risorse

- [Paper Chronos](https://arxiv.org/abs/2403.07815)
- [GitHub Amazon](https://github.com/amazon-science/chronos-forecasting)
- [Hugging Face Models](https://huggingface.co/amazon)

## Contatti

Per problemi o domande sul codice, contattare il team di sviluppo.
