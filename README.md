# Chronos Forecaster Pro by Almo

Dashboard interattiva per previsioni di serie temporali usando **Chronos AI** di Amazon Science.

## Demo Live

Prova l'app: [chronos-forecaster.streamlit.app](https://chronos-forecaster.streamlit.app)

## Caratteristiche

- **8 modelli AI** disponibili (Bolt e T5)
- **Zero-shot learning** - funziona senza addestramento
- **Previsioni probabilistiche** con intervalli di confidenza
- **GPU support** automatico (CUDA, Apple MPS)
- **Export** in CSV e JSON
- **API REST** inclusa

## Modelli Disponibili

| Famiglia | Modello | Parametri | Velocita |
|----------|---------|-----------|----------|
| Bolt | bolt-mini | 9M | Velocissimo |
| Bolt | bolt-small | 48M | Veloce |
| Bolt | bolt-base | 205M | Medio |
| T5 | t5-tiny | 8M | Medio |
| T5 | t5-mini | 20M | Lento |
| T5 | t5-small | 46M | Lento |
| T5 | t5-base | 200M | Molto lento |
| T5 | t5-large | 710M | Molto lento |

## Installazione Locale

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/chronos-forecaster.git
cd chronos-forecaster

# Virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oppure: venv\Scripts\activate  # Windows

# Dipendenze
pip install -r requirements.txt

# Avvio
streamlit run app.py
```

## API REST

```bash
# Avvia API server
uvicorn api_server:app --host 0.0.0.0 --port 8000

# Documentazione Swagger
open http://localhost:8000/docs
```

### Esempio chiamata

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
print(response.json())
```

## Docker

```bash
# Build
docker build -t chronos-forecaster .

# Run Dashboard
docker run -p 8501:8501 chronos-forecaster

# Run API
docker run -p 8000:8000 chronos-forecaster api
```

## Tecnologie

- [Chronos](https://github.com/amazon-science/chronos-forecasting) - Amazon Science
- [Streamlit](https://streamlit.io/) - Dashboard
- [FastAPI](https://fastapi.tiangolo.com/) - API REST
- [Plotly](https://plotly.com/) - Grafici interattivi

## Licenza

MIT License
