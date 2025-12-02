"""
CHRONOS FORECAST API SERVER
============================
API REST per previsioni con Chronos.
Avvia con: uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import torch
import numpy as np
from contextlib import asynccontextmanager

# Modelli disponibili
AVAILABLE_MODELS = {
    # Bolt (veloci)
    "bolt-mini": "amazon/chronos-bolt-mini",
    "bolt-small": "amazon/chronos-bolt-small",
    "bolt-base": "amazon/chronos-bolt-base",
    # T5 (originali)
    "t5-tiny": "amazon/chronos-t5-tiny",
    "t5-mini": "amazon/chronos-t5-mini",
    "t5-small": "amazon/chronos-t5-small",
    "t5-base": "amazon/chronos-t5-base",
    "t5-large": "amazon/chronos-t5-large",
}

# Cache dei modelli caricati
loaded_models = {}

def get_device():
    """Rileva il miglior dispositivo disponibile"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_model(model_key: str):
    """Carica un modello (con cache)"""
    if model_key in loaded_models:
        return loaded_models[model_key]

    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Modello non valido: {model_key}")

    from chronos import BaseChronosPipeline

    model_name = AVAILABLE_MODELS[model_key]
    device = get_device()

    pipeline = BaseChronosPipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float32
    )

    loaded_models[model_key] = pipeline
    return pipeline

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Precarica il modello di default all'avvio"""
    print("Caricamento modello di default (bolt-small)...")
    load_model("bolt-small")
    print(f"Modello caricato su {get_device().upper()}")
    yield
    # Cleanup
    loaded_models.clear()

# Inizializza FastAPI
app = FastAPI(
    title="Chronos Forecast API",
    description="API REST per previsioni di serie temporali con Chronos AI",
    version="2.0.0",
    lifespan=lifespan
)

# CORS per permettere chiamate da qualsiasi origine
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# MODELLI PYDANTIC
# ============================================

class ForecastRequest(BaseModel):
    """Richiesta di previsione"""
    values: List[float] = Field(
        ...,
        description="Lista di valori storici (almeno 10 elementi consigliati)",
        min_items=3,
        example=[100, 120, 115, 130, 125, 140, 135, 150, 148, 155]
    )
    prediction_length: int = Field(
        default=6,
        ge=1,
        le=64,
        description="Numero di periodi da prevedere"
    )
    model: str = Field(
        default="bolt-small",
        description="Modello da usare: bolt-mini, bolt-small, bolt-base, t5-tiny, t5-mini, t5-small, t5-base, t5-large"
    )
    num_samples: int = Field(
        default=20,
        ge=10,
        le=100,
        description="Numero di campioni per modelli T5 (ignorato per Bolt)"
    )

class ForecastResponse(BaseModel):
    """Risposta con le previsioni"""
    model: str
    device: str
    prediction_length: int
    quantiles: dict = Field(
        description="Previsioni per ogni quantile (q10, q20, q30, q40, q50/median, q60, q70, q80, q90)"
    )

class ModelInfo(BaseModel):
    """Informazioni su un modello"""
    key: str
    full_name: str
    family: str
    loaded: bool

class HealthResponse(BaseModel):
    """Risposta health check"""
    status: str
    device: str
    models_loaded: List[str]
    available_models: List[str]

# ============================================
# ENDPOINTS
# ============================================

@app.get("/", tags=["Info"])
async def root():
    """Informazioni sull'API"""
    return {
        "name": "Chronos Forecast API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Verifica stato dell'API"""
    return HealthResponse(
        status="healthy",
        device=get_device(),
        models_loaded=list(loaded_models.keys()),
        available_models=list(AVAILABLE_MODELS.keys())
    )

@app.get("/models", response_model=List[ModelInfo], tags=["Info"])
async def list_models():
    """Lista tutti i modelli disponibili"""
    return [
        ModelInfo(
            key=key,
            full_name=name,
            family="bolt" if "bolt" in key else "t5",
            loaded=key in loaded_models
        )
        for key, name in AVAILABLE_MODELS.items()
    ]

@app.post("/forecast", response_model=ForecastResponse, tags=["Forecast"])
async def forecast(request: ForecastRequest):
    """
    Genera previsioni per una serie temporale.

    **Modelli disponibili:**
    - `bolt-mini`, `bolt-small`, `bolt-base`: Veloci, restituiscono quantili direttamente
    - `t5-tiny`, `t5-mini`, `t5-small`, `t5-base`, `t5-large`: Generano campioni probabilistici

    **Esempio:**
    ```json
    {
        "values": [100, 120, 115, 130, 125, 140],
        "prediction_length": 6,
        "model": "bolt-small"
    }
    ```
    """
    try:
        # Valida modello
        if request.model not in AVAILABLE_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Modello non valido. Usa uno di: {list(AVAILABLE_MODELS.keys())}"
            )

        # Carica modello
        pipeline = load_model(request.model)

        # Prepara input
        context = torch.tensor(request.values, dtype=torch.float32).unsqueeze(0)

        # Genera previsioni
        if "bolt" in request.model:
            # Modelli Bolt restituiscono quantili direttamente
            forecast = pipeline.predict(context, prediction_length=request.prediction_length)

            quantiles = {
                "q10": forecast[0, 0, :].cpu().numpy().tolist(),
                "q20": forecast[0, 1, :].cpu().numpy().tolist(),
                "q30": forecast[0, 2, :].cpu().numpy().tolist(),
                "q40": forecast[0, 3, :].cpu().numpy().tolist(),
                "median": forecast[0, 4, :].cpu().numpy().tolist(),
                "q60": forecast[0, 5, :].cpu().numpy().tolist(),
                "q70": forecast[0, 6, :].cpu().numpy().tolist(),
                "q80": forecast[0, 7, :].cpu().numpy().tolist(),
                "q90": forecast[0, 8, :].cpu().numpy().tolist(),
            }
        else:
            # Modelli T5 generano campioni
            forecast = pipeline.predict(
                context,
                prediction_length=request.prediction_length,
                num_samples=request.num_samples
            )

            samples = forecast[0].cpu().numpy()

            quantiles = {
                "q10": np.percentile(samples, 10, axis=0).tolist(),
                "q20": np.percentile(samples, 20, axis=0).tolist(),
                "q30": np.percentile(samples, 30, axis=0).tolist(),
                "q40": np.percentile(samples, 40, axis=0).tolist(),
                "median": np.percentile(samples, 50, axis=0).tolist(),
                "q60": np.percentile(samples, 60, axis=0).tolist(),
                "q70": np.percentile(samples, 70, axis=0).tolist(),
                "q80": np.percentile(samples, 80, axis=0).tolist(),
                "q90": np.percentile(samples, 90, axis=0).tolist(),
            }

        return ForecastResponse(
            model=request.model,
            device=get_device(),
            prediction_length=request.prediction_length,
            quantiles=quantiles
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast/simple", tags=["Forecast"])
async def forecast_simple(
    values: str = Query(..., description="Valori separati da virgola, es: 100,120,115,130"),
    length: int = Query(default=6, ge=1, le=64, description="Periodi da prevedere"),
    model: str = Query(default="bolt-small", description="Modello da usare")
):
    """
    Endpoint semplificato per previsioni rapide via GET.

    **Esempio:**
    ```
    GET /forecast/simple?values=100,120,115,130,125,140&length=6
    ```
    """
    try:
        values_list = [float(v.strip()) for v in values.split(",")]

        request = ForecastRequest(
            values=values_list,
            prediction_length=length,
            model=model
        )

        return await forecast(request)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Valori non validi: {e}")

@app.post("/load/{model_key}", tags=["Models"])
async def preload_model(model_key: str):
    """
    Precarica un modello in memoria.

    Utile per ridurre la latenza della prima richiesta.
    """
    if model_key not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Modello non valido. Usa uno di: {list(AVAILABLE_MODELS.keys())}"
        )

    try:
        load_model(model_key)
        return {
            "status": "loaded",
            "model": model_key,
            "device": get_device()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
