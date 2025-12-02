"""
CHRONOS FORECASTER - Dashboard Completa di Previsione
=====================================================
Dashboard interattiva per previsioni di serie temporali con tutti i modelli Chronos AI
Versione 2.0 - Completa con tutti i modelli, GPU support, API REST
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import torch
import io
import json
import time

# Configurazione pagina
st.set_page_config(
    page_title="Chronos Forecaster Pro by Almo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizzato
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .speed-fast { color: #28a745; }
    .speed-medium { color: #ffc107; }
    .speed-slow { color: #dc3545; }
    .gpu-badge {
        background-color: #76b900;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
    }
    .cpu-badge {
        background-color: #0066cc;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# CONFIGURAZIONE MODELLI
# ============================================
MODELLI_DISPONIBILI = {
    # Modelli Bolt (più veloci, quantili diretti)
    "amazon/chronos-bolt-mini": {
        "nome": "Chronos Bolt Mini",
        "tipo": "bolt",
        "dimensione": "~20MB",
        "parametri": "9M",
        "velocita": 5,
        "accuratezza": 3,
        "descrizione": "Il più veloce, ideale per test rapidi",
        "quantili": True
    },
    "amazon/chronos-bolt-small": {
        "nome": "Chronos Bolt Small",
        "tipo": "bolt",
        "dimensione": "~50MB",
        "parametri": "48M",
        "velocita": 4,
        "accuratezza": 4,
        "descrizione": "Ottimo equilibrio velocità/accuratezza",
        "quantili": True
    },
    "amazon/chronos-bolt-base": {
        "nome": "Chronos Bolt Base",
        "tipo": "bolt",
        "dimensione": "~200MB",
        "parametri": "205M",
        "velocita": 3,
        "accuratezza": 5,
        "descrizione": "Alta accuratezza, raccomandato per produzione",
        "quantili": True
    },
    # Modelli T5 originali (più lenti, campionamento probabilistico)
    "amazon/chronos-t5-tiny": {
        "nome": "Chronos T5 Tiny",
        "tipo": "t5",
        "dimensione": "~30MB",
        "parametri": "8M",
        "velocita": 3,
        "accuratezza": 3,
        "descrizione": "Versione minimale per dispositivi limitati",
        "quantili": False
    },
    "amazon/chronos-t5-mini": {
        "nome": "Chronos T5 Mini",
        "tipo": "t5",
        "dimensione": "~80MB",
        "parametri": "20M",
        "velocita": 2,
        "accuratezza": 3,
        "descrizione": "Compatto ma efficace",
        "quantili": False
    },
    "amazon/chronos-t5-small": {
        "nome": "Chronos T5 Small",
        "tipo": "t5",
        "dimensione": "~150MB",
        "parametri": "46M",
        "velocita": 2,
        "accuratezza": 4,
        "descrizione": "Buon compromesso qualità/risorse",
        "quantili": False
    },
    "amazon/chronos-t5-base": {
        "nome": "Chronos T5 Base",
        "tipo": "t5",
        "dimensione": "~400MB",
        "parametri": "200M",
        "velocita": 1,
        "accuratezza": 5,
        "descrizione": "Alta qualità, richiede più tempo",
        "quantili": False
    },
    "amazon/chronos-t5-large": {
        "nome": "Chronos T5 Large",
        "tipo": "t5",
        "dimensione": "~1.2GB",
        "parametri": "710M",
        "velocita": 1,
        "accuratezza": 5,
        "descrizione": "Massima accuratezza, richiede molta RAM",
        "quantili": False
    }
}

# ============================================
# FUNZIONI UTILITY
# ============================================

def get_device():
    """Rileva automaticamente il miglior dispositivo disponibile"""
    if torch.cuda.is_available():
        return "cuda", "GPU NVIDIA"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps", "Apple Silicon GPU"
    else:
        return "cpu", "CPU"

@st.cache_resource
def load_model(model_name, device="auto"):
    """Carica il modello Chronos con supporto GPU automatico"""
    from chronos import BaseChronosPipeline

    # Determina device
    if device == "auto":
        device_map = "cuda" if torch.cuda.is_available() else "cpu"
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_map = "mps"
    else:
        device_map = device

    pipeline = BaseChronosPipeline.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch.float32
    )
    return pipeline, device_map

def generate_forecast_bolt(pipeline, data, prediction_length):
    """Genera previsioni con modelli Bolt (quantili diretti)"""
    context = torch.tensor(data.values, dtype=torch.float32).unsqueeze(0)
    forecast = pipeline.predict(context, prediction_length=prediction_length)

    # Bolt restituisce 9 quantili: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    return {
        'q10': forecast[0, 0, :].cpu().numpy(),
        'q20': forecast[0, 1, :].cpu().numpy(),
        'q30': forecast[0, 2, :].cpu().numpy(),
        'q40': forecast[0, 3, :].cpu().numpy(),
        'median': forecast[0, 4, :].cpu().numpy(),  # q50
        'q60': forecast[0, 5, :].cpu().numpy(),
        'q70': forecast[0, 6, :].cpu().numpy(),
        'q80': forecast[0, 7, :].cpu().numpy(),
        'q90': forecast[0, 8, :].cpu().numpy(),
    }

def generate_forecast_t5(pipeline, data, prediction_length, num_samples=20):
    """Genera previsioni con modelli T5 (campionamento probabilistico)"""
    context = torch.tensor(data.values, dtype=torch.float32).unsqueeze(0)

    # T5 genera campioni, poi calcoliamo i quantili
    forecast = pipeline.predict(context, prediction_length=prediction_length, num_samples=num_samples)

    # forecast shape: [batch, samples, prediction_length]
    samples = forecast[0].cpu().numpy()  # [samples, prediction_length]

    return {
        'q10': np.percentile(samples, 10, axis=0),
        'q20': np.percentile(samples, 20, axis=0),
        'q30': np.percentile(samples, 30, axis=0),
        'q40': np.percentile(samples, 40, axis=0),
        'median': np.percentile(samples, 50, axis=0),
        'q60': np.percentile(samples, 60, axis=0),
        'q70': np.percentile(samples, 70, axis=0),
        'q80': np.percentile(samples, 80, axis=0),
        'q90': np.percentile(samples, 90, axis=0),
        'samples': samples  # Mantieni i campioni originali
    }

def generate_forecast(pipeline, data, prediction_length, model_info, num_samples=20):
    """Wrapper che sceglie la funzione corretta in base al tipo di modello"""
    if model_info['tipo'] == 'bolt':
        return generate_forecast_bolt(pipeline, data, prediction_length)
    else:
        return generate_forecast_t5(pipeline, data, prediction_length, num_samples)

def create_sample_data(tipo="vendite"):
    """Crea dati di esempio per testing"""
    np.random.seed(42)
    n = 36  # 3 anni

    if tipo == "vendite":
        dates = pd.date_range(start='2021-01-01', periods=n, freq='ME')
        trend = np.linspace(100, 180, n)
        seasonal = 25 * np.sin(np.linspace(0, 6*np.pi, n))
        noise = np.random.normal(0, 8, n)
        values = trend + seasonal + noise
        col_name = "vendite"

    elif tipo == "temperatura":
        dates = pd.date_range(start='2021-01-01', periods=n, freq='ME')
        seasonal = 15 * np.cos(np.linspace(0, 6*np.pi, n)) + 15
        noise = np.random.normal(0, 2, n)
        values = seasonal + noise
        col_name = "temperatura"

    elif tipo == "visite_web":
        dates = pd.date_range(start='2021-01-01', periods=n, freq='ME')
        trend = np.linspace(5000, 15000, n)
        seasonal = 2000 * np.sin(np.linspace(0, 6*np.pi, n))
        noise = np.random.normal(0, 500, n)
        values = trend + seasonal + noise
        col_name = "visite"

    elif tipo == "energia":
        dates = pd.date_range(start='2021-01-01', periods=n, freq='ME')
        trend = np.linspace(1000, 1200, n)
        seasonal = 300 * np.cos(np.linspace(0, 6*np.pi, n))  # Picco in inverno
        noise = np.random.normal(0, 50, n)
        values = trend + seasonal + noise
        col_name = "consumo_kwh"

    elif tipo == "stock":
        dates = pd.date_range(start='2021-01-01', periods=n, freq='ME')
        # Random walk con drift
        returns = np.random.normal(0.01, 0.05, n)
        values = 100 * np.cumprod(1 + returns)
        col_name = "prezzo"

    return pd.DataFrame({
        'data': dates,
        col_name: values
    })

def render_speed_stars(speed):
    """Renderizza stelle per la velocità"""
    return "⚡" * speed + "·" * (5 - speed)

def render_accuracy_stars(accuracy):
    """Renderizza stelle per l'accuratezza"""
    return "⭐" * accuracy + "☆" * (5 - accuracy)

# ============================================
# SIDEBAR - NAVIGAZIONE E CONFIGURAZIONE
# ============================================
with st.sidebar:
    st.image("https://raw.githubusercontent.com/amazon-science/chronos-forecasting/main/figures/main-figure.png", use_container_width=True)
    st.title("⚙️ Configurazione")

    # Selezione pagina
    pagina = st.radio(
        "📑 Navigazione",
        ["🏠 Home", "📊 Previsioni", "🤖 Modelli", "🔌 API", "📚 Documentazione", "❓ FAQ"],
        index=0
    )

    st.divider()

    # Info dispositivo
    device, device_name = get_device()
    if device == "cuda":
        st.markdown(f'<span class="gpu-badge">🎮 {device_name}</span>', unsafe_allow_html=True)
    elif device == "mps":
        st.markdown(f'<span class="gpu-badge">🍎 {device_name}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="cpu-badge">💻 {device_name}</span>', unsafe_allow_html=True)

    st.divider()

    # Selezione modello con info dettagliate
    st.subheader("🤖 Modello AI")

    # Raggruppa modelli per tipo
    modelli_bolt = [k for k, v in MODELLI_DISPONIBILI.items() if v['tipo'] == 'bolt']
    modelli_t5 = [k for k, v in MODELLI_DISPONIBILI.items() if v['tipo'] == 't5']

    tipo_modello = st.radio("Famiglia modello:", ["⚡ Bolt (Veloci)", "🎯 T5 (Originali)"])

    if "Bolt" in tipo_modello:
        modello_selezionato = st.selectbox(
            "Modello:",
            modelli_bolt,
            format_func=lambda x: MODELLI_DISPONIBILI[x]['nome']
        )
    else:
        modello_selezionato = st.selectbox(
            "Modello:",
            modelli_t5,
            format_func=lambda x: MODELLI_DISPONIBILI[x]['nome']
        )

    # Mostra info modello selezionato
    info_modello = MODELLI_DISPONIBILI[modello_selezionato]
    st.caption(f"📦 {info_modello['dimensione']} | {info_modello['parametri']} parametri")
    st.caption(f"⚡ Velocità: {render_speed_stars(info_modello['velocita'])}")
    st.caption(f"🎯 Accuratezza: {render_accuracy_stars(info_modello['accuratezza'])}")

    st.divider()

    st.caption("Creato con ❤️ usando Chronos di Amazon")
    st.caption("v2.0 - Dicembre 2024")

# ============================================
# HOME PAGE
# ============================================
if pagina == "🏠 Home":
    st.markdown('<h1 class="main-header">📈 Chronos Forecaster Pro by Almo</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Previsioni intelligenti per le tue serie temporali - Tutti i modelli inclusi</p>', unsafe_allow_html=True)

    st.divider()

    # Statistiche
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🤖 Modelli Disponibili", len(MODELLI_DISPONIBILI))
    with col2:
        st.metric("⚡ Modelli Bolt", len([m for m in MODELLI_DISPONIBILI.values() if m['tipo'] == 'bolt']))
    with col3:
        st.metric("🎯 Modelli T5", len([m for m in MODELLI_DISPONIBILI.values() if m['tipo'] == 't5']))
    with col4:
        device, device_name = get_device()
        st.metric("💻 Dispositivo", device_name.split()[0])

    st.divider()

    # Introduzione
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🎯 Cosa fa?")
        st.write("""
        Chronos Forecaster ti permette di fare **previsioni sul futuro**
        basandosi sui tuoi dati storici. Puoi prevedere:
        - Vendite future
        - Trend di mercato
        - Consumi energetici
        - Traffico web
        - Prezzi e molto altro!
        """)

    with col2:
        st.markdown("### 🚀 Come funziona?")
        st.write("""
        1. **Scegli il modello** (Bolt per velocità, T5 per precisione)
        2. **Carica i tuoi dati** (CSV o Excel)
        3. **Configura** i parametri di previsione
        4. **Genera** con un click
        5. **Esporta o usa l'API** per integrazione
        """)

    with col3:
        st.markdown("### 🧠 La tecnologia")
        st.write("""
        Usa **Chronos** di Amazon Science:
        - **8 modelli** da scegliere
        - **Zero-shot learning** (nessun addestramento richiesto)
        - **Previsioni probabilistiche** con intervalli di confidenza
        - **GPU acceleration** automatico
        """)

    st.divider()

    # Quick Start
    st.markdown("### ⚡ Quick Start - Prova subito!")

    col1, col2 = st.columns(2)
    with col1:
        st.info("👉 Vai a **📊 Previsioni** per iniziare con i tuoi dati")
    with col2:
        st.info("👉 Vai a **🤖 Modelli** per confrontare i modelli disponibili")

    # Demo con dati di esempio
    st.markdown("### 🎮 Demo Interattiva")
    col1, col2 = st.columns([1, 2])

    with col1:
        demo_tipo = st.selectbox(
            "Scegli un dataset di esempio:",
            ["vendite", "temperatura", "visite_web", "energia", "stock"],
            format_func=lambda x: {
                "vendite": "📦 Vendite mensili",
                "temperatura": "🌡️ Temperature medie",
                "visite_web": "🌐 Visite al sito web",
                "energia": "⚡ Consumi energetici",
                "stock": "📈 Prezzi azioni"
            }[x]
        )

        demo_periodi = st.slider("Periodi da prevedere:", 3, 12, 6)

        if st.button("🔮 Genera Previsione Demo", type="primary", use_container_width=True):
            with st.spinner(f"Caricamento {MODELLI_DISPONIBILI[modello_selezionato]['nome']}..."):
                pipeline, device_used = load_model(modello_selezionato)
                st.toast(f"Modello caricato su {device_used.upper()}")

            with st.spinner("Generazione previsioni..."):
                start_time = time.time()
                df_demo = create_sample_data(demo_tipo)
                col_valore = df_demo.columns[1]

                forecast = generate_forecast(
                    pipeline,
                    df_demo[col_valore],
                    demo_periodi,
                    MODELLI_DISPONIBILI[modello_selezionato]
                )
                elapsed = time.time() - start_time

                st.session_state['demo_df'] = df_demo
                st.session_state['demo_forecast'] = forecast
                st.session_state['demo_periodi'] = demo_periodi
                st.session_state['demo_col'] = col_valore
                st.session_state['demo_time'] = elapsed

    with col2:
        if 'demo_forecast' in st.session_state:
            df_demo = st.session_state['demo_df']
            forecast = st.session_state['demo_forecast']
            demo_periodi = st.session_state['demo_periodi']
            col_valore = st.session_state['demo_col']
            elapsed = st.session_state.get('demo_time', 0)

            # Info tempo
            st.caption(f"⏱️ Previsione generata in {elapsed:.2f} secondi")

            # Crea date future
            last_date = df_demo['data'].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=demo_periodi, freq='ME')

            # Crea grafico
            fig = go.Figure()

            # Dati storici
            fig.add_trace(go.Scatter(
                x=df_demo['data'], y=df_demo[col_valore],
                mode='lines+markers', name='Dati Storici',
                line=dict(color='#1f77b4', width=2)
            ))

            # Previsione
            fig.add_trace(go.Scatter(
                x=future_dates, y=forecast['median'],
                mode='lines+markers', name='Previsione',
                line=dict(color='#ff7f0e', width=3)
            ))

            # Intervallo di confidenza 80%
            fig.add_trace(go.Scatter(
                x=list(future_dates) + list(future_dates)[::-1],
                y=list(forecast['q90']) + list(forecast['q10'])[::-1],
                fill='toself', fillcolor='rgba(255,127,14,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Intervallo 80%'
            ))

            fig.update_layout(
                title=f"Previsione {col_valore.title()}",
                xaxis_title="Data",
                yaxis_title=col_valore.title(),
                hovermode='x unified',
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

# ============================================
# MODELLI PAGE
# ============================================
elif pagina == "🤖 Modelli":
    st.title("🤖 Confronto Modelli Chronos")

    st.markdown("""
    Chronos offre due famiglie di modelli:
    - **Bolt**: Fino a 250x più veloci, restituiscono quantili direttamente
    - **T5**: Modelli originali, generano campioni probabilistici
    """)

    st.divider()

    # Tabella comparativa
    st.markdown("### 📊 Tabella Comparativa")

    df_modelli = pd.DataFrame([
        {
            "Modello": info['nome'],
            "Famiglia": info['tipo'].upper(),
            "Dimensione": info['dimensione'],
            "Parametri": info['parametri'],
            "Velocità": render_speed_stars(info['velocita']),
            "Accuratezza": render_accuracy_stars(info['accuratezza']),
            "Descrizione": info['descrizione']
        }
        for info in MODELLI_DISPONIBILI.values()
    ])

    st.dataframe(df_modelli, use_container_width=True, hide_index=True)

    st.divider()

    # Dettagli per famiglia
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ⚡ Famiglia Bolt")
        st.success("""
        **Vantaggi:**
        - Fino a 250x più veloci dei T5
        - Restituiscono 9 quantili direttamente
        - Ideali per applicazioni real-time
        - Minor consumo di memoria

        **Quando usarli:**
        - Applicazioni in produzione
        - Dashboard interattive
        - Previsioni batch su molti dati
        """)

        for key, info in MODELLI_DISPONIBILI.items():
            if info['tipo'] == 'bolt':
                with st.expander(f"📦 {info['nome']}"):
                    st.write(f"**Dimensione:** {info['dimensione']}")
                    st.write(f"**Parametri:** {info['parametri']}")
                    st.write(f"**Descrizione:** {info['descrizione']}")
                    st.code(f"model_name = '{key}'")

    with col2:
        st.markdown("### 🎯 Famiglia T5")
        st.info("""
        **Vantaggi:**
        - Maggiore flessibilità nelle previsioni
        - Generano campioni probabilistici
        - Possono essere fine-tuned
        - Modelli originali testati su 27+ benchmark

        **Quando usarli:**
        - Analisi approfondite
        - Quando serve campionamento Monte Carlo
        - Ricerca e sperimentazione
        """)

        for key, info in MODELLI_DISPONIBILI.items():
            if info['tipo'] == 't5':
                with st.expander(f"📦 {info['nome']}"):
                    st.write(f"**Dimensione:** {info['dimensione']}")
                    st.write(f"**Parametri:** {info['parametri']}")
                    st.write(f"**Descrizione:** {info['descrizione']}")
                    st.code(f"model_name = '{key}'")

    st.divider()

    # Raccomandazioni
    st.markdown("### 💡 Raccomandazioni")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🚀 Per iniziare")
        st.write("""
        **Usa: `chronos-bolt-small`**

        Equilibrio perfetto tra velocità e accuratezza.
        Ideale per la maggior parte dei casi d'uso.
        """)

    with col2:
        st.markdown("#### 🏭 Per produzione")
        st.write("""
        **Usa: `chronos-bolt-base`**

        Massima accuratezza tra i modelli veloci.
        Raccomandato per previsioni critiche.
        """)

    with col3:
        st.markdown("#### 🔬 Per ricerca")
        st.write("""
        **Usa: `chronos-t5-large`**

        Massima accuratezza assoluta.
        Richiede più risorse e tempo.
        """)

# ============================================
# API PAGE
# ============================================
elif pagina == "🔌 API":
    st.title("🔌 API REST per Integrazioni")

    st.markdown("""
    Questa sezione mostra come usare Chronos programmaticamente per integrarlo
    nelle tue applicazioni.
    """)

    st.divider()

    # Python SDK
    st.markdown("### 🐍 Uso con Python")

    st.code("""
# Installazione
pip install chronos-forecasting torch pandas

# Uso base con Bolt (veloce)
from chronos import BaseChronosPipeline
import torch
import pandas as pd

# Carica modello
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-small",
    device_map="cpu"  # o "cuda" per GPU
)

# Prepara i dati (array di valori storici)
dati_storici = [100, 120, 115, 130, 125, 140, 135, 150]
context = torch.tensor(dati_storici, dtype=torch.float32).unsqueeze(0)

# Genera previsioni
forecast = pipeline.predict(context, prediction_length=6)

# Estrai risultati (Bolt restituisce 9 quantili)
mediana = forecast[0, 4, :].numpy()  # Quantile 50%
lower_10 = forecast[0, 0, :].numpy()  # Quantile 10%
upper_90 = forecast[0, 8, :].numpy()  # Quantile 90%

print(f"Previsione: {mediana}")
print(f"Intervallo: {lower_10} - {upper_90}")
    """, language="python")

    st.divider()

    # T5 con campionamento
    st.markdown("### 🎲 Uso con modelli T5 (campionamento)")

    st.code("""
# Per modelli T5 che generano campioni
from chronos import BaseChronosPipeline
import torch
import numpy as np

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu"
)

dati_storici = [100, 120, 115, 130, 125, 140, 135, 150]
context = torch.tensor(dati_storici, dtype=torch.float32).unsqueeze(0)

# T5 genera campioni (num_samples)
forecast = pipeline.predict(context, prediction_length=6, num_samples=20)

# Calcola quantili dai campioni
samples = forecast[0].numpy()  # [samples, prediction_length]
mediana = np.percentile(samples, 50, axis=0)
lower_10 = np.percentile(samples, 10, axis=0)
upper_90 = np.percentile(samples, 90, axis=0)
    """, language="python")

    st.divider()

    # FastAPI example
    st.markdown("### 🌐 Creare un API Server con FastAPI")

    st.code("""
# api_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from chronos import BaseChronosPipeline
import torch

app = FastAPI(title="Chronos Forecast API")

# Carica modello all'avvio
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-small",
    device_map="cpu"
)

class ForecastRequest(BaseModel):
    values: List[float]
    prediction_length: int = 6

class ForecastResponse(BaseModel):
    median: List[float]
    lower_10: List[float]
    upper_90: List[float]

@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    try:
        context = torch.tensor(request.values, dtype=torch.float32).unsqueeze(0)
        forecast = pipeline.predict(context, prediction_length=request.prediction_length)

        return ForecastResponse(
            median=forecast[0, 4, :].tolist(),
            lower_10=forecast[0, 0, :].tolist(),
            upper_90=forecast[0, 8, :].tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Avvia con: uvicorn api_server:app --host 0.0.0.0 --port 8000
    """, language="python")

    st.divider()

    # Esempio chiamata API
    st.markdown("### 📡 Esempio Chiamata API")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Request (curl):**")
        st.code("""
curl -X POST "http://localhost:8000/forecast" \\
  -H "Content-Type: application/json" \\
  -d '{
    "values": [100, 120, 115, 130, 125, 140],
    "prediction_length": 6
  }'
        """, language="bash")

    with col2:
        st.markdown("**Response:**")
        st.code("""
{
  "median": [145.2, 148.7, 152.1, 155.4, 158.8, 162.1],
  "lower_10": [135.1, 136.2, 137.5, 138.9, 140.3, 141.8],
  "upper_90": [155.3, 161.2, 166.7, 172.0, 177.3, 182.4]
}
        """, language="json")

    st.divider()

    # Requirements
    st.markdown("### 📋 Requirements per deployment")

    st.code("""
# requirements.txt
chronos-forecasting>=1.2.0
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
fastapi>=0.100.0
uvicorn>=0.22.0
    """)

# ============================================
# PREVISIONI PAGE
# ============================================
elif pagina == "📊 Previsioni":
    st.title("📊 Genera Previsioni")

    # Step 1: Caricamento dati
    st.markdown("### 1️⃣ Carica i tuoi dati")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Carica un file CSV o Excel",
            type=['csv', 'xlsx', 'xls'],
            help="Il file deve avere almeno una colonna con date e una con valori numerici"
        )

    with col2:
        st.markdown("**Formato richiesto:**")
        st.code("""
data,valore
2023-01-01,100
2023-02-01,120
2023-03-01,115
...
        """)

    # Opzione dati di esempio
    usa_esempio = st.checkbox("🎮 Usa dati di esempio invece")

    if usa_esempio:
        tipo_esempio = st.selectbox(
            "Tipo di dati:",
            ["vendite", "temperatura", "visite_web", "energia", "stock"],
            format_func=lambda x: {
                "vendite": "📦 Vendite mensili",
                "temperatura": "🌡️ Temperature",
                "visite_web": "🌐 Visite web",
                "energia": "⚡ Consumi energetici",
                "stock": "📈 Prezzi azioni"
            }[x]
        )
        df = create_sample_data(tipo_esempio)
        st.success(f"✅ Caricati {len(df)} record di esempio")
    elif uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, parse_dates=True)
            else:
                df = pd.read_excel(uploaded_file, parse_dates=True)
            st.success(f"✅ File caricato: {len(df)} righe")
        except Exception as e:
            st.error(f"❌ Errore nel caricamento: {e}")
            df = None
    else:
        df = None

    if df is not None:
        st.divider()

        # Step 2: Configurazione colonne
        st.markdown("### 2️⃣ Configura le colonne")

        col1, col2 = st.columns(2)

        with col1:
            date_cols = [c for c in df.columns if 'data' in c.lower() or 'date' in c.lower() or 'time' in c.lower()]
            if not date_cols:
                date_cols = df.columns.tolist()

            colonna_data = st.selectbox(
                "📅 Colonna con le date:",
                date_cols if date_cols else df.columns.tolist(),
                index=0
            )

        with col2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            colonna_valore = st.selectbox(
                "📊 Colonna con i valori:",
                numeric_cols if numeric_cols else df.columns.tolist(),
                index=0 if numeric_cols else 0
            )

        # Converti colonna data
        try:
            df[colonna_data] = pd.to_datetime(df[colonna_data])
            df = df.sort_values(colonna_data)
        except:
            st.warning("⚠️ Impossibile convertire la colonna date. Assicurati che sia nel formato corretto.")

        # Mostra anteprima
        st.markdown("**Anteprima dati:**")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.dataframe(df[[colonna_data, colonna_valore]].tail(10), use_container_width=True)

        with col2:
            st.metric("📊 Record totali", len(df))
            st.metric("📅 Primo dato", df[colonna_data].min().strftime('%Y-%m-%d'))
            st.metric("📅 Ultimo dato", df[colonna_data].max().strftime('%Y-%m-%d'))

        st.divider()

        # Step 3: Parametri previsione
        st.markdown("### 3️⃣ Parametri Previsione")

        col1, col2, col3 = st.columns(3)

        with col1:
            periodi_previsione = st.slider(
                "🔮 Periodi da prevedere:",
                min_value=1,
                max_value=24,
                value=6,
                help="Quanti periodi nel futuro vuoi prevedere"
            )

        with col2:
            intervallo_confidenza = st.selectbox(
                "📏 Intervallo di confidenza:",
                ["80%", "90%", "60%"],
                help="Quanto vuoi essere sicuro delle previsioni"
            )

        with col3:
            info_mod = MODELLI_DISPONIBILI[modello_selezionato]
            st.markdown("**Modello selezionato:**")
            st.info(f"{info_mod['nome']}")

            # Opzioni avanzate per T5
            if info_mod['tipo'] == 't5':
                num_samples = st.number_input(
                    "Campioni (T5):",
                    min_value=10,
                    max_value=100,
                    value=20,
                    help="Numero di campioni per T5"
                )
            else:
                num_samples = 20

        st.divider()

        # Step 4: Genera previsioni
        st.markdown("### 4️⃣ Genera Previsioni")

        if st.button("🚀 GENERA PREVISIONI", type="primary", use_container_width=True):

            progress_bar = st.progress(0)
            status_text = st.empty()

            # Carica modello
            status_text.text(f"⏳ Caricamento {info_mod['nome']}...")
            progress_bar.progress(20)
            pipeline, device_used = load_model(modello_selezionato)

            status_text.text(f"✅ Modello caricato su {device_used.upper()}")
            progress_bar.progress(40)

            # Genera previsioni
            status_text.text("🔮 Generazione previsioni...")
            progress_bar.progress(60)

            start_time = time.time()
            forecast = generate_forecast(
                pipeline,
                df[colonna_valore],
                periodi_previsione,
                info_mod,
                num_samples
            )
            elapsed = time.time() - start_time

            progress_bar.progress(100)
            status_text.text(f"✅ Previsioni generate in {elapsed:.2f}s!")

            # Salva risultati
            st.session_state['forecast'] = forecast
            st.session_state['df'] = df
            st.session_state['colonna_data'] = colonna_data
            st.session_state['colonna_valore'] = colonna_valore
            st.session_state['periodi'] = periodi_previsione
            st.session_state['intervallo'] = intervallo_confidenza
            st.session_state['elapsed'] = elapsed

        # Mostra risultati
        if 'forecast' in st.session_state:
            st.divider()
            st.markdown("### 📈 Risultati")

            forecast = st.session_state['forecast']
            df_orig = st.session_state['df']
            col_data = st.session_state['colonna_data']
            col_val = st.session_state['colonna_valore']
            periodi = st.session_state['periodi']
            intervallo = st.session_state['intervallo']
            elapsed = st.session_state.get('elapsed', 0)

            st.caption(f"⏱️ Tempo di elaborazione: {elapsed:.2f} secondi")

            # Crea date future
            last_date = df_orig[col_data].max()
            if len(df_orig) > 1:
                freq_days = (df_orig[col_data].iloc[-1] - df_orig[col_data].iloc[-2]).days
                if freq_days > 25:
                    freq = 'ME'
                elif freq_days > 5:
                    freq = 'W'
                else:
                    freq = 'D'
            else:
                freq = 'ME'

            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periodi, freq=freq)

            # Grafico interattivo
            fig = go.Figure()

            # Dati storici
            fig.add_trace(go.Scatter(
                x=df_orig[col_data], y=df_orig[col_val],
                mode='lines+markers', name='📊 Dati Storici',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=4)
            ))

            # Previsione mediana
            fig.add_trace(go.Scatter(
                x=future_dates, y=forecast['median'],
                mode='lines+markers', name='🔮 Previsione',
                line=dict(color='#ff7f0e', width=3),
                marker=dict(size=8)
            ))

            # Intervallo di confidenza
            if intervallo == "90%":
                upper = forecast['q90']
                lower = forecast['q10']
                fill_color = 'rgba(255,127,14,0.15)'
            elif intervallo == "80%":
                upper = forecast['q90']
                lower = forecast['q10']
                fill_color = 'rgba(255,127,14,0.2)'
            else:  # 60%
                upper = forecast['q80']
                lower = forecast['q20']
                fill_color = 'rgba(255,127,14,0.25)'

            fig.add_trace(go.Scatter(
                x=list(future_dates) + list(future_dates)[::-1],
                y=list(upper) + list(lower)[::-1],
                fill='toself', fillcolor=fill_color,
                line=dict(color='rgba(255,255,255,0)'),
                name=f'📏 Intervallo {intervallo}'
            ))

            # Linea verticale "oggi"
            fig.add_vline(x=last_date, line_dash="dash", line_color="gray", annotation_text="Oggi")

            fig.update_layout(
                title=f"Previsione {col_val.title()} - Prossimi {periodi} periodi",
                xaxis_title="Data",
                yaxis_title=col_val.title(),
                hovermode='x unified',
                template='plotly_white',
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Tabella risultati
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("**📋 Dettaglio Previsioni:**")

                df_forecast = pd.DataFrame({
                    'Data': future_dates,
                    'Previsione': forecast['median'].round(1),
                    'Q10': forecast['q10'].round(1),
                    'Q20': forecast['q20'].round(1),
                    'Q80': forecast['q80'].round(1),
                    'Q90': forecast['q90'].round(1)
                })

                st.dataframe(df_forecast, use_container_width=True, hide_index=True)

            with col2:
                st.markdown("**📊 Statistiche:**")

                variazione = ((forecast['median'][-1] - df_orig[col_val].iloc[-1]) / df_orig[col_val].iloc[-1] * 100)

                st.metric(
                    "Previsione media",
                    f"{forecast['median'].mean():.1f}",
                    f"{variazione:+.1f}% vs ultimo valore"
                )

                st.metric("Valore minimo previsto", f"{forecast['median'].min():.1f}")
                st.metric("Valore massimo previsto", f"{forecast['median'].max():.1f}")

            # Export
            st.divider()
            st.markdown("### 💾 Esporta Risultati")

            col1, col2, col3 = st.columns(3)

            with col1:
                csv_buffer = io.StringIO()
                df_forecast.to_csv(csv_buffer, index=False)
                st.download_button(
                    "📥 Scarica CSV",
                    csv_buffer.getvalue(),
                    "previsioni_chronos.csv",
                    "text/csv",
                    use_container_width=True
                )

            with col2:
                # Combina dati storici e previsioni
                df_completo = pd.concat([
                    df_orig[[col_data, col_val]].rename(columns={col_data: 'Data', col_val: 'Valore'}),
                    pd.DataFrame({
                        'Data': future_dates,
                        'Valore': forecast['median'],
                        'Tipo': 'Previsione'
                    })
                ], ignore_index=True)
                df_completo['Tipo'] = df_completo['Tipo'].fillna('Storico')

                csv_completo = io.StringIO()
                df_completo.to_csv(csv_completo, index=False)
                st.download_button(
                    "📥 Scarica Tutto",
                    csv_completo.getvalue(),
                    "dati_completi_chronos.csv",
                    "text/csv",
                    use_container_width=True
                )

            with col3:
                # Export JSON per API
                json_export = {
                    "model": modello_selezionato,
                    "prediction_length": periodi,
                    "forecasts": {
                        "dates": [d.strftime('%Y-%m-%d') for d in future_dates],
                        "median": forecast['median'].tolist(),
                        "q10": forecast['q10'].tolist(),
                        "q90": forecast['q90'].tolist()
                    }
                }
                st.download_button(
                    "📥 Scarica JSON",
                    json.dumps(json_export, indent=2),
                    "previsioni_chronos.json",
                    "application/json",
                    use_container_width=True
                )

# ============================================
# DOCUMENTAZIONE PAGE
# ============================================
elif pagina == "📚 Documentazione":
    st.title("📚 Documentazione Completa")

    st.markdown("""
    ## Cos'è Chronos?

    **Chronos** è una famiglia di modelli di intelligenza artificiale sviluppata da
    **Amazon Science** per fare previsioni su serie temporali. È stato rilasciato
    nel 2024 ed è considerato uno dei migliori modelli open-source per questo compito.

    ### 🎯 Caratteristiche principali

    | Caratteristica | Descrizione |
    |----------------|-------------|
    | **Zero-shot** | Funziona senza bisogno di addestramento specifico sui tuoi dati |
    | **Probabilistico** | Non dà solo un valore, ma anche l'intervallo di incertezza |
    | **8 modelli** | Da tiny a large, per ogni esigenza |
    | **GPU Support** | Accelerazione automatica su GPU NVIDIA e Apple Silicon |

    ---

    ## Modelli Disponibili

    ### Famiglia Bolt (Veloci)

    | Modello | Parametri | Dimensione | Uso consigliato |
    |---------|-----------|------------|-----------------|
    | chronos-bolt-mini | 9M | ~20MB | Test rapidi |
    | chronos-bolt-small | 48M | ~50MB | Uso generale |
    | chronos-bolt-base | 205M | ~200MB | Produzione |

    ### Famiglia T5 (Originali)

    | Modello | Parametri | Dimensione | Uso consigliato |
    |---------|-----------|------------|-----------------|
    | chronos-t5-tiny | 8M | ~30MB | Dispositivi limitati |
    | chronos-t5-mini | 20M | ~80MB | Uso leggero |
    | chronos-t5-small | 46M | ~150MB | Uso generale |
    | chronos-t5-base | 200M | ~400MB | Alta qualità |
    | chronos-t5-large | 710M | ~1.2GB | Massima accuratezza |

    ---

    ## Come usare questa dashboard

    ### 1️⃣ Prepara i tuoi dati

    I tuoi dati devono essere in un file **CSV** o **Excel** con:
    - Una colonna con le **date** (es: `data`, `date`, `timestamp`)
    - Una colonna con i **valori** numerici (es: `vendite`, `temperatura`, `visite`)

    **Esempio:**
    ```
    data,vendite
    2023-01-01,1500
    2023-02-01,1800
    2023-03-01,1650
    2023-04-01,2000
    ```

    ### 2️⃣ Scegli il modello

    Nella sidebar:
    1. Seleziona la famiglia (Bolt o T5)
    2. Scegli il modello specifico
    3. Considera velocità vs accuratezza

    ### 3️⃣ Carica e configura

    1. Vai in **📊 Previsioni**
    2. Carica il file o usa dati di esempio
    3. Seleziona le colonne corrette
    4. Imposta i parametri

    ### 4️⃣ Genera ed esporta

    1. Clicca "GENERA PREVISIONI"
    2. Analizza il grafico e le statistiche
    3. Esporta in CSV, JSON o completo

    ---

    ## Interpretare i risultati

    ### La linea di previsione
    Rappresenta il valore **più probabile** (mediana delle previsioni).

    ### I quantili
    - **Q10**: 10% di probabilità che il valore sia sotto
    - **Q20**: 20% di probabilità che il valore sia sotto
    - **Q50 (mediana)**: Valore centrale
    - **Q80**: 20% di probabilità che il valore sia sopra
    - **Q90**: 10% di probabilità che il valore sia sopra

    ### Perché l'intervallo si allarga?
    Più ci si allontana nel futuro, maggiore è l'incertezza.
    È normale che l'intervallo si allarghi!

    ---

    ## Limitazioni

    ⚠️ **Attenzione:**

    - Le previsioni sono stime, non certezze
    - Funziona meglio con dati regolari (giornalieri, settimanali, mensili)
    - Non considera eventi esterni (es: pandemie, crisi economiche)
    - Richiede almeno 10-20 punti storici per buone previsioni
    - Modelli più grandi richiedono più RAM

    ---

    ## Risorse utili

    - 📄 [Paper originale di Chronos](https://arxiv.org/abs/2403.07815)
    - 💻 [Repository GitHub](https://github.com/amazon-science/chronos-forecasting)
    - 🤗 [Modelli su Hugging Face](https://huggingface.co/amazon)
    - 📊 [Benchmark e valutazioni](https://github.com/amazon-science/chronos-forecasting#evaluation)
    """)

# ============================================
# FAQ PAGE
# ============================================
elif pagina == "❓ FAQ":
    st.title("❓ Domande Frequenti")

    with st.expander("📊 Quanti dati storici mi servono?"):
        st.write("""
        **Minimo consigliato**: 10-20 punti dati

        **Ideale**: 30+ punti dati (es: 2-3 anni di dati mensili)

        Più dati hai, migliori saranno le previsioni, specialmente se ci sono
        pattern stagionali (es: vendite natalizie).
        """)

    with st.expander("⚡ Bolt vs T5: quale scegliere?"):
        st.write("""
        **Usa Bolt quando:**
        - Hai bisogno di velocità
        - Fai previsioni in tempo reale
        - Hai risorse limitate

        **Usa T5 quando:**
        - Vuoi campionamento Monte Carlo
        - Fai analisi approfondite
        - La velocità non è critica

        In generale, **Bolt è consigliato** per la maggior parte dei casi.
        """)

    with st.expander("🖥️ Come funziona il supporto GPU?"):
        st.write("""
        Il sistema rileva automaticamente:
        - **GPU NVIDIA (CUDA)**: Accelerazione completa
        - **Apple Silicon (MPS)**: Accelerazione su Mac M1/M2/M3
        - **CPU**: Fallback se non c'è GPU

        Non devi configurare nulla, funziona automaticamente!
        """)

    with st.expander("🎯 Quanto sono accurate le previsioni?"):
        st.write("""
        L'accuratezza dipende da molti fattori:

        - **Qualità dei dati**: Dati puliti = previsioni migliori
        - **Regolarità**: Pattern stabili sono più facili da prevedere
        - **Orizzonte temporale**: Previsioni a breve termine sono più accurate
        - **Modello scelto**: Modelli più grandi = più accurati

        Usa l'intervallo di confidenza per capire l'incertezza!
        """)

    with st.expander("🔄 Posso prevedere più variabili insieme?"):
        st.write("""
        Attualmente questa dashboard supporta una variabile alla volta
        (univariato).

        Per previsioni multivariate, puoi:
        1. Fare previsioni separate per ogni variabile
        2. Usare l'API e combinarle nel tuo codice
        3. Aspettare Chronos-2 che supporterà multivariate
        """)

    with st.expander("💾 In che formato posso esportare?"):
        st.write("""
        **Formati disponibili:**

        - **CSV**: Compatibile con Excel, Google Sheets, ecc.
        - **JSON**: Per integrazioni API e applicazioni web
        - **CSV Completo**: Include dati storici + previsioni
        """)

    with st.expander("🖥️ Funziona offline?"):
        st.write("""
        **Prima esecuzione**: Serve internet per scaricare il modello

        **Esecuzioni successive**: Funziona offline! Il modello viene salvato
        nella cache del tuo computer.

        La cache si trova in: `~/.cache/huggingface/`
        """)

    with st.expander("🐌 È lento, cosa posso fare?"):
        st.write("""
        Prova questi suggerimenti:

        1. **Usa Chronos-Bolt** invece di T5
        2. **Scegli un modello più piccolo** (mini o small)
        3. **Riduci i periodi da prevedere**
        4. **Chiudi altre applicazioni** per liberare RAM
        5. **Usa una GPU** se disponibile

        La prima esecuzione è sempre più lenta (download modello).
        """)

    with st.expander("❌ Ho un errore, cosa faccio?"):
        st.write("""
        **Errori comuni:**

        1. **"Colonna non trovata"**: Verifica i nomi delle colonne
        2. **"Formato data non valido"**: Usa formati standard (YYYY-MM-DD)
        3. **"Memoria insufficiente"**: Prova con modello più piccolo
        4. **"CUDA out of memory"**: Passa a CPU o modello più piccolo

        Se il problema persiste, prova a ricaricare la pagina (F5).
        """)

    with st.expander("🔌 Come integro le previsioni nella mia app?"):
        st.write("""
        Hai diverse opzioni:

        1. **Esporta JSON** e importalo nella tua app
        2. **Usa l'API FastAPI** (vedi sezione 🔌 API)
        3. **Copia il codice Python** dalla documentazione
        4. **Usa Docker** per deployare il servizio

        Vedi la sezione **🔌 API** per esempi dettagliati.
        """)

    st.divider()

    st.info("""
    💡 **Hai altre domande?**

    Questa dashboard è stata creata per sperimentare con Chronos.
    Per supporto avanzato, consulta la [documentazione ufficiale](https://github.com/amazon-science/chronos-forecasting).
    """)
