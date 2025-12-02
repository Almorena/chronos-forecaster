#!/bin/bash
# Chronos Forecaster - Script di avvio
# =====================================

echo "🚀 Avvio Chronos Forecaster..."
echo ""

# Vai nella cartella del progetto
cd "$(dirname "$0")"

# Attiva l'ambiente virtuale
source venv/bin/activate

# Avvia Streamlit
echo "📊 Apro la dashboard nel browser..."
streamlit run app.py --server.headless true
