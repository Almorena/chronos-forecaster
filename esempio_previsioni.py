"""
CHRONOS - Esempio di Previsione Serie Temporali
================================================
Questo script mostra come usare Chronos per fare previsioni.
Usiamo dati finti di vendite mensili per prevedere i prossimi 6 mesi.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import torch

# ============================================
# 1. CREIAMO DEI DATI DI ESEMPIO
# ============================================
# Simuliamo 24 mesi di vendite con un trend crescente e stagionalita'

print("=" * 50)
print("CHRONOS - Previsione Serie Temporali")
print("=" * 50)
print("\n1. Creazione dati di esempio...")

# Creiamo date mensili per 2 anni (24 mesi)
date_iniziali = pd.date_range(start='2022-01-01', periods=24, freq='ME')

# Creiamo vendite con:
# - Trend crescente (le vendite aumentano nel tempo)
# - Stagionalita' (piu' vendite in estate e natale)
np.random.seed(42)  # Per risultati riproducibili
trend = np.linspace(100, 200, 24)  # Da 100 a 200
stagionalita = 20 * np.sin(np.linspace(0, 4*np.pi, 24))  # Ciclo stagionale
rumore = np.random.normal(0, 10, 24)  # Variazioni casuali
vendite = trend + stagionalita + rumore

# Creiamo un DataFrame (tabella)
dati = pd.DataFrame({
    'data': date_iniziali,
    'vendite': vendite
})

print(f"   Creati {len(dati)} mesi di dati storici")
print(f"   Periodo: da {dati['data'].min().strftime('%Y-%m')} a {dati['data'].max().strftime('%Y-%m')}")
print(f"   Vendite: min={vendite.min():.0f}, max={vendite.max():.0f}, media={vendite.mean():.0f}")

# ============================================
# 2. CARICHIAMO IL MODELLO CHRONOS
# ============================================
print("\n2. Caricamento modello Chronos...")
print("   (la prima volta scarica il modello, potrebbe richiedere un po')")

from chronos import BaseChronosPipeline

# Usiamo Chronos-Bolt che e' il piu' veloce
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-small",
    device_map="cpu"  # Usa CPU (su Mac senza CUDA)
)
print("   Modello caricato!")

# ============================================
# 3. FACCIAMO LE PREVISIONI
# ============================================
print("\n3. Generazione previsioni per i prossimi 6 mesi...")

# Convertiamo le vendite in formato tensor
context = torch.tensor(vendite, dtype=torch.float32).unsqueeze(0)  # [1, 24]

# Generiamo le previsioni
# ChronosBolt restituisce direttamente i quantili
# Output shape: [batch, num_quantiles, prediction_length]
forecast = pipeline.predict(context, prediction_length=6)

# ChronosBolt output ha 9 quantili predefiniti:
# [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# Indice 4 = mediana (0.5), indice 0 = 10%, indice 8 = 90%
previsioni_mediana = forecast[0, 4, :].numpy()  # Mediana (quantile 0.5)
previsioni_basso = forecast[0, 0, :].numpy()    # Quantile 0.1 (10%)
previsioni_alto = forecast[0, 8, :].numpy()     # Quantile 0.9 (90%)

print(f"   Previsioni generate per i prossimi 6 mesi!")

# ============================================
# 4. MOSTRIAMO I RISULTATI
# ============================================
print("\n4. Risultati delle previsioni:")
print("-" * 40)

# Creiamo le date future
date_future = pd.date_range(start=dati['data'].max() + timedelta(days=1), periods=6, freq='ME')

for i, (data, prev, basso, alto) in enumerate(zip(date_future, previsioni_mediana, previsioni_basso, previsioni_alto)):
    print(f"   {data.strftime('%Y-%m')}: {prev:.0f} vendite (intervallo: {basso:.0f} - {alto:.0f})")

# ============================================
# 5. CREIAMO UN GRAFICO
# ============================================
print("\n5. Creazione grafico...")

plt.figure(figsize=(12, 6))

# Dati storici (blu)
plt.plot(dati['data'], dati['vendite'], 'b-', linewidth=2, label='Dati storici', marker='o', markersize=4)

# Previsioni (rosso)
plt.plot(date_future, previsioni_mediana, 'r-', linewidth=2, label='Previsione', marker='o', markersize=6)

# Intervallo di confidenza (area rossa chiara)
plt.fill_between(date_future, previsioni_basso, previsioni_alto, color='red', alpha=0.2, label='Intervallo 80%')

# Linea verticale per separare passato e futuro
plt.axvline(x=dati['data'].max(), color='gray', linestyle='--', alpha=0.5)
plt.text(dati['data'].max(), plt.ylim()[1], ' FUTURO', fontsize=10, color='gray')

plt.title('Previsione Vendite con Chronos', fontsize=14, fontweight='bold')
plt.xlabel('Data')
plt.ylabel('Vendite')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Salviamo il grafico
plt.savefig('previsione_vendite.png', dpi=150)
print("   Grafico salvato come 'previsione_vendite.png'")

print("\n" + "=" * 50)
print("FATTO! Hai appena usato Chronos per fare previsioni!")
print("=" * 50)
print("\nApri il file 'previsione_vendite.png' per vedere il grafico.")
