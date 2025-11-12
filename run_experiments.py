"""
run_experiments.py

Executa experimentos comparando 9 variações do Tabu Search 
para otimização de portfólios (baseado em Schaerf, 2001).
"""

import sys
sys.path.append('./')

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime

from data_loader import PortfolioDataLoader
from experimentos import executar_experimento


# ============================================================
# Definição das 9 estratégias clássicas (Schaerf, 2001)
# ============================================================

config_estrategias = {
    # --- Grupo 1: Estratégias IDID (idID) ---
    "idID_short_tabu": {
        "max_iter": 1000,
        "tabu_tenure": (5, 10),
        "step_size": 0.1,
        "step_delta": 0.02,
        "neighbor_sample_ratio": 0.8,
        "token_ring_enabled": False,
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "NFLX", "TSLA", "ORCL",
            "IBM", "INTC", "ADBE", "AMD", "CRM", "CSCO", "PEP", "KO", "DIS", "V", "MA", "PYPL"]
    },
    "idID_medium_tabu": {
        "max_iter": 1000,
        "tabu_tenure": (10, 20),
        "step_size": 0.1,
        "step_delta": 0.02,
        "neighbor_sample_ratio": 0.8,
        "token_ring_enabled": False,
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "NFLX", "TSLA", "ORCL",
            "IBM", "INTC", "ADBE", "AMD", "CRM", "CSCO", "PEP", "KO", "DIS", "V", "MA", "PYPL"]
    },
    "idID_long_tabu": {
        "max_iter": 1000,
        "tabu_tenure": (20, 30),
        "step_size": 0.1,
        "step_delta": 0.02,
        "neighbor_sample_ratio": 0.8,
        "token_ring_enabled": False,
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "NFLX", "TSLA", "ORCL",
            "IBM", "INTC", "ADBE", "AMD", "CRM", "CSCO", "PEP", "KO", "DIS", "V", "MA", "PYPL"]
    },

    # --- Grupo 2: Estratégias TID ---
    "TID_short_tabu": {
        "max_iter": 1000,
        "tabu_tenure": (5, 10),
        "step_size": 0.1,
        "step_delta": 0.02,
        "neighbor_sample_ratio": 0.8,
        "token_ring_enabled": False,
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "NFLX", "TSLA", "ORCL",
            "IBM", "INTC", "ADBE", "AMD", "CRM", "CSCO", "PEP", "KO", "DIS", "V", "MA", "PYPL"]
    },
    "TID_medium_tabu": {
        "max_iter": 1000,
        "tabu_tenure": (10, 20),
        "step_size": 0.1,
        "step_delta": 0.02,
        "neighbor_sample_ratio": 0.8,
        "token_ring_enabled": False,
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "NFLX", "TSLA", "ORCL",
            "IBM", "INTC", "ADBE", "AMD", "CRM", "CSCO", "PEP", "KO", "DIS", "V", "MA", "PYPL"]
    },
    "TID_long_tabu": {
        "max_iter": 1000,
        "tabu_tenure": (20, 30),
        "step_size": 0.1,
        "step_delta": 0.02,
        "neighbor_sample_ratio": 0.8,
        "token_ring_enabled": False,
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "NFLX", "TSLA", "ORCL",
            "IBM", "INTC", "ADBE", "AMD", "CRM", "CSCO", "PEP", "KO", "DIS", "V", "MA", "PYPL"]
    },

    # --- Grupo 3: Estratégias híbridas com Token-Ring (idID+TID) ---
    "idID_TID_short_tabu": {
        "max_iter": 1000,
        "tabu_tenure": (5, 10),
        "step_size": 0.1,
        "step_delta": 0.02,
        "neighbor_sample_ratio": 0.8,
        "token_ring_enabled": True,
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "NFLX", "TSLA", "ORCL",
            "IBM", "INTC", "ADBE", "AMD", "CRM", "CSCO", "PEP", "KO", "DIS", "V", "MA", "PYPL"]
    },
    "idID_TID_medium_tabu": {
        "max_iter": 1000,
        "tabu_tenure": (10, 20),
        "step_size": 0.1,
        "step_delta": 0.02,
        "neighbor_sample_ratio": 0.8,
        "token_ring_enabled": True,
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "NFLX", "TSLA", "ORCL",
            "IBM", "INTC", "ADBE", "AMD", "CRM", "CSCO", "PEP", "KO", "DIS", "V", "MA", "PYPL"]
    },
    "idID_TID_long_tabu": {
        "max_iter": 1000,
        "tabu_tenure": (20, 30),
        "step_size": 0.1,
        "step_delta": 0.02,
        "neighbor_sample_ratio": 0.8,
        "token_ring_enabled": True,
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "NFLX", "TSLA", "ORCL",
            "IBM", "INTC", "ADBE", "AMD", "CRM", "CSCO", "PEP", "KO", "DIS", "V", "MA", "PYPL"]
    },
}


# ============================================================
# Execução dos experimentos
# ============================================================

def main():
    print("=" * 70)
    print("INICIANDO EXPERIMENTOS DE TABU SEARCH (9 ESTRATÉGIAS)")
    print(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    resultados = []

    for strategy_name, config in config_estrategias.items():
        print(f"\n[{strategy_name}] Iniciando execuções...")
        print("-" * 70)

        for run in range(1, 6):  # 5 execuções por estratégia
            seed = 100 + run
            print(f"Run {run}/5...", end=" ", flush=True)
            try:
                res = executar_experimento(strategy_name, config, seed, verbose=False)
                resultados.append(res)
                print(f"Custo: {res['best_cost']:.6f} | "
                      f"Viável: {res['feasible']} | "
                      f"Tempo: {res['time']:.2f}s")
            except Exception as e:
                print(f"Erro: {e}")

    # Consolidação dos resultados
    if len(resultados) == 0:
        print("\n⚠️ Nenhum resultado foi gerado (todas as execuções falharam).")
        return

    df_resultados = pd.DataFrame(resultados)
    df_resultados["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs("results", exist_ok=True)
    detailed_path = f"results/results_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_resultados.to_csv(detailed_path, index=False)

    resumo = df_resultados.groupby("strategy").agg(
        custo_medio=("best_cost", "mean"),
        tempo_medio=("time", "mean"),
        viabilidade_media=("feasible", "mean")
    ).reset_index()

    summary_path = f"results/results_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    resumo.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("EXPERIMENTOS CONCLUÍDOS")
    print(f"Resultados detalhados: {detailed_path}")
    print(f"Resumo agregado:       {summary_path}")
    print("=" * 70)

    print("\nResumo geral:")
    print(resumo.to_string(index=False))


# ============================================================
# Execução direta
# ============================================================

if __name__ == "__main__":
    main()
