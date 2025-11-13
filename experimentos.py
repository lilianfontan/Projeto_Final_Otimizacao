"""
experimentos.py

Módulo responsável por configurar e executar experimentos
comparando diferentes estratégias de Tabu Search para otimização
de portfólios, conforme metodologia de Schaerf (2001).

Inclui salvamento automático das instâncias utilizadas
em cada execução de estratégia (em formato .pkl).
"""

import os
import pickle
import numpy as np
import pandas as pd
import time
from datetime import datetime

from data_loader import PortfolioDataLoader
from tabu_search import TabuSearch


# ============================================================
# Função auxiliar para salvar instâncias
# ============================================================

# ============================================================
# Função auxiliar para salvar instâncias (apenas em .txt)
# ============================================================

def save_instance(instance_data, strategy_name, seed):
    """Salva os dados da instância em arquivo .txt legível"""
    os.makedirs("instances", exist_ok=True)
    filename = f"instances/instance_{strategy_name}_seed_{seed}.txt"

    expected_returns = instance_data["expected_returns"]
    cov_matrix = instance_data["cov_matrix"]

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Estratégia: {strategy_name}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Número de ativos: {instance_data['n_assets']}\n")
        f.write(f"Ativos: {', '.join(instance_data['tickers'])}\n\n")
        f.write("Parâmetros:\n")
        for k, v in instance_data["params"].items():
            f.write(f"  {k}: {v}\n")
        f.write("\nRetornos esperados:\n")
        for val in expected_returns:
            f.write(f"{val:.6f}\n")
        f.write("\nMatriz de covariância:\n")
        for row in cov_matrix:
            f.write(" ".join(f"{x:.6e}" for x in row) + "\n")

    print(f"→ Instância salva em {filename}")



# ============================================================
# Função principal de execução de experimento
# ============================================================

def executar_experimento(strategy_name, config, seed=42, verbose=True):
    """
    Executa uma instância do Tabu Search com a estratégia especificada.

    Parâmetros:
        strategy_name : str
            Nome da estratégia (ex: 'idID_TID', 'TokenRing', etc.)
        config : dict
            Dicionário com os parâmetros do experimento
        seed : int
            Semente aleatória para reprodutibilidade
        verbose : bool
            Se True, imprime logs detalhados
    """

    #np.random.seed(seed)

    # ------------------------------------------------------------
    # 1. Carregar dados via PortfolioDataLoader
    # ------------------------------------------------------------
    tickers = config.get("tickers", ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"])
    loader = PortfolioDataLoader(tickers)
    loader.download_data(verbose=verbose)

    expected_returns = loader.get_expected_returns()
    cov_matrix = loader.get_covariance_matrix()
    n_assets = len(expected_returns)
    valid_tickers = loader.get_valid_tickers()

    if expected_returns is None or cov_matrix is None:
        raise ValueError("Falha ao carregar dados de mercado. Nenhum ativo válido encontrado.")

    # Empacotar dados e salvar instância
    instance_data = {
        "strategy": strategy_name,
        "seed": seed,
        "n_assets": n_assets,
        "tickers": valid_tickers,
        "expected_returns": expected_returns,
        "cov_matrix": cov_matrix,
        "params": config
    }
    save_instance(instance_data, strategy_name, seed)

    # ------------------------------------------------------------
    # 2. Executar Tabu Search
    # ------------------------------------------------------------

    # Garante que tabu_tenure seja sempre uma tupla válida (min, max)
    tabu_tenure = config.get("tabu_tenure", (10, 25))
    if isinstance(tabu_tenure, int):
        tabu_tenure = (tabu_tenure, tabu_tenure + 5)
    elif isinstance(tabu_tenure, (list, tuple)) and len(tabu_tenure) == 1:
        tabu_tenure = (tabu_tenure[0], tabu_tenure[0] + 5)

    tabu = TabuSearch(
        n_assets=n_assets,
        k_max=5,
        epsilon=0.01,
        delta=0.4,
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        min_return=np.mean(expected_returns),
        max_iter=config.get("max_iter", 1000),
        tabu_tenure=tabu_tenure,
        strategy=strategy_name,
        step_size=config.get("step_size", 0.1),
        step_delta=config.get("step_delta", 0.02),
        neighbor_sample_ratio=config.get("neighbor_sample_ratio", 0.3),
        token_ring_enabled=config.get("token_ring_enabled", False),
    )


    start_time = time.time()
    best_solution = tabu.run(verbose=verbose)
    elapsed = time.time() - start_time

    # Coleta informações da melhor solução
    info = tabu.get_portfolio_info(best_solution)
    feasible = info["is_feasible"] if info else False
    best_cost = best_solution.cost if best_solution else np.nan

    # ------------------------------------------------------------
    # 3. Retornar resultados
    # ------------------------------------------------------------
    result = {
        "strategy": strategy_name,
        "seed": seed,
        "best_cost": best_cost,
        "feasible": feasible,
        "time": elapsed
    }

    if verbose:
        print(f"Run concluído → Custo: {best_cost:.6f} | "
              f"Viável: {feasible} | Tempo: {elapsed:.2f}s")

    return result


# ============================================================
# Função para execução em lote (usada por run_experiments.py)
# ============================================================

def rodar_experimentos(config_estrategias, n_runs=10, verbose=True):
    """
    Executa várias estratégias de Tabu Search e retorna resultados agregados.
    """
    resultados = []

    for strategy_name, config in config_estrategias.items():
        if verbose:
            print("=" * 60)
            print(f"Executando estratégia: {strategy_name}")
            print("=" * 60)

        for run in range(1, n_runs + 1):
            seed = 100 + run
            if verbose:
                print(f"Run {run}/{n_runs}... ", end="", flush=True)
            try:
                res = executar_experimento(strategy_name, config, seed, verbose=False)
                resultados.append(res)
                if verbose:
                    print(f"Custo: {res['best_cost']:.6f} | "
                          f"Viável: {res['feasible']} | "
                          f"Tempo: {res['time']:.2f}s")
            except Exception as e:
                print(f"Erro na estratégia {strategy_name}, run {run}: {e}")

    df_resultados = pd.DataFrame(resultados)
    df_resultados["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Salvar resultados detalhados
    os.makedirs("results", exist_ok=True)
    detailed_path = f"results/results_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_resultados.to_csv(detailed_path, index=False)

    # Corrige caso falte a coluna "strategy"
    if "strategy" not in df_resultados.columns:
        df_resultados["strategy"] = "TabuSearch"

    # Salvar resumo
    resumo = df_resultados.groupby("strategy").agg(
        custo_medio=("best_cost", "mean"),
        tempo_medio=("time", "mean"),
        viabilidade_media=("feasible", "mean")
    ).reset_index()

    summary_path = f"results/results_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    resumo.to_csv(summary_path, index=False)

    if verbose:
        print("\nResultados salvos em:")
        print(f"  - {detailed_path}")
        print(f"  - {summary_path}")

    return df_resultados, resumo
