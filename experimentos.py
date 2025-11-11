"""
experimentos.py

Gera dados sintéticos de retornos esperados e matrizes de covariância
para simulações do Tabu Search em otimização de portfólios.

Inspirado no modelo usado por Schaerf (2001).
"""

import numpy as np

def gerar_dados_sinteticos(n_ativos=100, semente=42):
    """
    Gera dados sintéticos para o problema de otimização de portfólio.

    Args:
        n_ativos (int): Número de ativos a gerar.
        semente (int): Seed para reprodutibilidade.

    Returns:
        tuple: (retornos_esperados, matriz_covariancia, tickers)
    """
    np.random.seed(semente)

    # --- Retornos esperados ---
    # valores pequenos (0.0005–0.002) simulando retornos diários médios
    retornos_esperados = np.random.uniform(0.0005, 0.002, n_ativos)

    # --- Matriz de covariância ---
    # gera uma matriz positiva definida (simétrica e semi-definida positiva)
    A = np.random.rand(n_ativos, n_ativos)
    matriz_covariancia = np.dot(A, A.T) / n_ativos  # normalização

    # --- Tickers fictícios ---
    tickers = [f"ASSET_{i+1}" for i in range(n_ativos)]

    return retornos_esperados, matriz_covariancia, tickers


if __name__ == "__main__":
    # Teste rápido
    mu, cov, names = gerar_dados_sinteticos(5)
    print("Retornos esperados:", mu)
    print("Matriz de covariância:\n", cov)
    print("Tickers:", names)
