"""
run_experiments.py

Script para executar experimentos comparando diferentes estratégias de Tabu Search
para otimização de portfólios, conforme metodologia de Schaerf (2001).

Os resultados podem ser usados para gerar tabelas e gráficos similares aos do artigo.
"""

import sys
sys.path.append('/mnt/project')

import numpy as np
import pandas as pd
import time
import json
from datetime import datetime, timedelta
from tabu_search import TabuSearch
from experimentos import gerar_dados_sinteticos
from data_loader import PortfolioDataLoader


# ============================================================================
# CONFIGURAÇÕES DOS EXPERIMENTOS
# ============================================================================

class ExperimentConfig:
    """Configuração de um experimento"""
    def __init__(self, name, use_idID=True, use_TID=True, 
                 tabu_min=10, tabu_max=25, max_iter=500,
                 K=20, H=1, step_size=0.3, step_variation=0.3):
        self.name = name
        self.use_idID = use_idID
        self.use_TID = use_TID
        self.tabu_min = tabu_min
        self.tabu_max = tabu_max
        self.max_iter = max_iter
        self.K = K  # Iterações viáveis para diminuir penalidade
        self.H = H  # Iterações inviáveis para aumentar penalidade
        self.step_size = step_size
        self.step_variation = step_variation


# Definir as estratégias a serem testadas (inspiradas em Schaerf)
STRATEGIES = [
    # Baseline: apenas idID
    ExperimentConfig(
        name="idID_only",
        use_idID=True,
        use_TID=False,
        tabu_min=10,
        tabu_max=25
    ),
    
    # Apenas TID
    ExperimentConfig(
        name="TID_only", 
        use_idID=False,
        use_TID=True,
        tabu_min=10,
        tabu_max=25
    ),
    
    # Combinação idID + TID (estratégia completa)
    ExperimentConfig(
        name="idID_TID",
        use_idID=True,
        use_TID=True,
        tabu_min=10,
        tabu_max=25
    ),
    
    # Lista tabu curta
    ExperimentConfig(
        name="idID_TID_short_tabu",
        use_idID=True,
        use_TID=True,
        tabu_min=5,
        tabu_max=15
    ),
    
    # Lista tabu longa
    ExperimentConfig(
        name="idID_TID_long_tabu",
        use_idID=True,
        use_TID=True,
        tabu_min=15,
        tabu_max=35
    ),
    
    # Shifting penalty agressivo (diminui peso rapidamente)
    ExperimentConfig(
        name="idID_TID_aggressive_penalty",
        use_idID=True,
        use_TID=True,
        tabu_min=10,
        tabu_max=25,
        K=10,  # Diminui mais rápido
        H=1
    ),
    
    # Shifting penalty conservador (aumenta peso rapidamente)
    ExperimentConfig(
        name="idID_TID_conservative_penalty",
        use_idID=True,
        use_TID=True,
        tabu_min=10,
        tabu_max=25,
        K=30,  # Diminui mais devagar
        H=3    # Aumenta mais rápido
    ),
    
    # Step size maior (movimentos mais agressivos)
    ExperimentConfig(
        name="idID_TID_large_step",
        use_idID=True,
        use_TID=True,
        tabu_min=10,
        tabu_max=25,
        step_size=0.5,
        step_variation=0.3
    ),
    
    # Step size menor (movimentos mais conservadores)
    ExperimentConfig(
        name="idID_TID_small_step",
        use_idID=True,
        use_TID=True,
        tabu_min=10,
        tabu_max=25,
        step_size=0.15,
        step_variation=0.15
    ),
]


# ============================================================================
# FUNÇÃO PARA CARREGAR DADOS REAIS
# ============================================================================

def load_real_data(market='sp500', n_assets=30, verbose=True):
    """
    Carrega dados reais de ativos usando data_loader.
    
    Args:
        market: 'ibovespa' ou 'sp500'
        n_assets: Número máximo de ativos
        verbose: Se True, imprime informações
        
    Returns:
        Tupla (retornos_esperados, matriz_covariancia, tickers_validos)
        ou None se falhar
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"CARREGANDO DADOS REAIS: {market.upper()}")
        print(f"{'='*70}")
    
    # Definir tickers por mercado
    if market == 'ibovespa':
        # Principais ações da B3
        tickers_disponiveis = [
            'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA',
            'BBAS3.SA', 'WEGE3.SA', 'RENT3.SA', 'SUZB3.SA', 'RADL3.SA',
            'RAIL3.SA', 'LREN3.SA', 'MGLU3.SA', 'PRIO3.SA', 'JBSS3.SA',
            'EMBR3.SA', 'GGBR4.SA', 'CSNA3.SA', 'USIM5.SA', 'GOAU4.SA',
            'TOTS3.SA', 'CSAN3.SA', 'BPAC11.SA', 'HAPV3.SA', 'CPLE6.SA',
            'EQTL3.SA', 'ELET3.SA', 'CMIG4.SA', 'CPFE3.SA', 'VIVT3.SA'
        ]
    elif market == 'sp500':
        # Principais ações do S&P 500
        tickers_disponiveis = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
            'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
            'V', 'WMT', 'JPM', 'MA', 'PG',
            'XOM', 'HD', 'CVX', 'MRK', 'ABBV',
            'LLY', 'KO', 'PEP', 'COST', 'AVGO',
            'TMO', 'MCD', 'CSCO', 'ACN', 'DIS'
        ]
    else:
        raise ValueError(f"Mercado '{market}' não reconhecido. Use 'ibovespa' ou 'sp500'.")
    
    # Limitar ao número solicitado
    tickers_selecionados = tickers_disponiveis[:min(n_assets, len(tickers_disponiveis))]
    
    # Definir período (últimos 2 anos)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    if verbose:
        print(f"Período: {start_date} a {end_date}")
        print(f"Tentando carregar {len(tickers_selecionados)} ativos...")
    
    # Carregar dados
    try:
        loader = PortfolioDataLoader(
            tickers=tickers_selecionados,
            start_date=start_date,
            end_date=end_date
        )
        
        precos = loader.download_data(min_data_threshold=0.7, verbose=verbose)
        
        if precos is None or len(loader.get_valid_tickers()) == 0:
            if verbose:
                print("⚠️  Nenhum ativo com dados válidos foi carregado.")
            return None
        
        # Calcular retornos e estatísticas
        retornos = loader.calculate_returns()
        retornos_esperados = loader.get_expected_returns()
        matriz_covariancia = loader.get_covariance_matrix()
        tickers_validos = loader.get_valid_tickers()
        
        if verbose:
            sumario = loader.summary()
            print(f"\n{'='*70}")
            print("RESUMO DOS DADOS CARREGADOS")
            print(f"{'='*70}")
            print(f"Ativos com dados válidos: {sumario['n_assets']}")
            print(f"Dias de negociação: {sumario['n_days']}")
            print(f"Retorno anual médio: {sumario['avg_return_annual']:.2%}")
            print(f"Volatilidade anual média: {sumario['avg_volatility_annual']:.2%}")
            print(f"Tickers: {', '.join(tickers_validos[:10])}{'...' if len(tickers_validos) > 10 else ''}")
            print(f"{'='*70}\n")
        
        return retornos_esperados, matriz_covariancia, tickers_validos
        
    except Exception as e:
        if verbose:
            print(f"\n⚠️  ERRO ao carregar dados: {e}")
        return None


# ============================================================================
# FUNÇÃO PARA EXECUTAR UM EXPERIMENTO
# ============================================================================

def run_single_experiment(config, returns, cov_matrix, k_max, min_return, 
                         epsilon, delta, n_runs=10, seed_base=42):
    """
    Executa um experimento com uma configuração específica.
    
    Args:
        config: ExperimentConfig com parâmetros do experimento
        returns: Retornos esperados dos ativos
        cov_matrix: Matriz de covariância
        k_max: Cardinalidade máxima
        min_return: Retorno mínimo exigido
        epsilon: Alocação mínima por ativo
        delta: Alocação máxima por ativo
        n_runs: Número de execuções independentes
        seed_base: Seed base para reprodutibilidade
        
    Returns:
        DataFrame com resultados agregados
    """
    results = []
    n_assets = len(returns)
    
    print(f"\n{'='*70}")
    print(f"Executando estratégia: {config.name}")
    print(f"{'='*70}")
    
    for run in range(n_runs):
        np.random.seed(seed_base + run)
        
        print(f"  Run {run + 1}/{n_runs}...", end=" ", flush=True)
        
        # Inicializar Tabu Search
        ts = TabuSearch(
            n_assets=n_assets,
            k_max=k_max,
            epsilon=epsilon,
            delta=delta,
            expected_returns=returns,
            cov_matrix=cov_matrix,
            min_return=min_return
        )
        
        # Configurar parâmetros específicos
        ts.tabu_list.min_size = config.tabu_min
        ts.tabu_list.max_size = config.tabu_max
        ts.max_iterations = config.max_iter
        ts.K = config.K
        ts.H = config.H
        ts.neighbor_generator.step_size = config.step_size
        ts.neighbor_generator.step_variation = config.step_variation
        
        # Executar otimização
        start_time = time.time()
        solution = ts.run(verbose=False)
        elapsed_time = time.time() - start_time
        
        # Calcular métricas
        portfolio_info = ts.get_portfolio_info(solution)
        
        # Métricas da convergência
        n_iterations = len(ts.history['iteration'])
        final_feasible = portfolio_info['is_feasible']
        
        # Proporção de iterações viáveis
        feasible_iters = sum(ts.history['is_feasible'])
        pct_feasible = feasible_iters / n_iterations if n_iterations > 0 else 0
        
        results.append({
            'strategy': config.name,
            'run': run + 1,
            'variance': portfolio_info['variance'],
            'std_dev': portfolio_info['std_dev'],
            'return': portfolio_info['return'],
            'sharpe': portfolio_info['sharpe_ratio'],
            'n_assets': portfolio_info['n_assets'],
            'is_feasible': final_feasible,
            'time_sec': elapsed_time,
            'iterations': n_iterations,
            'pct_feasible_iters': pct_feasible,
            'final_cost': solution.cost,
            # Parâmetros da estratégia
            'tabu_min': config.tabu_min,
            'tabu_max': config.tabu_max,
            'K': config.K,
            'H': config.H,
            'step_size': config.step_size,
            'use_idID': config.use_idID,
            'use_TID': config.use_TID,
        })
        
        print(f"Custo: {solution.cost:.6f} | Viável: {final_feasible} | Tempo: {elapsed_time:.2f}s")
    
    return pd.DataFrame(results)


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """Executa todos os experimentos e salva resultados."""
    
    print("\n" + "="*70)
    print("EXPERIMENTOS DE COMPARAÇÃO DE ESTRATÉGIAS - TABU SEARCH")
    print("="*70 + "\n")
    
    # ========================================================================
    # PARTE 1: EXPERIMENTOS COM DADOS SINTÉTICOS
    # ========================================================================
    
    print("="*70)
    print("PARTE 1: DADOS SINTÉTICOS")
    print("="*70 + "\n")
    
    # Configuração da instância sintética
    N_ASSETS_SYNTH = 100
    K_MAX_SYNTH = 15
    MIN_RETURN_SYNTH = 0.0005
    EPSILON = 0.01
    DELTA = 0.5
    N_RUNS = 10  # Número de execuções independentes por estratégia
    SEED = 42
    
    print("Configuração dos experimentos sintéticos:")
    print(f"  Número de ativos: {N_ASSETS_SYNTH}")
    print(f"  Cardinalidade máxima: {K_MAX_SYNTH}")
    print(f"  Retorno mínimo: {MIN_RETURN_SYNTH}")
    print(f"  Execuções por estratégia: {N_RUNS}")
    print(f"  Total de estratégias: {len(STRATEGIES)}")
    print(f"  Total de runs: {len(STRATEGIES) * N_RUNS}")
    
    # Gerar dados sintéticos
    print("\nGerando dados sintéticos...")
    returns_synth, cov_synth, tickers_synth = gerar_dados_sinteticos(
        n_ativos=N_ASSETS_SYNTH, 
        semente=SEED
    )
    print(f"Dados gerados: {N_ASSETS_SYNTH} ativos")
    
    # Executar experimentos sintéticos
    all_results_synth = []
    
    start_synth = time.time()
    
    for i, config in enumerate(STRATEGIES, 1):
        print(f"\n[{i}/{len(STRATEGIES)}] Testando estratégia: {config.name}")
        
        results_df = run_single_experiment(
            config=config,
            returns=returns_synth,
            cov_matrix=cov_synth,
            k_max=K_MAX_SYNTH,
            min_return=MIN_RETURN_SYNTH,
            epsilon=EPSILON,
            delta=DELTA,
            n_runs=N_RUNS,
            seed_base=SEED
        )
        
        # Adicionar tipo de dados
        results_df['data_type'] = 'synthetic'
        results_df['market'] = 'synthetic'
        
        all_results_synth.append(results_df)
    
    time_synth = time.time() - start_synth
    
    print(f"\n{'='*70}")
    print(f"DADOS SINTÉTICOS CONCLUÍDOS - Tempo: {time_synth:.1f}s ({time_synth/60:.1f} min)")
    print(f"{'='*70}")
    
    # ========================================================================
    # PARTE 2: EXPERIMENTOS COM DADOS REAIS
    # ========================================================================
    
    print("\n" + "="*70)
    print("PARTE 2: DADOS REAIS")
    print("="*70 + "\n")
    
    # Selecionar apenas as melhores estratégias para dados reais
    # (para não demorar muito)
    BEST_STRATEGIES = [
        STRATEGIES[2],  # idID_TID (combinação completa)
        STRATEGIES[5],  # idID_TID_aggressive_penalty
        STRATEGIES[7],  # idID_TID_large_step
    ]
    
    N_RUNS_REAL = 5  # Menos runs para dados reais
    
    print(f"Configuração dos experimentos reais:")
    print(f"  Estratégias selecionadas: {len(BEST_STRATEGIES)}")
    print(f"  Execuções por estratégia: {N_RUNS_REAL}")
    print(f"  Mercados: S&P 500 e Ibovespa")
    
    all_results_real = []
    
    # Lista de mercados para testar
    markets_config = [
        {
            'name': 'sp500',
            'n_assets': 30,
            'k_max': 10,
            'min_return': 0.0003,
        },
        {
            'name': 'ibovespa',
            'n_assets': 20,
            'k_max': 8,
            'min_return': 0.0003,
        }
    ]
    
    start_real = time.time()
    
    for market_cfg in markets_config:
        market_name = market_cfg['name']
        
        print(f"\n{'='*70}")
        print(f"MERCADO: {market_name.upper()}")
        print(f"{'='*70}")
        
        # Carregar dados reais
        data = load_real_data(
            market=market_name,
            n_assets=market_cfg['n_assets'],
            verbose=True
        )
        
        if data is None:
            print(f"⚠️  Pulando mercado {market_name} - dados não disponíveis")
            continue
        
        returns_real, cov_real, tickers_real = data
        
        print(f"\nExecutando experimentos com {len(tickers_real)} ativos...")
        
        # Executar apenas as melhores estratégias
        for i, config in enumerate(BEST_STRATEGIES, 1):
            print(f"\n[{i}/{len(BEST_STRATEGIES)}] Testando estratégia: {config.name}")
            
            results_df = run_single_experiment(
                config=config,
                returns=returns_real,
                cov_matrix=cov_real,
                k_max=market_cfg['k_max'],
                min_return=market_cfg['min_return'],
                epsilon=EPSILON,
                delta=DELTA,
                n_runs=N_RUNS_REAL,
                seed_base=SEED
            )
            
            # Adicionar informações do mercado
            results_df['data_type'] = 'real'
            results_df['market'] = market_name
            results_df['n_assets_available'] = len(tickers_real)
            
            all_results_real.append(results_df)
    
    time_real = time.time() - start_real
    
    print(f"\n{'='*70}")
    print(f"DADOS REAIS CONCLUÍDOS - Tempo: {time_real:.1f}s ({time_real/60:.1f} min)")
    print(f"{'='*70}")
    
    # ========================================================================
    # CONSOLIDAÇÃO DOS RESULTADOS
    # ========================================================================
    
    total_time = time_synth + time_real
    
    # Consolidar todos os resultados
    all_results = all_results_synth + all_results_real
    results_df = pd.concat(all_results, ignore_index=True)
    
    # Calcular estatísticas agregadas por estratégia e tipo de dado
    summary = results_df.groupby(['data_type', 'market', 'strategy']).agg({
        'variance': ['mean', 'std', 'min'],
        'std_dev': ['mean', 'std'],
        'return': ['mean', 'std'],
        'sharpe': ['mean', 'std', 'max'],
        'n_assets': ['mean', 'std'],
        'is_feasible': ['sum', 'mean'],
        'time_sec': ['mean', 'std'],
        'iterations': ['mean', 'std'],
        'pct_feasible_iters': ['mean', 'std'],
        'final_cost': ['mean', 'std', 'min'],
    }).round(6)
    
    # Salvar resultados
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Resultados detalhados
    detailed_file = f'results_detailed_{timestamp}.csv'
    results_df.to_csv(detailed_file, index=False)
    print(f"\nResultados detalhados salvos em: {detailed_file}")
    
    # Sumário estatístico
    summary_file = f'results_summary_{timestamp}.csv'
    summary.to_csv(summary_file)
    print(f"Sumário estatístico salvo em: {summary_file}")
    
    # Salvar configurações
    config_file = f'experiment_config_{timestamp}.json'
    config_data = {
        'synthetic': {
            'n_assets': N_ASSETS_SYNTH,
            'k_max': K_MAX_SYNTH,
            'min_return': MIN_RETURN_SYNTH,
            'epsilon': EPSILON,
            'delta': DELTA,
            'n_runs': N_RUNS,
            'seed': SEED,
        },
        'real_markets': markets_config,
        'n_runs_real': N_RUNS_REAL,
        'strategies': [
            {
                'name': cfg.name,
                'use_idID': cfg.use_idID,
                'use_TID': cfg.use_TID,
                'tabu_min': cfg.tabu_min,
                'tabu_max': cfg.tabu_max,
                'max_iter': cfg.max_iter,
                'K': cfg.K,
                'H': cfg.H,
                'step_size': cfg.step_size,
                'step_variation': cfg.step_variation,
            }
            for cfg in STRATEGIES
        ]
    }
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    print(f"Configurações salvas em: {config_file}")
    
    # ========================================================================
    # EXIBIR RESUMOS
    # ========================================================================
    
    print("\n" + "="*70)
    print("RESUMO GERAL DOS RESULTADOS")
    print("="*70)
    print(f"\nTempo total de execução: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  - Dados sintéticos: {time_synth:.1f}s ({time_synth/60:.1f} min)")
    print(f"  - Dados reais: {time_real:.1f}s ({time_real/60:.1f} min)")
    
    # Resumo por tipo de dado
    print("\n" + "="*70)
    print("RANKING POR SHARPE RATIO - DADOS SINTÉTICOS")
    print("="*70)
    
    synth_results = results_df[results_df['data_type'] == 'synthetic']
    if len(synth_results) > 0:
        synth_summary = synth_results.groupby('strategy').agg({
            'sharpe': 'mean',
            'variance': 'mean',
            'is_feasible': 'mean',
            'time_sec': 'mean'
        }).sort_values('sharpe', ascending=False)
        
        for i, (strategy, row) in enumerate(synth_summary.iterrows(), 1):
            print(f"{i:2d}. {strategy:30s} | Sharpe: {row['sharpe']:7.4f} | "
                  f"Viável: {row['is_feasible']*100:5.1f}% | "
                  f"Variância: {row['variance']:.6f} | "
                  f"Tempo: {row['time_sec']:5.2f}s")
    
    # Resumo de dados reais
    real_results = results_df[results_df['data_type'] == 'real']
    if len(real_results) > 0:
        print("\n" + "="*70)
        print("RANKING POR SHARPE RATIO - DADOS REAIS")
        print("="*70)
        
        for market in real_results['market'].unique():
            market_data = real_results[real_results['market'] == market]
            
            print(f"\nMercado: {market.upper()}")
            print("-" * 70)
            
            market_summary = market_data.groupby('strategy').agg({
                'sharpe': 'mean',
                'variance': 'mean',
                'is_feasible': 'mean',
                'time_sec': 'mean',
                'n_assets': 'mean'
            }).sort_values('sharpe', ascending=False)
            
            for i, (strategy, row) in enumerate(market_summary.iterrows(), 1):
                print(f"{i:2d}. {strategy:30s} | Sharpe: {row['sharpe']:7.4f} | "
                      f"Viável: {row['is_feasible']*100:5.1f}% | "
                      f"Ativos: {row['n_assets']:.1f} | "
                      f"Tempo: {row['time_sec']:5.2f}s")
    
    print("\n" + "="*70)
    print("EXPERIMENTOS CONCLUÍDOS!")
    print("="*70)
    
    return results_df, summary


if __name__ == "__main__":
    results, summary = main()
