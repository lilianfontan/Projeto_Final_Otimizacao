# Script de Experimentos - Comparação de Estratégias Tabu Search

## Objetivo

Este script executa experimentos comparando diferentes estratégias do Tabu Search para otimização de portfólios, seguindo a metodologia do artigo de Schaerf (2001) "Local Search Techniques for Constrained Portfolio Selection".

O script executa experimentos tanto com **dados sintéticos** quanto com **dados reais** de mercado (S&P 500 e Ibovespa).

## Estrutura dos Experimentos

### PARTE 1: Dados Sintéticos

Testa **9 estratégias** com dados gerados artificialmente:

1. **idID_only**: Apenas vizinhança idID (increase/decrease/Insert/Delete)
2. **TID_only**: Apenas vizinhança TID (Transfer/Insert/Delete)
3. **idID_TID**: Combinação completa de ambas as vizinhanças
4. **idID_TID_short_tabu**: Lista tabu curta (5-15)
5. **idID_TID_long_tabu**: Lista tabu longa (15-35)
6. **idID_TID_aggressive_penalty**: Shifting penalty agressivo (K=10, H=1)
7. **idID_TID_conservative_penalty**: Shifting penalty conservador (K=30, H=3)
8. **idID_TID_large_step**: Step size maior (0.5 ± 0.3)
9. **idID_TID_small_step**: Step size menor (0.15 ± 0.15)

**Configuração:**
- 100 ativos sintéticos
- Cardinalidade máxima: 15
- 10 runs por estratégia
- Total: 90 execuções

### PARTE 2: Dados Reais

Testa as **3 melhores estratégias** identificadas nos dados sintéticos:

1. **idID_TID**: Estratégia completa (baseline)
2. **idID_TID_aggressive_penalty**: Penalização agressiva
3. **idID_TID_large_step**: Movimentos maiores

**Mercados testados:**
- **S&P 500**: 30 ativos (AAPL, MSFT, GOOGL, etc.)
  - Cardinalidade máxima: 10
- **Ibovespa**: 20 ativos (PETR4, VALE3, ITUB4, etc.)
  - Cardinalidade máxima: 8

**Configuração:**
- 5 runs por estratégia por mercado
- Dados reais dos últimos 2 anos (Yahoo Finance)
- Total: 30 execuções (3 estratégias × 2 mercados × 5 runs)

## Como Executar

```bash
python run_experiments.py
```

## Parâmetros dos Experimentos

### Dados Sintéticos
- **Número de ativos**: 100
- **Cardinalidade máxima (k)**: 15
- **Retorno mínimo**: 0.0005
- **Epsilon (alocação mínima)**: 0.01 (1%)
- **Delta (alocação máxima)**: 0.5 (50%)
- **Execuções por estratégia**: 10 runs independentes
- **Iterações máximas**: 500 por run

### Dados Reais
- **S&P 500**:
  - Ativos: até 30 (AAPL, MSFT, GOOGL, AMZN, NVDA, etc.)
  - Cardinalidade máxima: 10
  - Retorno mínimo: 0.0003
- **Ibovespa**:
  - Ativos: até 20 (PETR4, VALE3, ITUB4, BBDC4, etc.)
  - Cardinalidade máxima: 8
  - Retorno mínimo: 0.0003
- **Período**: Últimos 2 anos de dados históricos
- **Execuções por estratégia**: 5 runs independentes
- **Iterações máximas**: 500 por run

## Resultados Gerados

O script gera 3 arquivos com timestamp:

### 1. results_detailed_YYYYMMDD_HHMMSS.csv

Resultados detalhados de todas as execuções, contendo:
- strategy: Nome da estratégia
- run: Número da execução
- variance: Variância do portfólio
- std_dev: Desvio padrão
- return: Retorno esperado
- sharpe: Índice de Sharpe
- n_assets: Número de ativos no portfólio
- is_feasible: Se a solução é viável
- time_sec: Tempo de execução em segundos
- iterations: Número de iterações realizadas
- pct_feasible_iters: Percentual de iterações viáveis
- final_cost: Custo final (objetivo)
- **data_type**: 'synthetic' ou 'real'
- **market**: 'synthetic', 'sp500' ou 'ibovespa'
- **n_assets_available**: Total de ativos disponíveis (apenas dados reais)
- Parâmetros da estratégia (tabu_min, tabu_max, K, H, step_size, etc.)

### 2. results_summary_YYYYMMDD_HHMMSS.csv

Estatísticas agregadas por **estratégia, tipo de dado e mercado** (média, desvio padrão, min/max) para todas as métricas.

Permite comparar:
- Desempenho em dados sintéticos vs reais
- Diferenças entre mercados (S&P 500 vs Ibovespa)
- Robustez das estratégias em diferentes cenários

### 3. experiment_config_YYYYMMDD_HHMMSS.json

Configurações completas do experimento para reprodutibilidade, incluindo:
- Parâmetros dos experimentos sintéticos
- Configuração de cada mercado real
- Definição completa de cada estratégia testada

## Métricas de Avaliação

As principais métricas comparadas são:

1. **Sharpe Ratio**: Eficiência risco-retorno
2. **Variância**: Risco do portfólio
3. **Retorno**: Retorno esperado
4. **Viabilidade**: Percentual de soluções viáveis encontradas
5. **Tempo Computacional**: Eficiência do algoritmo
6. **Estabilidade**: Variabilidade entre diferentes execuções

## Análise dos Resultados

O script exibe ao final **dois rankings** por Sharpe Ratio:

### 1. Ranking - Dados Sintéticos
Compara todas as 9 estratégias mostrando:
- Sharpe ratio médio
- Percentual de soluções viáveis
- Variância média
- Tempo médio de execução

### 2. Ranking - Dados Reais (por mercado)
Para cada mercado (S&P 500 e Ibovespa), compara as 3 melhores estratégias:
- Sharpe ratio médio
- Percentual de soluções viáveis
- Número médio de ativos selecionados
- Tempo médio de execução

## Comparações Possíveis

Com os resultados gerados, você pode analisar:

### Dados Sintéticos vs Reais
- Como as estratégias se comportam em diferentes tipos de dados
- Robustez das abordagens em cenários reais vs controlados
- Impacto da estrutura de correlação real dos mercados

### Comparação entre Mercados
- S&P 500 vs Ibovespa
- Diferenças de volatilidade e estrutura de correlação
- Adaptabilidade das estratégias a diferentes mercados

### Análise de Estratégias
- Eficácia de diferentes vizinhanças (idID vs TID)
- Impacto do tamanho da lista tabu
- Efeito do shifting penalty mechanism
- Influência do step size nos movimentos

## Tempo Estimado

Com as configurações padrão:

### Dados Sintéticos
- 9 estratégias × 10 runs × 500 iterações
- Tempo estimado: 15-25 minutos

### Dados Reais
- 3 estratégias × 2 mercados × 5 runs × 500 iterações
- Download de dados: 2-5 minutos (primeira vez)
- Otimização: 8-15 minutos
- **Total estimado: 10-20 minutos**

**Tempo total: 25-45 minutos** (depende do hardware e conexão de internet)

## Personalizar Experimentos

### Modificar Estratégias Testadas
Para adicionar novas estratégias, inclua objetos `ExperimentConfig` na lista `STRATEGIES`:

```python
ExperimentConfig(
    name="minha_estrategia",
    use_idID=True,
    use_TID=True,
    tabu_min=10,
    tabu_max=25,
    K=20,
    H=1,
    step_size=0.3,
    step_variation=0.3
)
```

### Modificar Parâmetros dos Dados Sintéticos
No início da função `main()`:
```python
N_ASSETS_SYNTH = 100    # Número de ativos
K_MAX_SYNTH = 15        # Cardinalidade máxima  
MIN_RETURN_SYNTH = 0.0005  # Retorno mínimo
N_RUNS = 10             # Execuções por estratégia
```

### Modificar Mercados Reais
Altere a lista `markets_config` em `main()`:
```python
markets_config = [
    {
        'name': 'sp500',
        'n_assets': 30,      # Máximo de ativos a tentar carregar
        'k_max': 10,         # Cardinalidade máxima
        'min_return': 0.0003,  # Retorno mínimo
    },
    # Adicione mais mercados aqui
]
```

### Selecionar Estratégias para Dados Reais
Modifique `BEST_STRATEGIES` para escolher quais estratégias executar com dados reais:
```python
BEST_STRATEGIES = [
    STRATEGIES[0],  # Índice da estratégia desejada
    STRATEGIES[2],
    # ...
]
```

## Requisitos

### Bibliotecas Python
```bash
pip install numpy pandas yfinance
```

### Arquivos do Projeto
O script requer os seguintes módulos do projeto:
- `tabu_search.py`: Implementação do Tabu Search
- `experimentos.py`: Funções auxiliares (gerar_dados_sinteticos)
- `data_loader.py`: Carregamento de dados reais (PortfolioDataLoader)

## Próximos Passos: Análise e Visualização

Com os resultados CSV gerados, você pode criar:

### 1. Gráficos de Comparação (estilo Schaerf)
- **Convergência**: Custo ao longo das iterações
- **Risco-Retorno**: Dispersão dos portfólios encontrados
- **Box plots**: Distribuição de tempo/qualidade por estratégia
- **Barras comparativas**: Sharpe ratio, tempo, viabilidade

### 2. Análise Estatística
- Testes de significância (t-test, ANOVA)
- Análise de variância entre estratégias
- Correlações entre parâmetros e desempenho
- Intervalo de confiança das métricas

### 3. Tabelas Comparativas (estilo artigo)
- Melhor/média/pior solução por estratégia
- Tempo médio ± desvio padrão
- Taxa de sucesso (viabilidade)
- Comparação sintético vs real
- Comparação entre mercados

### 4. Análise Específica de Dados Reais
- Composição dos portfólios (quais ativos selecionados)
- Setores/indústrias representados
- Comparação com benchmarks de mercado
- Fronteira eficiente empírica vs teórica

- Schaerf, A. (2001). Local Search Techniques for Constrained Portfolio Selection Problems
- Glover, F., & Laguna, M. (1997). Tabu Search
