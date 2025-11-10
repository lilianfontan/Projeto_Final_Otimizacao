# PortfolioDataLoader --- Resumo e Uso

Este arquivo descreve de forma objetiva o funcionamento do script
`data_loader.py` e como utilizá-lo em projetos de análise de portfólio.

## ✅ O que o script faz

A classe **`PortfolioDataLoader`** realiza:

1.  **Download de preços históricos** de ativos a partir do Yahoo
    Finance.\
2.  **Limpeza automática dos dados**, removendo ativos ou linhas com
    dados insuficientes.\
3.  **Cálculo dos retornos diários**.\
4.  Geração do **vetor de retornos esperados** (médias).\
5.  Cálculo da **matriz de covariância**.\
6.  Resumo estatístico anualizado do portfólio.

## ✅ Como usar

``` python
from data_loader import PortfolioDataLoader

# Lista de ativos
tickers = ["AAPL", "MSFT", "GOOG"]

# Criar loader
loader = PortfolioDataLoader(tickers)

# Baixar dados
prices = loader.download_data(verbose=True)

# Calcular retornos
returns = loader.calculate_returns()

# Obter retornos esperados
mu = loader.get_expected_returns()

# Obter matriz de covariância
cov = loader.get_covariance_matrix()

# Resumo estatístico
summary = loader.summary()
print(summary)
```

## ✅ Principais métodos

  -----------------------------------------------------------------------
  Método                      Descrição
  --------------------------- -------------------------------------------
  `download_data()`           Faz o download e filtra ativos com dados
                              suficientes.

  `calculate_returns()`       Calcula retornos percentuais diários.

  `get_expected_returns()`    Retorno médio esperado para cada ativo.

  `get_covariance_matrix()`   Covariância entre os retornos.

  `get_valid_tickers()`       Lista final de tickers válidos.

  `summary()`                 Estatísticas anualizadas do portfólio.
  -----------------------------------------------------------------------

## ✅ Requisitos

-   Python 3\
-   `yfinance`\
-   `pandas`\
-   `numpy`

Instalação:

``` bash
pip install yfinance pandas numpy
```

------------------------------------------------------------------------

Arquivo gerado automaticamente para documentação de repositórios GitHub.
