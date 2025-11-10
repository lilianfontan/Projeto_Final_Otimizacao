import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class PortfolioDataLoader:
    """
    Classe para download e processamento de dados de ativos do Yahoo Finance
    """
    
    def __init__(self, tickers, start_date=None, end_date=None):
        """
        Inicializa o loader de dados
        
        Args:
            tickers: Lista de símbolos dos ativos ou string única
            start_date: Data inicial (formato 'YYYY-MM-DD')
            end_date: Data final (formato 'YYYY-MM-DD')
        """
        self.tickers = tickers if isinstance(tickers, list) else [tickers]
        self.start_date = start_date or (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.prices = None
        self.returns = None
        self.valid_tickers = []
        
    def download_data(self, min_data_threshold=0.7, verbose=False):
        """
        Baixa os preços de fechamento ajustados dos ativos
        
        Args:
            min_data_threshold: Porcentagem mínima de dados válidos (0.0 a 1.0)
            verbose: Se True, imprime informações sobre o download
            
        Returns:
            DataFrame com preços válidos ou None se nenhum dado válido
        """
        if verbose:
            print(f"Baixando dados de {len(self.tickers)} ativos...")
            print(f"Período: {self.start_date} a {self.end_date}")
        
        try:
            # Tentar diferentes formas de download
            all_prices = pd.DataFrame()
            
            # Estratégia 1: Download individual (mais confiável)
            for ticker in self.tickers:
                try:
                    data = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                    
                    if data is not None and len(data) > 0:
                        # Extrair Adj Close ou Close
                        if 'Adj Close' in data.columns:
                            prices_col = data['Adj Close']
                        elif 'Close' in data.columns:
                            prices_col = data['Close']
                        else:
                            # Se tem apenas uma coluna, usar ela
                            prices_col = data.iloc[:, 0]
                        
                        # Adicionar ao DataFrame
                        all_prices[ticker] = prices_col
                        
                        if verbose:
                            print(f"  ✓ {ticker}: {len(prices_col)} dias")
                            
                except Exception as e:
                    if verbose:
                        print(f"  ✗ {ticker}: {str(e)[:50]}")
                    continue
            
            if len(all_prices.columns) == 0:
                if verbose:
                    print("⚠️ Nenhum ativo baixado com sucesso")
                return None
            
            self.prices = all_prices
            
            # Remover colunas com muitos NaN
            min_valid_rows = int(len(self.prices) * min_data_threshold)
            valid_columns = []
            
            for col in self.prices.columns:
                valid_count = self.prices[col].notna().sum()
                if valid_count >= min_valid_rows:
                    valid_columns.append(col)
                elif verbose:
                    print(f"  ✗ {col}: dados insuficientes ({valid_count}/{len(self.prices)})")
            
            if len(valid_columns) == 0:
                if verbose:
                    print(f"⚠️ Nenhum ativo tem dados suficientes (mín: {min_data_threshold*100:.0f}%)")
                return None
            
            # Filtrar apenas colunas válidas
            self.prices = self.prices[valid_columns]
            
            # Remover linhas com qualquer NaN
            rows_before = len(self.prices)
            self.prices = self.prices.dropna(axis=0)
            
            if verbose and rows_before > len(self.prices):
                print(f"  Removidas {rows_before - len(self.prices)} linhas com NaN")
            
            if len(self.prices) == 0:
                if verbose:
                    print("⚠️ Nenhuma linha válida após remoção de NaN")
                return None
            
            # Atualizar lista de tickers válidos
            self.valid_tickers = list(self.prices.columns)
            
            if verbose:
                print(f"\n✓ {len(self.valid_tickers)} ativos com dados válidos")
                print(f"  {len(self.prices)} dias de dados")
                print(f"  Tickers: {', '.join(self.valid_tickers)}")
            
            return self.prices
            
        except Exception as e:
            if verbose:
                print(f"❌ Erro ao baixar dados: {e}")
            return None
    
    def calculate_returns(self):
        """
        Calcula os retornos diários dos ativos
        
        Returns:
            DataFrame com retornos ou None se sem dados
        """
        if self.prices is None:
            self.download_data()
        
        if self.prices is None or len(self.prices) == 0:
            return None
        
        # Usar fill_method=None para evitar warning
        try:
            self.returns = self.prices.pct_change(fill_method=None).dropna()
        except TypeError:
            # Versão antiga do pandas
            self.returns = self.prices.pct_change().dropna()
        
        if len(self.returns) == 0:
            return None
            
        return self.returns
    
    def get_expected_returns(self):
        """
        Calcula o vetor de retornos esperados (média dos retornos)
        
        Returns:
            Array numpy com retornos esperados ou None se sem dados
        """
        if self.returns is None:
            self.calculate_returns()
        
        if self.returns is None or len(self.returns) == 0:
            return None
            
        return self.returns.mean().values
    
    def get_covariance_matrix(self):
        """
        Calcula a matriz de covariância dos retornos
        
        Returns:
            Array numpy 2D com matriz de covariância ou None se sem dados
        """
        if self.returns is None:
            self.calculate_returns()
        
        if self.returns is None or len(self.returns) == 0:
            return None
            
        return self.returns.cov().values
    
    def get_valid_tickers(self):
        """
        Retorna lista de tickers que têm dados válidos
        
        Returns:
            Lista de strings com tickers válidos
        """
        return self.valid_tickers
    
    def summary(self):
        """
        Retorna resumo dos dados carregados
        
        Returns:
            Dicionário com estatísticas
        """
        if self.returns is None:
            self.calculate_returns()
        
        if self.returns is None:
            return None
        
        returns_annual = self.returns.mean() * 252
        volatility_annual = self.returns.std() * np.sqrt(252)
        
        summary = {
            'n_assets': len(self.valid_tickers),
            'tickers': self.valid_tickers,
            'n_days': len(self.returns),
            'avg_return_annual': returns_annual.mean(),
            'avg_volatility_annual': volatility_annual.mean(),
            'min_return': returns_annual.min(),
            'max_return': returns_annual.max(),
            'min_volatility': volatility_annual.min(),
            'max_volatility': volatility_annual.max()
        }
        
        return summary
