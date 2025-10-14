# Otimização de Portfólios utilizando Tabu Search

**Disciplina:** Tópicos em Otimização Combinatória  
**Autores:** Lilian Fontan de Oliveira e Maurício P. Lopes  
**Data:** Outubro de 2025

## 📋 Sobre o Projeto

Este projeto investiga e implementa técnicas de otimização combinatória aplicadas ao problema de seleção de portfólios de investimento, com ênfase no método **Tabu Search**. O objetivo é determinar a alocação ótima de recursos entre ativos financeiros, equilibrando risco e retorno esperado sob restrições realistas.

## 🎯 Motivação

O modelo clássico de média-variância de Markowitz (1952), embora teoricamente elegante, torna-se computacionalmente intratável quando incorpora restrições práticas como:

- **Cardinalidade**: número máximo de ativos na carteira
- **Limites de alocação**: proporções mínimas e máximas por ativo
- **Custos de transação**: penalidades de compra, venda e rebalanceamento

Essas restrições transformam o problema em uma formulação de programação inteira mista (MIP) de alta complexidade, justificando o uso de metaheurísticas.

## 📐 Formulação Matemática

O problema é formulado como:

$\min \quad \sum_{i=1}^n \sum_{j=1}^n \sigma_{ij} x_i x_j$

sujeito a:
- $Σᵢ rᵢ xᵢ ≥ R$           (retorno mínimo)
- $Σᵢ xᵢ = 1$              (soma unitária)
- $εᵢ zᵢ ≤ xᵢ ≤ δᵢ zᵢ$    (limites de alocação)
- $Σᵢ zᵢ ≤ k$              (cardinalidade máxima)
- $zᵢ ∈ {0,1}$             (variáveis binárias)


Onde:
- **xᵢ**: fração de capital investida no ativo i
- **rᵢ**: retorno esperado do ativo i
- **σᵢⱼ**: covariância entre os retornos dos ativos i e j
- **R**: retorno mínimo exigido
- **εᵢ, δᵢ**: limites mínimo e máximo de alocação
- **zᵢ**: variável binária indicando se o ativo i está incluído
- **k**: número máximo de ativos permitidos

## 🔧 Metodologia

### Tabu Search Adaptativo

A implementação utiliza:

- **Representação híbrida**: vetor binário (ativos selecionados) + vetor contínuo (proporções)
- **Operadores de vizinhança**: 
  - Adição de ativo
  - Remoção de ativo
  - Troca de ativo
- **Lista tabu**: memória de curto prazo para evitar ciclos
- **Critério de aspiração**: permite movimentos proibidos se melhorarem a melhor solução global
- **Estratégias avançadas**:
  - Intensificação adaptativa (reinicialização a partir de elite set)
  - Diversificação probabilística (exploração de novas regiões)
  - Penalização dinâmica (ajuste automático de restrições)

## 📊 Avaliação

### Instâncias de Teste
- **Reais**: dados históricos de índices como S&P 500 e IBrX 100 (Brasil)
- **Sintéticas**: matrizes de covariância controladas com diferentes níveis de correlação
- **Tamanho**: entre 50 e 200 ativos
- **Cardinalidade**: variando entre 10% e 30% do total de ativos

### Métricas de Desempenho
- Retorno esperado E[R]
- Risco (variância do portfólio)
- Índice de Sharpe (eficiência risco-retorno)
- Desvio percentual em relação à fronteira eficiente
- Tempo computacional médio
- Estabilidade das soluções (múltiplas execuções)

## 🛠️ Implementação

**Linguagem:** Python 3.8+  
**Bibliotecas principais:** 
- NumPy (manipulação de matrizes)
- pandas (processamento de dados)
- Matplotlib (visualização)


## 📈 Resultados Esperados

O projeto visa demonstrar que a Tabu Search adaptativa é capaz de:

- Gerar portfólios próximos à fronteira eficiente teórica
- Operar eficientemente em instâncias com 50-200 ativos
- Convergir para soluções de qualidade em tempo computacional reduzido
- Manter estabilidade e robustez sob diferentes parametrizações

## 📚 Referências

- **Markowitz, H. M. (1952)**. Portfolio selection. *Handbook of Finance*. Wiley, 2008.
- **Glover, F. and Laguna, M. (1997)**. Tabu Search. *Operations Research/Computer Science Interfaces Series*. Springer, Boston, MA.
- **Schaerf, A. (2001)**. A survey of tabu search metaheuristics for portfolio selection. *Journal of Heuristics*, 7(2):139-172.

## 📄 Licença

Este projeto foi desenvolvido como trabalho acadêmico para a disciplina de Tópicos em Otimização Combinatória. 

---
