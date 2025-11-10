# Explicação Completa: Tabu Search para Otimização de Portfólios

## Índice

1. [Introdução e Contexto](#1-introdução-e-contexto)
2. [O Problema de Otimização](#2-o-problema-de-otimização)
3. [Visão Geral do Tabu Search](#3-visão-geral-do-tabu-search)
4. [Representação da Solução](#4-representação-da-solução)
5. [Lista Tabu (Memória de Curto Prazo)](#5-lista-tabu)
6. [Shifting Penalty Mechanism](#6-shifting-penalty-mechanism)
7. [Operadores de Vizinhança](#7-operadores-de-vizinhança)
8. [Função Objetivo e Avaliação](#8-função-objetivo-e-avaliação)
9. [Critério de Aspiração](#9-critério-de-aspiração)
10. [Loop Principal do Algoritmo](#10-loop-principal-do-algoritmo)
11. [Estratégias do Artigo de Schaerf](#11-estratégias-do-artigo-de-schaerf)
12. [Parâmetros e Configurações](#12-parâmetros-e-configurações)
13. [Análise de Resultados](#13-análise-de-resultados)

---

## 1. Introdução e Contexto

### 1.1 O Problema de Portfólios

O problema de seleção de portfólios consiste em determinar como alocar capital entre diferentes ativos financeiros de forma a:
- **Maximizar** o retorno esperado
- **Minimizar** o risco (variância)
- Satisfazer restrições práticas de investimento

### 1.2 Modelo de Markowitz (1952)

O modelo clássico propõe:

```
min  x^T Σ x                    (variância do portfólio)
s.a. r^T x ≥ R                  (retorno mínimo)
     Σ xᵢ = 1                   (todo capital investido)
     xᵢ ≥ 0                     (sem vendas a descoberto)
```

### 1.3 Extensões Realistas

Para capturar restrições do mundo real, adicionamos:

**Restrição de Cardinalidade:**
```
Σ zᵢ ≤ k
onde zᵢ = 1 se ativo i está no portfólio, 0 caso contrário
```

**Restrição de Quantidade:**
```
εᵢzᵢ ≤ xᵢ ≤ δᵢzᵢ
```

Essas restrições transformam o problema em **programação inteira mista não-linear**, tornando-o NP-difícil.

---

## 2. O Problema de Otimização

### 2.1 Formulação Matemática Completa

**Dados de Entrada:**
- `n`: número de ativos disponíveis
- `μ = [μ₁, ..., μₙ]`: vetor de retornos esperados
- `Σ`: matriz de covariância (n×n)
- `R`: retorno mínimo desejado
- `k`: cardinalidade máxima
- `ε`: proporção mínima por ativo
- `δ`: proporção máxima por ativo

**Variáveis de Decisão:**
- `x = [x₁, ..., xₙ]`: proporções de investimento (contínuas)
- `z = [z₁, ..., zₙ]`: ativos selecionados (binárias)

**Problema:**
```
min  f(x, z) = x^T Σ x

s.a. (1) μ^T x ≥ R                    (retorno mínimo)
     (2) Σ xᵢ = 1                     (orçamento)
     (3) 0 ≤ xᵢ ≤ 1, ∀i               (limites gerais)
     (4) εᵢzᵢ ≤ xᵢ ≤ δᵢzᵢ, ∀i         (quantidade)
     (5) Σ zᵢ ≤ k                     (cardinalidade)
     (6) zᵢ ∈ {0, 1}, ∀i              (binária)
```

### 2.2 Dificuldades do Problema

1. **Não-linearidade:** Função objetivo quadrática
2. **Variáveis mistas:** Contínuas (x) + Binárias (z)
3. **Combinatória:** C(n, k) combinações possíveis
4. **Múltiplas restrições:** Interação complexa

**Exemplo:** Para n=100 ativos e k=10:
- Combinações: C(100, 10) ≈ 1.7 × 10¹³
- Inviável resolver por enumeração exaustiva

---

## 3. Visão Geral do Tabu Search

### 3.1 Princípio Fundamental

O Tabu Search é uma **metaheurística de busca local** que:

1. Parte de uma solução inicial
2. A cada iteração, move-se para um **vizinho** da solução atual
3. Usa **memória** (lista tabu) para evitar ciclos
4. Aceita **movimentos de piora** para escapar de ótimos locais
5. Mantém a **melhor solução** encontrada

### 3.2 Diferenças de Outros Métodos

| Característica | Hill Climbing | Simulated Annealing | Tabu Search |
|----------------|---------------|---------------------|-------------|
| Aceita piora? | Não | Sim (probabilístico) | Sim (sempre) |
| Usa memória? | Não | Não | **Sim** |
| Estratégia | Gulosa | Aleatória | **Guiada** |
| Escapa ótimos locais? | Não | Sim | **Sim** |

### 3.3 Fluxo Geral

```
┌─────────────────┐
│ Solução Inicial │
└────────┬────────┘
         │
    ┌────▼────┐
    │  Loop   │◄──────────┐
    │Principal│           │
    └────┬────┘           │
         │                │
    ┌────▼──────────┐     │
    │ Gera Vizinhos │     │
    └────┬──────────┘     │
         │                │
    ┌────▼──────────────┐ │
    │ Seleciona Melhor  │ │
    │  (não-tabu ou     │ │
    │   aspiração)      │ │
    └────┬──────────────┘ │
         │                │
    ┌────▼──────────┐     │
    │ Executa Movimento│  │
    └────┬──────────┘     │
         │                │
    ┌────▼──────────┐     │
    │ Adiciona à    │     │
    │ Lista Tabu    │     │
    └────┬──────────┘     │
         │                │
    ┌────▼──────────┐     │
    │ Atualiza      │     │
    │ Penalização   │     │
    └────┬──────────┘     │
         │                │
    ┌────▼──────────┐     │
    │ Atualiza      │     │
    │ Melhor Global │     │
    └────┬──────────┘     │
         │                │
    ┌────▼──────────┐     │
    │  Continua?    ├─────┘
    └────┬──────────┘
         │ Não
    ┌────▼────┐
    │ Retorna │
    │  Melhor │
    └─────────┘
```

---

## 4. Representação da Solução

### 4.1 Estrutura Híbrida

A solução é representada por **dois vetores**:

**Vetor Binário (z):**
```python
z = [z₁, z₂, ..., zₙ]
onde zᵢ = 1 se ativo i está no portfólio
         0 caso contrário
```

**Vetor de Proporções (x):**
```python
x = [x₁, x₂, ..., xₙ]
onde xᵢ = fração do capital investida em ativo i
     Σ xᵢ = 1
```

### 4.2 Exemplo

Para n=5 ativos e k=3:

```python
z = [1, 0, 1, 1, 0]
x = [0.30, 0.00, 0.45, 0.25, 0.00]

Interpretação:
- Ativos selecionados: {1, 3, 4}
- 30% em ativo 1
- 45% em ativo 3
- 25% em ativo 4
- Total: 100%
```

### 4.3 Classe Solution

```python
@dataclass
class Solution:
    z: np.ndarray  # [n] binário
    x: np.ndarray  # [n] contínuo
    cost: float    # f(x, z)
```

**Propriedades garantidas:**
- `Σ xᵢ = 1` (sempre)
- `Σ zᵢ ≤ k` (sempre)
- `εzᵢ ≤ xᵢ ≤ δzᵢ` (sempre)
- `μ^T x ≥ R` (nem sempre, tratado por penalização)

---

## 5. Lista Tabu

### 5.1 Conceito

A **lista tabu** é uma memória de curto prazo que armazena movimentos recentemente executados e os proíbe temporariamente.

**Objetivo:** Evitar ciclos e forçar exploração de novas regiões.

### 5.2 Estrutura

```python
tabu_moves = {
    ('increase', 3, -1): 15,  # movimento, tenure
    ('transfer', 2, 7): 8,
    ('decrease', 5, -1): 3
}
```

**Tenure:** Número de iterações que o movimento permanece proibido.

### 5.3 Funcionamento

#### Adicionar Movimento
```python
def add_move(move, tenure=None):
    if tenure is None:
        tenure = random(10, 25)  # tenure aleatório
    tabu_moves[move] = tenure
```

#### Verificar Status Tabu
```python
def is_tabu(move):
    return move in tabu_moves and tabu_moves[move] > 0
```

#### Atualizar Lista
```python
def update():
    for move in tabu_moves:
        tabu_moves[move] -= 1
    # Remove movimentos com tenure = 0
```

### 5.4 Exemplo de Evolução

```
Iteração 1:
  Executa: ('increase', 3, -1)
  Adiciona com tenure=15
  Lista: {('increase', 3, -1): 15}

Iteração 2:
  Executa: ('transfer', 2, 7)
  Adiciona com tenure=12
  Atualiza tenures
  Lista: {('increase', 3, -1): 14, ('transfer', 2, 7): 12}

...

Iteração 16:
  Atualiza tenures
  Lista: {('increase', 3, -1): 0, ...}
  Remove ('increase', 3, -1)
  Movimento volta a ser permitido
```

### 5.5 Tamanho da Lista

**Tamanho variável:** [min_size, max_size] = [10, 25]

**Por que aleatório?**
- Tenure fixo pode criar padrões cíclicos
- Aleatoriedade aumenta diversificação
- Range [10, 25] determinado experimentalmente

**Impacto:**
- Tenure pequeno: pouca memória, pode ciclar
- Tenure grande: muita restrição, busca lenta

---

## 6. Shifting Penalty Mechanism

### 6.1 Motivação

**Problema:** Como tratar a restrição de retorno mínimo?

**Abordagens tradicionais:**
1. **Hard constraint:** Elimina soluções inviáveis
   - Problema: Busca pode ficar presa na fronteira viável
   
2. **Penalização fixa:** f(x) = variância + w × violação
   - Problema: Peso w difícil de calibrar

**Solução:** Penalização **adaptativa** (shifting penalty)

### 6.2 Funcionamento

O peso da penalização `w₁` é ajustado dinamicamente:

```python
f(x) = x^T Σ x + w₁(t) × max(0, R - μ^T x)
```

onde `w₁(t)` varia ao longo do tempo.

### 6.3 Regras de Atualização

**Contadores:**
- `consecutive_feasible`: iterações viáveis consecutivas
- `consecutive_infeasible`: iterações inviáveis consecutivas

**Regra 1: Diminuir peso**
```
SE consecutive_feasible ≥ K:
    γ ← random(1.5, 2.0)
    w₁ ← w₁ / γ
    consecutive_feasible ← 0
```

**Regra 2: Aumentar peso**
```
SE consecutive_infeasible ≥ H:
    γ ← random(1.5, 2.0)
    w₁ ← w₁ × γ
    consecutive_infeasible ← 0
```

### 6.4 Exemplo Numérico Detalhado

```
Configuração:
  w₁_inicial = 1000
  K = 20 (diminui após 20 viáveis)
  H = 1 (aumenta após 1 inviável)

Iteração 0:
  w₁ = 1000
  Solução: retorno = 0.0048, R = 0.005
  Inviável! (0.0048 < 0.005)
  consecutive_infeasible = 1

Iteração 1:
  consecutive_infeasible = 1 ≥ H
  γ = 1.8
  w₁ = 1000 × 1.8 = 1800
  Penalização aumenta!

Iteração 2-5:
  Soluções ainda inviáveis
  w₁ = 1800 → 3240 → 5832 → 10498
  Busca fortemente empurrada para região viável

Iteração 6:
  Primeira solução viável!
  consecutive_feasible = 1

Iteração 7-25:
  Todas viáveis
  consecutive_feasible = 20

Iteração 26:
  consecutive_feasible = 20 ≥ K
  γ = 1.6
  w₁ = 10498 / 1.6 = 6561
  Peso diminui, busca explora mais livremente

Iteração 27-46:
  Mais 20 viáveis
  w₁ = 6561 / 1.9 = 3453

...continua alternando...
```

### 6.5 Classe ShiftingPenalty

```python
class ShiftingPenalty:
    def __init__(self, initial_weight=1000.0):
        self.weight = initial_weight
        self.consecutive_feasible = 0
        self.consecutive_infeasible = 0
        self.weight_history = []
    
    def update(self, is_feasible, K=20, H=1):
        if is_feasible:
            self.consecutive_feasible += 1
            self.consecutive_infeasible = 0
            if self.consecutive_feasible >= K:
                gamma = random(1.5, 2.0)
                self.weight /= gamma
                self.consecutive_feasible = 0
                return True  # peso mudou
        else:
            self.consecutive_infeasible += 1
            self.consecutive_feasible = 0
            if self.consecutive_infeasible >= H:
                gamma = random(1.5, 2.0)
                self.weight *= gamma
                self.consecutive_infeasible = 0
                return True  # peso mudou
        return False
```

### 6.6 Vantagens

1. **Exploração guiada:**
   - Permite incursões temporárias na região inviável
   - Explora soluções próximas à fronteira

2. **Auto-ajuste:**
   - Não precisa calibrar manualmente
   - Adapta-se à dificuldade do problema

3. **Equilíbrio dinâmico:**
   - Peso alto → foca viabilidade
   - Peso baixo → foca otimização

### 6.7 Impacto no Custo

Quando o peso muda, **recalculamos** os custos:

```python
if weight_changed:
    current_solution.cost = calculate_cost(current_solution)
    best_solution.cost = calculate_cost(best_solution)
```

**Por que?** Para manter comparações consistentes.

**Exemplo:**
```
Antes: w₁ = 1000
  Solução A: variância = 0.0002, violação = 0.0001
  custo_A = 0.0002 + 1000×0.0001 = 0.1002

Depois: w₁ = 500
  custo_A = 0.0002 + 500×0.0001 = 0.0502
  
O custo mudou! Precisamos recalcular.
```

---

## 7. Operadores de Vizinhança

### 7.1 Conceito de Vizinhança

A **vizinhança** N(s) de uma solução s é o conjunto de soluções que podem ser alcançadas aplicando um **movimento** em s.

**Movimento:** Pequena modificação na solução atual.

### 7.2 Operador idID

**Nome:** increase, decrease, Insert, Delete

**Movimentos:**

#### 1. INCREASE
```
Ação: Aumenta xᵢ em step%
Operação: xᵢ ← xᵢ × (1 + step)
Rebalanceamento: Diminui outros ativos proporcionalmente
```

**Exemplo:**
```python
Antes:  x = [0.20, 0.30, 0.50]
        z = [1, 1, 1]

INCREASE ativo 0 com step=0.3:
  x₀ = 0.20 × (1+0.3) = 0.26  (+0.06)
  Precisa tirar 0.06 dos outros
  
  Espaço ajustável:
    x₁-ε = 0.30-0.01 = 0.29
    x₂-ε = 0.50-0.01 = 0.49
    Total = 0.78
  
  Peso de x₁: 0.29/0.78 = 0.372
  Peso de x₂: 0.49/0.78 = 0.628
  
  x₁ -= 0.06 × 0.372 = 0.0223 → x₁ = 0.2777
  x₂ -= 0.06 × 0.628 = 0.0377 → x₂ = 0.4623

Depois: x = [0.26, 0.2777, 0.4623]
        Soma = 1.0000 ✓
```

#### 2. DECREASE
```
Ação: Diminui xᵢ em step%
Operação: xᵢ ← xᵢ × (1 - step)
Caso especial: Se xᵢ < ε → DELETE
```

#### 3. INSERT
```
Ação: Adiciona novo ativo
Operação:
  zᵢ ← 1
  xᵢ ← ε
  Tira ε dos outros ativos
```

#### 4. DELETE (implícito)
```
Ação: Remove ativo
Operação:
  zᵢ ← 0
  xᵢ ← 0
  Redistribui xᵢ entre outros
```

### 7.3 Operador TID

**Nome:** Transfer, Insert, Delete

**Movimento:** TRANSFER
```
Ação: Transfere step% de xᵢ para xⱼ
Operação:
  amount = xᵢ × step
  xᵢ -= amount
  xⱼ += amount
```

**Casos especiais:**

1. **Origem cai abaixo de ε:**
```python
if x_from - amount < ε:
    amount = x_from  # transfere tudo
    z_from = 0       # remove ativo
```

2. **Destino não está no portfólio:**
```python
if z_to == 0:
    if Σzᵢ >= k:
        return None  # não pode adicionar
    z_to = 1
    if amount < ε:
        amount = ε   # garante mínimo
```

### 7.4 Comparação dos Operadores

| Aspecto | idID | TID |
|---------|------|-----|
| **Granularidade** | Fina | Grossa |
| **Foco** | Um ativo | Par de ativos |
| **Complexidade** | Média | Alta |
| **Tamanho vizinhança** | ~2k+n | ~k×n |
| **Eficácia** | Intensificação | Diversificação |

### 7.5 Geração Completa da Vizinhança

```python
def generate_all_neighbors(solution):
    neighbors = []
    
    # Operador idID
    for i in ativos_presentes:
        neighbors += increase(i)
        neighbors += decrease(i)
    
    for i in ativos_ausentes:
        if len(portfólio) < k:
            neighbors += insert(i)
    
    # Operador TID
    for i in ativos_presentes:
        for j in todos_ativos:
            if i != j:
                neighbors += transfer(i, j)
    
    return neighbors
```

**Tamanho típico:** Para n=50, k=10:
- idID: ~50 vizinhos
- TID: ~490 vizinhos
- **Total: ~540 vizinhos**

---

## 8. Função Objetivo e Avaliação

### 8.1 Função de Custo

```python
def calculate_cost(solution):
    x = solution.x
    
    # Termo 1: Variância (objetivo principal)
    variance = x^T @ Σ @ x
    
    # Termo 2: Penalização adaptativa
    current_return = μ^T @ x
    violation = max(0, R - current_return)
    penalty = w₁ × violation
    
    return variance + penalty
```

### 8.2 Decomposição do Custo

**Exemplo numérico:**
```
Solução: x = [0.3, 0.0, 0.4, 0.3, 0.0]
         z = [1, 0, 1, 1, 0]

Dados:
  μ = [0.05, 0.08, 0.06, 0.04, 0.07]
  R = 0.06
  w₁ = 1000

Cálculo:

1. Retorno atual:
   r = 0.3×0.05 + 0.4×0.06 + 0.3×0.04
     = 0.015 + 0.024 + 0.012
     = 0.051

2. Violação:
   v = max(0, 0.06 - 0.051) = 0.009

3. Penalização:
   p = 1000 × 0.009 = 9.0

4. Variância (supondo Σ):
   σ² = x^T Σ x = 0.000234

5. Custo total:
   f = 0.000234 + 9.0 = 9.000234

Análise:
- Dominado pela penalização (inviável)
- Busca será empurrada para aumentar retorno
```

### 8.3 Verificação de Viabilidade

```python
def is_solution_feasible(solution):
    return μ^T @ solution.x >= R
```

**Uso:** Determina comportamento do shifting penalty.

---

## 9. Critério de Aspiração

### 9.1 Conceito

O **critério de aspiração** permite executar um movimento **mesmo se estiver tabu**.

**Condição:** O movimento deve levar a uma solução **melhor que a melhor global**.

### 9.2 Implementação

```python
def aspiration_criterion(move_cost, best_cost):
    return move_cost < best_cost
```

### 9.3 Exemplo

```
Estado atual:
  s_current: custo = 120
  s_best: custo = 100
  
Vizinho considerado:
  Movimento m: ('transfer', 3, 7)
  s' resultante: custo = 95
  Status: m está TABU
  
Verificação:
  is_tabu(m) = True
  aspiration(95, 100) = 95 < 100 = True
  
Decisão: ACEITAR m (aspiração satisfeita)
  
Resultado:
  s_current ← s'
  s_best ← s' (novo melhor!)
```

### 9.4 Por que Aspiração?

**Sem aspiração:**
- Movimento excepcional pode ser perdido
- Lista tabu pode ser muito restritiva

**Com aspiração:**
- Garante que soluções melhores nunca são ignoradas
- "Rota de escape" quando lista tabu bloqueia progresso

---

## 10. Loop Principal do Algoritmo

### 10.1 Pseudocódigo Detalhado

```
Algoritmo: Tabu Search para Portfólios

Entrada:
  n, k, ε, δ, μ, Σ, R
  max_iterations, max_idle_iterations
  K, H (parâmetros shifting penalty)

Saída:
  s_best (melhor solução encontrada)

Inicialização:
1. s₀ ← generate_initial_solution()
2. s_current ← s₀
3. s_best ← s₀
4. tabu_list ← ∅
5. penalty_weight ← 1000
6. idle_counter ← 0

Loop Principal:
7. PARA iteration = 0 ATÉ max_iterations:
   
   a) Geração de Vizinhança:
      neighbors ← generate_all_neighbors(s_current)
      SE neighbors = ∅:
         BREAK  # sem vizinhos válidos
   
   b) Seleção:
      best_move ← None
      best_neighbor ← None
      best_cost ← ∞
      
      PARA CADA (move, s') EM neighbors:
         is_tabu ← tabu_list.contains(move)
         aspiration ← (cost(s') < cost(s_best))
         
         SE NÃO is_tabu OU aspiration:
            SE cost(s') < best_cost:
               best_cost ← cost(s')
               best_neighbor ← s'
               best_move ← move
      
      SE best_neighbor = None:
         BREAK  # nenhum movimento válido
   
   c) Movimento:
      s_current ← best_neighbor
   
   d) Atualização Tabu:
      tabu_list.add(best_move)
   
   e) Shifting Penalty:
      is_feasible ← is_solution_feasible(s_current)
      weight_changed ← penalty.update(is_feasible, K, H)
      
      SE weight_changed:
         cost(s_current) ← recalculate_cost(s_current)
         cost(s_best) ← recalculate_cost(s_best)
   
   f) Atualização Melhor:
      SE cost(s_current) < cost(s_best):
         s_best ← s_current
         idle_counter ← 0
      SENÃO:
         idle_counter ← idle_counter + 1
   
   g) Manutenção Tabu:
      tabu_list.update()  # decrementa tenures
   
   h) Critério de Parada:
      SE idle_counter ≥ max_idle_iterations:
         BREAK  # estagnação

8. RETORNAR s_best
```

### 10.2 Fluxograma Detalhado

```
        ┌──────────────┐
        │ Inicialização│
        │ s₀, s_best   │
        └──────┬───────┘
               │
        ┌──────▼───────┐
    ┌───┤ iteration < │◄──────────────┐
    │   │    max?     │               │
    │   └──────┬───────┘               │
    │          │ Sim                   │
    │   ┌──────▼───────┐               │
    │   │ Gera Vizinhos│               │
    │   └──────┬───────┘               │
    │          │                       │
    │   ┌──────▼────────┐              │
    │   │ Vazio?        │──Sim─►BREAK─┤
    │   └──────┬────────┘              │
    │          │ Não                   │
    │   ┌──────▼────────┐              │
    │   │ Para cada     │              │
    │   │ vizinho:      │              │
    │   │ - Verifica    │              │
    │   │   tabu        │              │
    │   │ - Verifica    │              │
    │   │   aspiração   │              │
    │   │ - Seleciona   │              │
    │   │   melhor      │              │
    │   └──────┬────────┘              │
    │          │                       │
    │   ┌──────▼────────┐              │
    │   │ Encontrou?    │──Não─►BREAK─┤
    │   └──────┬────────┘              │
    │          │ Sim                   │
    │   ┌──────▼────────┐              │
    │   │ Move para     │              │
    │   │ s' (aceita    │              │
    │   │ sempre!)      │              │
    │   └──────┬────────┘              │
    │          │                       │
    │   ┌──────▼────────┐              │
    │   │ Adiciona à    │              │
    │   │ lista tabu    │              │
    │   └──────┬────────┘              │
    │          │                       │
    │   ┌──────▼────────┐              │
    │   │ Verifica      │              │
    │   │ viabilidade   │              │
    │   └──────┬────────┘              │
    │          │                       │
    │   ┌──────▼────────┐              │
    │   │ Atualiza peso │              │
    │   │ penalização   │              │
    │   └──────┬────────┘              │
    │          │                       │
    │   ┌──────▼────────┐              │
    │   │ Peso mudou?   │──Sim─┐      │
    │   └──────┬────────┘      │      │
    │          │ Não           │      │
    │          │         ┌─────▼──────▼┐
    │          │         │ Recalcula  │
    │          │         │ custos     │
    │          │         └─────┬──────┘
    │          │               │      │
    │   ┌──────▼───────────────▼──┐   │
    │   │ s' melhor que s_best?  │   │
    │   └──────┬────────┬────────┘   │
    │          │Sim     │Não         │
    │   ┌──────▼──┐  ┌──▼──────┐     │
    │   │s_best←s'│  │idle_ctr++│    │
    │   │idle←0   │  └──┬──────┘     │
    │   └──────┬──┘     │            │
    │          └────────┘            │
    │                  │             │
    │   ┌──────────────▼───────┐     │
    │   │ Atualiza lista tabu  │     │
    │   │ (decrementa tenures) │     │
    │   └──────────────┬───────┘     │
    │                  │             │
    │   ┌──────────────▼───────┐     │
    │   │ idle_ctr ≥ max_idle? │─Sim─┤
    │   └──────────────┬───────┘     │
    │                  │Não          │
    │                  └─────────────┘
    │
    └──►┌────────────┐
        │ Retorna    │
        │ s_best     │
        └────────────┘
```

### 10.3 Características Importantes

#### 1. Movimento Sempre Aceito
```python
# A busca SEMPRE se move, mesmo que piore
s_current = best_neighbor
```

**Por que?**
- Permite escapar de ótimos locais
- A lista tabu garante que não voltamos imediatamente

#### 2. Duas Soluções Mantidas
```python
s_current  # Solução atual da busca (pode piorar)
s_best     # Melhor solução já encontrada (só melhora)
```

#### 3. Critérios de Parada

**Parada 1: Limite de iterações**
```python
if iteration >= max_iterations:
    break
```

**Parada 2: Estagnação**
```python
if idle_counter >= max_idle_iterations:
    break  # Muitas iterações sem melhoria
```

**Parada 3: Vizinhança vazia**
```python
if len(neighbors) == 0:
    break  # Raro, mas possível
```

---

## 11. Estratégias do Artigo de Schaerf

### 11.1 Contribuições Principais

O artigo de Schaerf (2001) propõe as seguintes estratégias específicas para portfólios:

#### 1. Representação Híbrida
- Combina variáveis binárias (z) e contínuas (x)
- Permite movimentos eficientes

#### 2. Múltiplos Operadores de Vizinhança
- idID para intensificação (passos menores)
- TID para diversificação (transferências)
- Uso combinado melhora exploração

#### 3. Shifting Penalty Adaptativo
- Ajuste dinâmico baseado em viabilidade
- Parâmetros: K=20, H=1
- Fator γ aleatório em [1.5, 2.0]

#### 4. Lista Tabu de Tamanho Variável
- Tenure aleatório entre [10, 25]
- Maior diversidade que tamanho fixo

#### 5. Passo Aleatório
- step_size base: 0.3 (30%)
- step_variation: 0.3
- Cada movimento usa step ∈ [0.0, 0.6]

### 11.2 Token-Ring Strategy (Opcional)

Schaerf também propõe alternar entre runners:

```
Runner 1: TS com idID, step grande (0.4)
         ↓
     melhora?
         ↓
Runner 2: TS com TID, step pequeno (0.05)
         ↓
     melhora?
         ↓
Volta para Runner 1
```

**Vantagem:** Combina exploração (R1) com refinamento (R2).

### 11.3 Comparação com Trabalhos Anteriores

**Rolland (1997):**
- Usou apenas TID
- Problema sem restrições de cardinalidade
- Step fixo: 0.01

**Chang et al. (2000):**
- Usou variante de idR
- Genetic Algorithms superaram TS
- Nossa implementação: TS competitivo

**Nossa implementação (baseada em Schaerf):**
- Combina idID + TID
- Shifting penalty
- Step aleatório
- Lista tabu variável

---

## 12. Parâmetros e Configurações

### 12.1 Tabela de Parâmetros

| Parâmetro | Símbolo | Padrão | Range Típico | Descrição |
|-----------|---------|--------|--------------|-----------|
| **Problema** |
| Ativos disponíveis | n | 50-200 | 10-500 | Tamanho do universo |
| Cardinalidade máxima | k | 10 | 5-30 | Ativos no portfólio |
| Proporção mínima | ε | 0.01 | 0.01-0.10 | Mínimo por ativo |
| Proporção máxima | δ | 1.0 | 0.5-1.0 | Máximo por ativo |
| Retorno mínimo | R | 0.005 | variável | Depende de μ |
| **Algoritmo** |
| Iterações máximas | - | 1000 | 500-5000 | Limite superior |
| Idle máximo | - | 100 | 50-200 | Parada por estagnação |
| **Lista Tabu** |
| Tenure mínimo | min | 10 | 5-20 | Limite inferior |
| Tenure máximo | max | 25 | 15-50 | Limite superior |
| **Vizinhança** |
| Tamanho do passo | q | 0.3 | 0.1-0.5 | Base do step |
| Variação do passo | d | 0.3 | 0.1-0.5 | Aleatoriedade |
| **Shifting Penalty** |
| Peso inicial | w₁ | 1000 | 100-5000 | Penalização inicial |
| Iterações viáveis | K | 20 | 10-50 | Para diminuir peso |
| Iterações inviáveis | H | 1 | 1-5 | Para aumentar peso |
| Gamma mínimo | γ_min | 1.5 | 1.2-2.0 | Fator multiplicativo |
| Gamma máximo | γ_max | 2.0 | 1.5-3.0 | Fator multiplicativo |

### 12.2 Sensibilidade dos Parâmetros

**Mais críticos:**
1. **step_size (q):** Afeta exploração vs intensificação
2. **K:** Afeta frequência de mudança do peso
3. **max_idle_iterations:** Determina quando parar

**Menos críticos:**
4. tenure (min, max): Range [10, 25] funciona bem
5. H: Valor 1 é adequado para maioria dos casos
6. γ: Aleatoriedade já ajuda, range exato menos importante

### 12.3 Configurações Recomendadas

**Para problemas pequenos (n < 50):**
```python
max_iterations = 500
max_idle_iterations = 50
step_size = 0.2
K = 10
```

**Para problemas médios (50 ≤ n ≤ 200):**
```python
max_iterations = 1000
max_idle_iterations = 100
step_size = 0.3
K = 20
```

**Para problemas grandes (n > 200):**
```python
max_iterations = 2000
max_idle_iterations = 200
step_size = 0.4
K = 30
```

---

## 13. Análise de Resultados

### 13.1 Histórico Armazenado

```python
history = {
    'iteration': [0, 1, 2, ...],
    'current_cost': [0.245, 0.238, 0.242, ...],
    'best_cost': [0.245, 0.238, 0.236, ...],
    'n_assets': [10, 10, 9, ...],
    'penalty_weight': [1000, 1800, 3240, ...],
    'is_feasible': [False, False, True, ...]
}
```

### 13.2 Métricas de Avaliação

**Métricas de qualidade:**
```python
info = get_portfolio_info(s_best)

Retorno esperado:  E[R] = μ^T x
Variância:         σ² = x^T Σ x
Desvio padrão:     σ = √σ²
Sharpe Ratio:      SR = E[R] / σ
```

**Métricas do algoritmo:**
- Iterações até convergência
- Número de melhorias
- Taxa de soluções viáveis
- Evolução do peso de penalização

### 13.3 Visualizações Úteis

**1. Convergência:**
```python
plt.plot(history['iteration'], history['best_cost'])
plt.xlabel('Iteração')
plt.ylabel('Melhor Custo')
```

**2. Viabilidade:**
```python
colors = ['g' if f else 'r' for f in history['is_feasible']]
plt.scatter(history['iteration'], history['current_cost'], c=colors)
```

**3. Peso de Penalização:**
```python
plt.plot(history['iteration'], history['penalty_weight'])
plt.yscale('log')
```

**4. Composição do Portfólio:**
```python
assets = info['selected_assets']
weights = [info['weights'][i] for i in assets]
plt.bar(assets, weights)
```

### 13.4 Indicadores de Sucesso

**Busca bem-sucedida:**
✓ Peso varia ao longo do tempo (não fica constante)
✓ Alterna entre viável/inviável (explora fronteira)
✓ Convergência suave (não estagna cedo)
✓ Solução final é viável
✓ Melhoria consistente no início

**Problemas potenciais:**
✗ Peso sempre aumenta → problema muito restritivo
✗ Sempre inviável → peso inicial muito baixo
✗ Estagna em ~50 iterações → aumentar step_size
✗ Não melhora após inicial → aumentar max_iterations

### 13.5 Comparação com Benchmark

**Fronteira Eficiente (UEF):**
```
Solução do modelo de Markowitz sem restrições:
  Risco UEF: 0.000215
  
Nossa solução (com restrições):
  Risco ACEF: 0.000228
  
Desvio percentual:
  δ = (0.000228 - 0.000215) / 0.000215 × 100%
    = 6.05%
```

**Interpretação:**
- δ < 5%: Excelente
- 5% ≤ δ < 10%: Bom
- 10% ≤ δ < 20%: Aceitável
- δ ≥ 20%: Precisa melhorar

---

## Conclusão

O algoritmo Tabu Search implementado combina:

1. **Representação eficiente:** Vetores z e x
2. **Memória adaptativa:** Lista tabu variável
3. **Exploração guiada:** Shifting penalty mechanism
4. **Operadores complementares:** idID + TID
5. **Critérios inteligentes:** Aspiração e parada

Essas estratégias, baseadas no artigo de Schaerf (2001), permitem encontrar soluções de alta qualidade para o problema de otimização de portfólios com restrições de cardinalidade e quantidade, mesmo em instâncias de grande porte.

O algoritmo equilibra exploração e intensificação, escapando de ótimos locais enquanto mantém foco em regiões promissoras do espaço de busca.
