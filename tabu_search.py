"""
tabu_search.py

Implementação do algoritmo Tabu Search para otimização de portfólios
baseado no artigo de Schaerf (2001): "Local Search Techniques for 
Constrained Portfolio Selection Problems"

Autor: Projeto Otimização de Portfólios
Data: 2025
"""

import numpy as np
from dataclasses import dataclass
from copy import deepcopy
from typing import List, Tuple, Optional, Dict


@dataclass
class Solution:
    """
    Representa uma solução do problema de portfólio
    
    Attributes:
        z: Vetor binário [z_1, ..., z_n] onde z_i = 1 se ativo i está no portfólio
        x: Vetor de proporções [x_1, ..., x_n] onde x_i é a fração investida no ativo i
        cost: Valor da função objetivo (risco = variância + penalidades)
    """
    z: np.ndarray
    x: np.ndarray
    cost: float


class TabuList:
    """
    Implementa a lista tabu com tamanho variável (memória de curto prazo)
    
    A lista tabu armazena movimentos recentemente executados para evitar
    ciclos e garantir diversificação da busca.
    """
    
    def __init__(self, min_size=10, max_size=25):
        """
        Inicializa a lista tabu
        
        Args:
            min_size: Tamanho mínimo da lista (tenure mínimo)
            max_size: Tamanho máximo da lista (tenure máximo)
        """
        self.min_size = min_size
        self.max_size = max_size
        self.tabu_moves = {}  # {movimento: iterações_restantes}
        
    def add_move(self, move: Tuple, tenure: Optional[int] = None):
        """
        Adiciona um movimento à lista tabu
        
        Args:
            move: Tupla (tipo_movimento, ativo_i, ativo_j) representando o movimento
            tenure: Número de iterações que o movimento permanece tabu.
                   Se None, é escolhido aleatoriamente entre min_size e max_size.
        """
        if tenure is None:
            tenure = np.random.randint(self.min_size, self.max_size + 1)
        self.tabu_moves[move] = tenure
        
    def is_tabu(self, move: Tuple) -> bool:
        """
        Verifica se um movimento está na lista tabu
        
        Args:
            move: Tupla representando o movimento
            
        Returns:
            True se o movimento está tabu (proibido), False caso contrário
        """
        return move in self.tabu_moves and self.tabu_moves[move] > 0
    
    def update(self):
        """
        Atualiza a lista tabu decrementando os contadores (tenures)
        Remove movimentos cujo tenure chegou a zero
        """
        moves_to_remove = []
        for move in self.tabu_moves:
            self.tabu_moves[move] -= 1
            if self.tabu_moves[move] <= 0:
                moves_to_remove.append(move)
                
        for move in moves_to_remove:
            del self.tabu_moves[move]
    
    def clear(self):
        """Limpa completamente a lista tabu"""
        self.tabu_moves.clear()


class ShiftingPenalty:
    """
    Gerencia o ajuste dinâmico dos pesos de penalização (Shifting Penalty Mechanism)
    
    Este mecanismo adapta automaticamente o peso da penalização de restrições
    violadas durante a busca, permitindo exploração de soluções inviáveis
    promissoras enquanto guia a busca de volta à região viável.
    """
    
    def __init__(self, initial_weight: float = 1000.0, 
                 gamma_min: float = 1.5, gamma_max: float = 2.0):
        """
        Inicializa o gerenciador de penalização adaptativa
        
        Args:
            initial_weight: Peso inicial da penalização (w₁)
            gamma_min: Fator mínimo de multiplicação/divisão
            gamma_max: Fator máximo de multiplicação/divisão
        """
        self.weight = initial_weight
        self.initial_weight = initial_weight
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        
        # Contadores de iterações consecutivas
        self.consecutive_feasible = 0
        self.consecutive_infeasible = 0
        
        # Histórico da evolução do peso
        self.weight_history = [initial_weight]
    
    def get_random_gamma(self) -> float:
        """
        Retorna um fator de ajuste aleatório entre gamma_min e gamma_max
        
        Returns:
            Valor aleatório γ usado para multiplicar ou dividir o peso
        """
        return np.random.uniform(self.gamma_min, self.gamma_max)
    
    def update(self, is_feasible: bool, K: int = 20, H: int = 1) -> bool:
        """
        Atualiza o peso de penalização baseado na viabilidade da solução atual
        
        Estratégia:
        - Se viável por K iterações consecutivas → DIMINUI peso (w₁ / γ)
        - Se inviável por H iterações consecutivas → AUMENTA peso (w₁ × γ)
        
        Args:
            is_feasible: True se a solução atual satisfaz todas as restrições
            K: Número de iterações viáveis consecutivas para diminuir peso
            H: Número de iterações inviáveis consecutivas para aumentar peso
            
        Returns:
            True se o peso foi modificado nesta iteração
        """
        weight_changed = False
        
        if is_feasible:
            self.consecutive_feasible += 1
            self.consecutive_infeasible = 0
            
            # Se viável por K iterações consecutivas, diminui penalização
            if self.consecutive_feasible >= K:
                gamma = self.get_random_gamma()
                self.weight = self.weight / gamma
                self.consecutive_feasible = 0
                weight_changed = True
        else:
            self.consecutive_infeasible += 1
            self.consecutive_feasible = 0
            
            # Se inviável por H iterações consecutivas, aumenta penalização
            if self.consecutive_infeasible >= H:
                gamma = self.get_random_gamma()
                self.weight = self.weight * gamma
                self.consecutive_infeasible = 0
                weight_changed = True
        
        # Armazena no histórico
        self.weight_history.append(self.weight)
        
        return weight_changed
    
    def reset(self):
        """Reseta o peso para o valor inicial e limpa contadores"""
        self.weight = self.initial_weight
        self.consecutive_feasible = 0
        self.consecutive_infeasible = 0
        self.weight_history = [self.initial_weight]


class NeighborhoodGenerator:
    """
    Gera soluções vizinhas usando diferentes operadores de movimento
    
    Implementa os operadores de vizinhança descritos no artigo de Schaerf:
    - idID (increase, decrease, Insert, Delete)
    - TID (Transfer, Insert, Delete)
    """
    
    def __init__(self, step_size: float = 0.3, step_variation: float = 0.3):
        """
        Inicializa o gerador de vizinhança
        
        Args:
            step_size: Tamanho base do passo q (padrão: 30%)
            step_variation: Variação aleatória do passo d (padrão: 30%)
        """
        self.step_size = step_size
        self.step_variation = step_variation
        
    def get_random_step(self) -> float:
        """
        Retorna um tamanho de passo aleatório no intervalo [q-d, q+d]
        
        A aleatoriedade no tamanho do passo ajuda a:
        - Evitar padrões repetitivos
        - Explorar diferentes granularidades de mudança
        - Aumentar a diversificação
        
        Returns:
            Valor aleatório entre (step_size - step_variation) e (step_size + step_variation)
        """
        lower = max(0.01, self.step_size - self.step_variation)
        upper = min(0.99, self.step_size + self.step_variation)
        return np.random.uniform(lower, upper)


class TabuSearch:
    """
    Implementação do algoritmo Tabu Search para otimização de portfólio
    com restrições de cardinalidade e quantidade
    
    Baseado no artigo:
    Schaerf, A. (2001). Local Search Techniques for Constrained Portfolio 
    Selection Problems. Computational Economics, 20(3), 177-190.
    """
    
    def __init__(self, n_assets: int, k_max: int, epsilon: float, delta: float,
                 expected_returns: np.ndarray, cov_matrix: np.ndarray, 
                 min_return: float):
        """
        Inicializa o algoritmo Tabu Search
        
        Args:
            n_assets: Número total de ativos disponíveis (n)
            k_max: Cardinalidade máxima do portfólio (restrição 5)
            epsilon: Proporção mínima de investimento em cada ativo (ε)
            delta: Proporção máxima de investimento em cada ativo (δ)
            expected_returns: Vetor μ de retornos esperados [μ₁, ..., μₙ]
            cov_matrix: Matriz Σ de covariâncias (n×n)
            min_return: Retorno mínimo desejado R (restrição 2)
        """
        # Parâmetros do problema
        self.n_assets = n_assets
        self.k_max = k_max
        self.epsilon = epsilon
        self.delta = delta
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.min_return = min_return
        
        # Componentes do algoritmo
        self.tabu_list = TabuList(min_size=10, max_size=25)
        self.neighbor_generator = NeighborhoodGenerator(step_size=0.3, 
                                                         step_variation=0.3)
        self.penalty_manager = ShiftingPenalty(initial_weight=1000.0)
        
        # Soluções
        self.best_solution = None
        self.current_solution = None
        
        # Parâmetros do algoritmo
        self.max_iterations = 1000
        self.max_idle_iterations = 100
        self.K = 20  # Iterações viáveis para diminuir peso
        self.H = 1   # Iterações inviáveis para aumentar peso
        
        # Histórico de execução
        self.history = {
            'iteration': [],
            'current_cost': [],
            'best_cost': [],
            'n_assets': [],
            'penalty_weight': [],
            'is_feasible': []
        }
    
    # ==================== FUNÇÃO OBJETIVO ====================
    
    def calculate_cost(self, solution: Solution) -> float:
        """
        Calcula o custo (variância + penalização) da solução
        
        Função objetivo:
        f(x) = x^T Σ x + w₁ × max(0, R - r^T x)
        
        onde:
        - x^T Σ x = variância do portfólio (risco)
        - R = retorno mínimo desejado
        - r^T x = retorno atual do portfólio
        - w₁ = peso adaptativo da penalização
        
        Args:
            solution: Objeto Solution a avaliar
            
        Returns:
            Custo total (variância + penalização adaptativa)
        """
        x = solution.x
        
        # Calcula variância (termo objetivo principal)
        variance = x.T @ self.cov_matrix @ x
        
        # Calcula retorno atual
        current_return = np.dot(self.expected_returns, x)
        
        # Penalização adaptativa por violação de restrição de retorno mínimo
        penalty = 0
        if current_return < self.min_return:
            violation = self.min_return - current_return
            penalty = self.penalty_manager.weight * violation
        
        return variance + penalty
    
    def is_solution_feasible(self, solution: Solution) -> bool:
        """
        Verifica se uma solução satisfaz todas as restrições hard
        
        Verifica:
        - Restrição 2: Σ rᵢxᵢ ≥ R (retorno mínimo)
        - Outras restrições já garantidas pela representação
        
        Args:
            solution: Objeto Solution a verificar
            
        Returns:
            True se a solução é completamente viável
        """
        current_return = np.dot(self.expected_returns, solution.x)
        return current_return >= self.min_return
    
    # ==================== SOLUÇÃO INICIAL ====================
    
    def generate_initial_solution(self) -> Solution:
        """
        Gera uma solução inicial aleatória viável
        
        Estratégia:
        1. Seleciona k_max ativos aleatoriamente
        2. Distribui investimento igualmente: xᵢ = 1/k_max
        3. Garante que todas as restrições sejam satisfeitas
        
        Returns:
            Objeto Solution inicial
        """
        # Seleciona k_max ativos aleatórios
        selected = np.random.choice(self.n_assets, self.k_max, replace=False)
        z = np.zeros(self.n_assets)
        z[selected] = 1
        
        # Distribui investimento igualmente
        x = np.zeros(self.n_assets)
        x[selected] = 1.0 / self.k_max
        
        # Cria e avalia solução
        solution = Solution(z=z, x=x, cost=0)
        solution.cost = self.calculate_cost(solution)
        
        return solution
    
    # ==================== OPERADORES DE VIZINHANÇA ====================
    
    def generate_neighbors_idID(self, solution: Solution, 
                                step: Optional[float] = None) -> List[Tuple]:
        """
        Gera vizinhança usando operador idID (increase, decrease, Insert, Delete)
        
        Movimentos possíveis:
        1. INCREASE: Aumenta proporção xᵢ de um ativo presente em step%
        2. DECREASE: Diminui proporção xᵢ de um ativo presente em step%
        3. INSERT: Adiciona novo ativo com proporção mínima ε
        4. DELETE: Remove ativo (implícito no DECREASE quando xᵢ < ε)
        
        Args:
            solution: Solução atual
            step: Tamanho do passo (None para aleatório)
            
        Returns:
            Lista de tuplas (tipo, asset_i, asset_j, solução_vizinha)
        """
        neighbors = []
        if step is None:
            step = self.neighbor_generator.get_random_step()
        
        # Para cada ativo no portfólio (zᵢ = 1)
        for i in range(self.n_assets):
            if solution.z[i] == 1:
                # Movimento INCREASE
                neighbor = self._increase_asset(solution, i, step)
                if neighbor is not None:
                    neighbors.append(('increase', i, -1, neighbor))
                
                # Movimento DECREASE
                neighbor = self._decrease_asset(solution, i, step)
                if neighbor is not None:
                    neighbors.append(('decrease', i, -1, neighbor))
        
        # Para cada ativo fora do portfólio (zᵢ = 0)
        for i in range(self.n_assets):
            if solution.z[i] == 0 and np.sum(solution.z) < self.k_max:
                # Movimento INSERT
                neighbor = self._insert_asset(solution, i)
                if neighbor is not None:
                    neighbors.append(('insert', i, -1, neighbor))
        
        return neighbors
    
    def _increase_asset(self, solution: Solution, asset_idx: int, 
                       step: float) -> Optional[Solution]:
        """
        Aumenta a proporção do ativo asset_idx em step%
        
        Operação: xᵢ ← xᵢ × (1 + step)
        
        Args:
            solution: Solução atual
            asset_idx: Índice do ativo
            step: Percentual de aumento
            
        Returns:
            Nova solução ou None se movimento inválido
        """
        new_solution = deepcopy(solution)
        
        # Aumenta xᵢ
        new_x_i = new_solution.x[asset_idx] * (1 + step)
        
        # Verifica limite máximo
        if new_x_i > self.delta:
            new_x_i = self.delta
        
        delta_x = new_x_i - new_solution.x[asset_idx]
        
        if delta_x <= 0:
            return None
        
        new_solution.x[asset_idx] = new_x_i
        
        # Rebalanceia outros ativos para manter Σxᵢ = 1
        self._rebalance_portfolio(new_solution, exclude_idx=asset_idx, 
                                  delta=-delta_x)
        
        new_solution.cost = self.calculate_cost(new_solution)
        return new_solution
    
    def _decrease_asset(self, solution: Solution, asset_idx: int, 
                       step: float) -> Optional[Solution]:
        """
        Diminui a proporção do ativo asset_idx em step%
        
        Operação: xᵢ ← xᵢ × (1 - step)
        Se xᵢ < ε após diminuição, remove o ativo (DELETE)
        
        Args:
            solution: Solução atual
            asset_idx: Índice do ativo
            step: Percentual de diminuição
            
        Returns:
            Nova solução ou None se movimento inválido
        """
        new_solution = deepcopy(solution)
        
        # Diminui xᵢ
        new_x_i = new_solution.x[asset_idx] * (1 - step)
        
        # Se cair abaixo do mínimo, deleta o ativo
        if new_x_i < self.epsilon:
            return self._delete_asset(solution, asset_idx)
        
        delta_x = new_solution.x[asset_idx] - new_x_i
        new_solution.x[asset_idx] = new_x_i
        
        # Rebalanceia outros ativos
        self._rebalance_portfolio(new_solution, exclude_idx=asset_idx, 
                                  delta=delta_x)
        
        new_solution.cost = self.calculate_cost(new_solution)
        return new_solution
    
    def _insert_asset(self, solution: Solution, 
                     asset_idx: int) -> Optional[Solution]:
        """
        Insere um novo ativo no portfólio com proporção mínima ε
        
        Operação:
        - zᵢ ← 1
        - xᵢ ← ε
        - Ajusta outros ativos para manter Σxⱼ = 1
        
        Args:
            solution: Solução atual
            asset_idx: Índice do ativo a inserir
            
        Returns:
            Nova solução
        """
        new_solution = deepcopy(solution)
        
        # Marca ativo como presente
        new_solution.z[asset_idx] = 1
        new_solution.x[asset_idx] = self.epsilon
        
        # Rebalanceia tirando ε dos outros ativos
        self._rebalance_portfolio(new_solution, exclude_idx=asset_idx, 
                                  delta=-self.epsilon)
        
        new_solution.cost = self.calculate_cost(new_solution)
        return new_solution
    
    def _delete_asset(self, solution: Solution, 
                     asset_idx: int) -> Solution:
        """
        Remove um ativo do portfólio
        
        Operação:
        - zᵢ ← 0
        - xᵢ ← 0
        - Redistribui xᵢ entre outros ativos
        
        Args:
            solution: Solução atual
            asset_idx: Índice do ativo a remover
            
        Returns:
            Nova solução
        """
        new_solution = deepcopy(solution)
        
        # Guarda quanto estava investido
        freed_amount = new_solution.x[asset_idx]
        
        # Remove o ativo
        new_solution.z[asset_idx] = 0
        new_solution.x[asset_idx] = 0
        
        # Redistribui o valor liberado
        self._rebalance_portfolio(new_solution, exclude_idx=asset_idx, 
                                  delta=freed_amount)
        
        new_solution.cost = self.calculate_cost(new_solution)
        return new_solution
    
    def _rebalance_portfolio(self, solution: Solution, exclude_idx: int, 
                            delta: float):
        """
        Rebalanceia o portfólio mantendo a soma das proporções igual a 1
        
        Estratégia:
        - Trabalha com (xᵢ - ε) para garantir que nenhum ativo caia abaixo do mínimo
        - Distribui delta proporcionalmente ao "espaço ajustável" de cada ativo
        
        Args:
            solution: Solução a ser rebalanceada (modificada in-place)
            exclude_idx: Índice do ativo que não deve ser alterado
            delta: Quantidade a distribuir (>0) ou remover (<0)
        """
        # Identifica ativos presentes (exceto o exclude_idx)
        active_assets = np.where((solution.z == 1) & 
                                (np.arange(self.n_assets) != exclude_idx))[0]
        
        if len(active_assets) == 0:
            return
        
        # Calcula espaço ajustável de cada ativo: (xᵢ - ε)
        adjustable = np.array([solution.x[i] - self.epsilon for i in active_assets])
        total_adjustable = np.sum(adjustable)
        
        if total_adjustable <= 0:
            # Se não há espaço, distribui uniformemente
            for i in active_assets:
                solution.x[i] += delta / len(active_assets)
        else:
            # Distribui proporcionalmente
            for idx, i in enumerate(active_assets):
                weight = adjustable[idx] / total_adjustable
                solution.x[i] += delta * weight
                
                # Garante limites [ε, δ]
                solution.x[i] = np.clip(solution.x[i], self.epsilon, self.delta)
    
    def generate_neighbors_TID(self, solution: Solution, 
                              step: Optional[float] = None) -> List[Tuple]:
        """
        Gera vizinhança usando operador TID (Transfer, Insert, Delete)
        
        Movimento:
        TRANSFER: Transfere step% de xᵢ para xⱼ
        - Se xᵢ < ε após transferência → DELETE ativo i
        - Se zⱼ = 0 → INSERT ativo j
        
        Args:
            solution: Solução atual
            step: Tamanho do passo (None para aleatório)
            
        Returns:
            Lista de tuplas (tipo, asset_i, asset_j, solução_vizinha)
        """
        neighbors = []
        if step is None:
            step = self.neighbor_generator.get_random_step()
        
        # Para cada par de ativos (i, j)
        for i in range(self.n_assets):
            if solution.z[i] == 0:
                continue
                
            for j in range(self.n_assets):
                if i == j:
                    continue
                
                # Transfere de i para j
                neighbor = self._transfer_between_assets(solution, i, j, step)
                if neighbor is not None:
                    neighbors.append(('transfer', i, j, neighbor))
        
        return neighbors
    
    def _transfer_between_assets(self, solution: Solution, from_idx: int, 
                                 to_idx: int, step: float) -> Optional[Solution]:
        """
        Transfere step% do investimento do ativo from_idx para to_idx
        
        Args:
            solution: Solução atual
            from_idx: Índice do ativo origem
            to_idx: Índice do ativo destino
            step: Percentual de transferência
            
        Returns:
            Nova solução ou None se movimento inválido
        """
        new_solution = deepcopy(solution)
        
        # Calcula quanto transferir
        transfer_amount = new_solution.x[from_idx] * step
        
        # Verifica se origem ficará abaixo do mínimo
        new_from = new_solution.x[from_idx] - transfer_amount
        if new_from < self.epsilon:
            # Transfere tudo e remove o ativo
            transfer_amount = new_solution.x[from_idx]
            new_solution.x[from_idx] = 0
            new_solution.z[from_idx] = 0
        else:
            new_solution.x[from_idx] = new_from
        
        # Se destino não está no portfólio, insere
        if new_solution.z[to_idx] == 0:
            if np.sum(new_solution.z) >= self.k_max:
                return None  # Não pode adicionar mais ativos
            
            new_solution.z[to_idx] = 1
            if transfer_amount < self.epsilon:
                transfer_amount = self.epsilon
        
        # Adiciona ao destino
        new_to = new_solution.x[to_idx] + transfer_amount
        
        # Verifica limite máximo
        if new_to > self.delta:
            new_to = self.delta
        
        new_solution.x[to_idx] = new_to
        
        # Normaliza para garantir Σxᵢ = 1
        total = np.sum(new_solution.x)
        if total > 0:
            new_solution.x = new_solution.x / total
        
        new_solution.cost = self.calculate_cost(new_solution)
        return new_solution
    
    def generate_all_neighbors(self, solution: Solution, 
                              use_idID: bool = True, 
                              use_TID: bool = True) -> List[Tuple]:
        """
        Gera toda a vizinhança combinando diferentes operadores
        
        Args:
            solution: Solução atual
            use_idID: Se True, usa operador idID
            use_TID: Se True, usa operador TID
            
        Returns:
            Lista completa de vizinhos: [(tipo, i, j, solução), ...]
        """
        all_neighbors = []
        
        if use_idID:
            neighbors_idID = self.generate_neighbors_idID(solution)
            all_neighbors.extend(neighbors_idID)
        
        if use_TID:
            neighbors_TID = self.generate_neighbors_TID(solution)
            all_neighbors.extend(neighbors_TID)
        
        return all_neighbors
    
    # ==================== SELEÇÃO E CRITÉRIOS ====================
    
    def aspiration_criterion(self, move_cost: float, 
                            current_best_cost: float) -> bool:
        """
        Critério de aspiração: permite movimento tabu se melhorar o melhor global
        
        Um movimento tabu pode ser aceito se:
        f(s') < f(s_best)
        
        onde s' é a solução resultante e s_best é a melhor conhecida.
        
        Args:
            move_cost: Custo da solução resultante do movimento
            current_best_cost: Custo da melhor solução encontrada
            
        Returns:
            True se o movimento satisfaz o critério de aspiração
        """
        return move_cost < current_best_cost
    
    def select_best_neighbor(self, neighbors: List[Tuple]) -> Tuple[Optional[Tuple], 
                                                                     Optional[Solution]]:
        """
        Seleciona o melhor vizinho não-tabu (ou que satisfaça aspiração)
        
        Estratégia:
        1. Para cada vizinho, verifica se movimento é tabu
        2. Se não-tabu OU satisfaz aspiração → candidato
        3. Retorna candidato com menor custo
        
        Args:
            neighbors: Lista de tuplas (tipo, i, j, solução)
            
        Returns:
            Tupla (movimento_escolhido, solução_escolhida) ou (None, None)
        """
        best_neighbor = None
        best_move = None
        best_cost = float('inf')
        
        for move_type, asset_i, asset_j, neighbor_solution in neighbors:
            # Representação do movimento para lista tabu
            move = (move_type, asset_i, asset_j)
            
            # Verifica status tabu
            is_tabu = self.tabu_list.is_tabu(move)
            
            # Aplica critério de aspiração
            aspiration = self.aspiration_criterion(
                neighbor_solution.cost, 
                self.best_solution.cost
            )
            
            # Aceita se não-tabu OU se satisfaz aspiração
            if not is_tabu or aspiration:
                if neighbor_solution.cost < best_cost:
                    best_cost = neighbor_solution.cost
                    best_neighbor = neighbor_solution
                    best_move = move
        
        return best_move, best_neighbor
    
    # ==================== HISTÓRICO ====================
    
    def update_history(self, iteration: int, is_feasible: bool):
        """
        Armazena informações da iteração atual no histórico
        
        Args:
            iteration: Número da iteração
            is_feasible: Se a solução atual é viável
        """
        self.history['iteration'].append(iteration)
        self.history['current_cost'].append(self.current_solution.cost)
        self.history['best_cost'].append(self.best_solution.cost)
        self.history['n_assets'].append(int(np.sum(self.current_solution.z)))
        self.history['penalty_weight'].append(self.penalty_manager.weight)
        self.history['is_feasible'].append(is_feasible)
    
    # ==================== LOOP PRINCIPAL ====================
    
    def run(self, max_iterations: Optional[int] = None, 
            max_idle_iterations: Optional[int] = None,
            K: Optional[int] = None, H: Optional[int] = None,
            verbose: bool = True) -> Solution:
        """
        Executa o algoritmo Tabu Search completo
        
        Pseudocódigo:
        1. Gerar solução inicial s₀
        2. s_best ← s₀, s_current ← s₀
        3. PARA cada iteração:
           a) Gerar vizinhança N(s_current)
           b) Selecionar melhor s' ∈ N não-tabu ou que satisfaz aspiração
           c) s_current ← s'
           d) Adicionar movimento à lista tabu
           e) Atualizar shifting penalty
           f) Se f(s') < f(s_best): s_best ← s'
           g) Atualizar lista tabu (decrementar tenures)
           h) Verificar critério de parada
        4. RETORNAR s_best
        
        Args:
            max_iterations: Número máximo de iterações
            max_idle_iterations: Máximo de iterações sem melhoria
            K: Iterações viáveis para diminuir penalização
            H: Iterações inviáveis para aumentar penalização
            verbose: Se True, imprime progresso
            
        Returns:
            Melhor solução encontrada
        """
        # Configura parâmetros
        if max_iterations is not None:
            self.max_iterations = max_iterations
        if max_idle_iterations is not None:
            self.max_idle_iterations = max_idle_iterations
        if K is not None:
            self.K = K
        if H is not None:
            self.H = H
        
        # Reseta componentes
        self.penalty_manager.reset()
        self.tabu_list.clear()
        self.history = {
            'iteration': [],
            'current_cost': [],
            'best_cost': [],
            'n_assets': [],
            'penalty_weight': [],
            'is_feasible': []
        }
        
        # Gera solução inicial
        self.current_solution = self.generate_initial_solution()
        self.best_solution = deepcopy(self.current_solution)
        
        if verbose:
            is_feas = self.is_solution_feasible(self.current_solution)
            print(f"Solução inicial:")
            print(f"  Custo: {self.current_solution.cost:.6f}")
            print(f"  Ativos: {int(np.sum(self.current_solution.z))}")
            print(f"  Viável: {is_feas}")
            print(f"  Peso penalização: {self.penalty_manager.weight:.2f}")
            print("-" * 60)
        
        # Contador de iterações sem melhoria
        idle_counter = 0
        
        # ========== LOOP PRINCIPAL ==========
        for iteration in range(self.max_iterations):
            
            # 1. Gera vizinhança completa
            neighbors = self.generate_all_neighbors(
                self.current_solution,
                use_idID=True,
                use_TID=True
            )
            
            if len(neighbors) == 0:
                if verbose:
                    print(f"Iteração {iteration}: Vizinhança vazia. Terminando.")
                break
            
            # 2. Seleciona melhor vizinho (respeitando tabu e aspiração)
            best_move, best_neighbor = self.select_best_neighbor(neighbors)
            
            if best_neighbor is None:
                if verbose:
                    print(f"Iteração {iteration}: Nenhum movimento válido. Terminando.")
                break
            
            # 3. Executa movimento
            self.current_solution = best_neighbor
            
            # 4. Adiciona à lista tabu
            self.tabu_list.add_move(best_move)
            
            # 5. Atualiza shifting penalty
            is_feasible = self.is_solution_feasible(self.current_solution)
            weight_changed = self.penalty_manager.update(is_feasible, self.K, self.H)
            
            # Se peso mudou, recalcula custos
            if weight_changed:
                self.current_solution.cost = self.calculate_cost(self.current_solution)
                self.best_solution.cost = self.calculate_cost(self.best_solution)
            
            # 6. Atualiza melhor solução
            if self.current_solution.cost < self.best_solution.cost:
                self.best_solution = deepcopy(self.current_solution)
                idle_counter = 0
                
                if verbose:
                    print(f"Iteração {iteration}: Nova melhor solução!")
                    print(f"  Custo: {self.best_solution.cost:.6f}")
                    print(f"  Ativos: {int(np.sum(self.best_solution.z))}")
                    print(f"  Viável: {is_feasible}")
                    print(f"  Movimento: {best_move[0]}")
                    print(f"  Peso: {self.penalty_manager.weight:.2f}")
            else:
                idle_counter += 1
            
            # 7. Atualiza lista tabu
            self.tabu_list.update()
            
            # 8. Armazena histórico
            self.update_history(iteration, is_feasible)
            
            # 9. Verifica critério de parada por estagnação
            if idle_counter >= self.max_idle_iterations:
                if verbose:
                    print(f"\nParada: {idle_counter} iterações sem melhoria")
                break
            
            # Print periódico
            if verbose and (iteration + 1) % 50 == 0:
                print(f"\nIteração {iteration + 1}/{self.max_iterations}")
                print(f"  Custo atual: {self.current_solution.cost:.6f}")
                print(f"  Melhor custo: {self.best_solution.cost:.6f}")
                print(f"  Viável: {is_feasible}")
                print(f"  Peso: {self.penalty_manager.weight:.2f}")
                print(f"  Tabu: {len(self.tabu_list.tabu_moves)} movimentos")
                print(f"  Idle: {idle_counter}")
        
        # Resumo final
        if verbose:
            final_feasible = self.is_solution_feasible(self.best_solution)
            print("\n" + "=" * 60)
            print("BUSCA FINALIZADA")
            print(f"Melhor custo: {self.best_solution.cost:.6f}")
            print(f"Ativos: {int(np.sum(self.best_solution.z))}")
            print(f"Viável: {final_feasible}")
            print(f"Iterações: {len(self.history['iteration'])}")
            print("=" * 60)
        
        return self.best_solution
    
    # ==================== INFORMAÇÕES ====================
    
    def get_portfolio_info(self, solution: Optional[Solution] = None) -> Dict:
        """
        Retorna informações detalhadas sobre uma solução
        
        Calcula:
        - Retorno esperado: E[R] = Σ μᵢxᵢ
        - Risco (variância): σ² = x^T Σ x
        - Desvio padrão: σ = √σ²
        - Índice de Sharpe: E[R] / σ (assumindo rf = 0)
        - Composição do portfólio
        
        Args:
            solution: Solução a analisar (usa best_solution se None)
            
        Returns:
            Dicionário com métricas financeiras e composição
        """
        if solution is None:
            solution = self.best_solution
        
        if solution is None:
            return None
        
        # Retorno esperado
        portfolio_return = np.dot(self.expected_returns, solution.x)
        
        # Risco
        portfolio_variance = solution.x.T @ self.cov_matrix @ solution.x
        portfolio_std = np.sqrt(portfolio_variance)
        
        # Sharpe ratio (rf = 0)
        sharpe_ratio = portfolio_return / portfolio_std if portfolio_std > 0 else 0
        
        # Ativos selecionados
        selected_assets = np.where(solution.z == 1)[0]
        
        info = {
            'return': portfolio_return,
            'variance': portfolio_variance,
            'std_dev': portfolio_std,
            'sharpe_ratio': sharpe_ratio,
            'n_assets': len(selected_assets),
            'selected_assets': selected_assets.tolist(),
            'weights': {int(i): float(solution.x[i]) for i in selected_assets},
            'is_feasible': self.is_solution_feasible(solution)
        }
        
        return info


# ==================== FUNÇÕES AUXILIARES ====================

def run_multiple_trials(tabu_search: TabuSearch, n_trials: int = 5, 
                       verbose: bool = False) -> Tuple[Solution, List[Solution]]:
    """
    Executa múltiplas rodadas do Tabu Search e retorna a melhor
    
    Devido aos componentes aleatórios (solução inicial, step, tenure),
    diferentes execuções podem encontrar soluções diferentes.
    Executar múltiplas vezes aumenta a chance de encontrar o ótimo global.
    
    Args:
        tabu_search: Instância configurada do TabuSearch
        n_trials: Número de rodadas
        verbose: Se True, imprime detalhes de cada rodada
        
    Returns:
        Tupla (melhor_solução_global, lista_todas_soluções)
    """
    all_solutions = []
    best_overall = None
    best_cost = float('inf')
    
    print(f"Executando {n_trials} rodadas do Tabu Search...")
    print("=" * 60)
    
    for trial in range(n_trials):
        print(f"\nRodada {trial + 1}/{n_trials}")
        print("-" * 60)
        
        # Executa busca
        solution = tabu_search.run(verbose=verbose)
        all_solutions.append(solution)
        
        # Atualiza melhor global
        if solution.cost < best_cost:
            best_cost = solution.cost
            best_overall = deepcopy(solution)
            print(f"  → Nova melhor global! Custo: {best_cost:.6f}")
    
    print("\n" + "=" * 60)
    print("TODAS AS RODADAS CONCLUÍDAS")
    print(f"Melhor custo: {best_cost:.6f}")
    print("=" * 60)
    
    return best_overall, all_solutions