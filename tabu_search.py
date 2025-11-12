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
from typing import List, Tuple, Optional, Dict, Union
import random


# =====================================================
# CLASSES AUXILIARES
# =====================================================

@dataclass
class Solution:
    """Representa uma solução do problema de portfólio"""
    z: np.ndarray          # vetor binário de seleção de ativos
    x: np.ndarray          # pesos do portfólio (somam 1)
    cost: float            # custo = variância + penalidades


class TabuList:
    """Lista Tabu com memória de curto prazo (tenure variável)"""
    def __init__(self, min_size: int = 10, max_size: int = 25):
        self.min_size = int(min_size)
        self.max_size = int(max_size)
        self.tabu_moves: Dict[Tuple, int] = {}

    def add_move(self, move: Tuple, tenure: Optional[int] = None) -> None:
        if tenure is None:
            tenure = np.random.randint(self.min_size, self.max_size + 1)
        self.tabu_moves[move] = int(tenure)

    def is_tabu(self, move: Tuple) -> bool:
        return move in self.tabu_moves and self.tabu_moves[move] > 0

    def update(self) -> None:
        to_remove = []
        for move in self.tabu_moves:
            self.tabu_moves[move] -= 1
            if self.tabu_moves[move] <= 0:
                to_remove.append(move)
        for move in to_remove:
            del self.tabu_moves[move]

    def clear(self) -> None:
        self.tabu_moves.clear()


class ShiftingPenalty:
    """Mecanismo de penalização adaptativa (Schaerf, 2001)"""
    def __init__(self, initial_weight: float = 1000.0, gamma_min: float = 1.5, gamma_max: float = 2.0):
        self.weight = float(initial_weight)
        self.initial_weight = float(initial_weight)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.consecutive_feasible = 0
        self.consecutive_infeasible = 0
        self.weight_history = [self.weight]

    def get_random_gamma(self) -> float:
        return float(np.random.uniform(self.gamma_min, self.gamma_max))

    def update(self, is_feasible: bool, K: int = 20, H: int = 1) -> bool:
        changed = False
        if is_feasible:
            self.consecutive_feasible += 1
            self.consecutive_infeasible = 0
            if self.consecutive_feasible >= K:
                self.weight /= self.get_random_gamma()
                self.consecutive_feasible = 0
                changed = True
        else:
            self.consecutive_infeasible += 1
            self.consecutive_feasible = 0
            if self.consecutive_infeasible >= H:
                self.weight *= self.get_random_gamma()
                self.consecutive_infeasible = 0
                changed = True
        self.weight_history.append(self.weight)
        return changed

    def reset(self) -> None:
        self.weight = self.initial_weight
        self.consecutive_feasible = 0
        self.consecutive_infeasible = 0
        self.weight_history = [self.initial_weight]


class NeighborhoodGenerator:
    """Gera vizinhança com operadores idID e TID"""
    def __init__(self, step_size: float = 0.3, step_variation: float = 0.3):
        self.step_size = float(step_size)
        self.step_variation = float(step_variation)

    def get_random_step(self) -> float:
        lower = max(0.01, self.step_size - self.step_variation)
        upper = min(0.99, self.step_size + self.step_variation)
        return float(np.random.uniform(lower, upper))


# =====================================================
# CLASSE PRINCIPAL: TABU SEARCH
# =====================================================

class TabuSearch:
    def __init__(
        self,
        n_assets: int,
        k_max: int,
        epsilon: float,
        delta: float,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        min_return: float,
        strategy: str = "idID_TID",
        max_iter: int = 1000,
        max_idle: int = 100,
        tabu_tenure: Union[int, Tuple[int, int]] = (10, 25),
        step_size: float = 0.3,
        step_delta: float = 0.3,
        neighbor_sample_ratio: float = 1.0,
        token_ring_enabled: bool = True
    ):
        """
        Inicializa o algoritmo Tabu Search.

        Args:
            n_assets: número total de ativos
            k_max: cardinalidade máxima
            epsilon: peso mínimo por ativo selecionado
            delta:  peso máximo por ativo
            expected_returns: vetor de retornos esperados (μ)
            cov_matrix: matriz de covariâncias (Σ)
            min_return: retorno mínimo desejado (restrição dura)
            strategy: string contendo os operadores a usar (ex.: "idID_TID")
            max_iter: iterações máximas do loop principal
            max_idle: iterações sem melhora para parada
            tabu_tenure: inteiro (tenure fixo) ou tupla (min, max)
            step_size: passo base para movimentos
            step_delta: variação do passo (±)
            neighbor_sample_ratio: fração de vizinhos avaliados (diversificação)
            token_ring_enabled: se verdadeiro, alterna operadores por fases
        """
        # Problema
        self.n_assets = int(n_assets)
        self.k_max = int(k_max)
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.expected_returns = expected_returns.astype(float)
        self.cov_matrix = cov_matrix.astype(float)
        self.min_return = float(min_return)

        # Parâmetros principais
        self.strategy = strategy or "idID_TID"
        self.use_idID = ("idID" in self.strategy) or ("ID_TID" in self.strategy)
        self.use_TID = ("TID" in self.strategy)
        self.max_iterations = int(max_iter)
        self.max_idle_iterations = int(max_idle)
        self.neighbor_sample_ratio = float(neighbor_sample_ratio)
        self.token_ring_enabled = bool(token_ring_enabled)

        # Token-ring e controle
        self.step_size = float(step_size)
        self.step_delta = float(step_delta)
        self.token_ring_sequence: List[Tuple[str, float]] = [('TID', 0.3), ('idID', 0.05)]
        self.token_ring_interval = 50
        self._token_phase = 0
        self.K = 20
        self.H = 1

        # Tenure (aceita int ou tupla)
        if isinstance(tabu_tenure, tuple):
            min_ten, max_ten = tabu_tenure
        else:
            min_ten = max_ten = int(tabu_tenure)

        # Componentes
        self.tabu_list = TabuList(min_ten, max_ten)
        self.neighbor_generator = NeighborhoodGenerator(step_size=self.step_size,
                                                        step_variation=self.step_delta)
        self.penalty_manager = ShiftingPenalty()

        # Soluções
        self.best_solution: Optional[Solution] = None
        self.current_solution: Optional[Solution] = None

        # Histórico
        self.history: Dict[str, List] = {
            'iteration': [],
            'current_cost': [],
            'best_cost': [],
            'n_assets': [],
            'penalty_weight': [],
            'is_feasible': []
        }

    # ==================== FUNÇÃO OBJETIVO ====================

    def calculate_cost(self, solution: Solution) -> float:
        """Custo = variância + penalidade por violar retorno mínimo."""
        x = solution.x
        variance = float(x.T @ self.cov_matrix @ x)
        current_return = float(np.dot(self.expected_returns, x))
        penalty = 0.0
        if current_return < self.min_return:
            violation = self.min_return - current_return
            penalty = self.penalty_manager.weight * violation
        return variance + penalty

    def is_solution_feasible(self, solution: Solution) -> bool:
        """Retorna True se a solução satisfaz o retorno mínimo."""
        current_return = float(np.dot(self.expected_returns, solution.x))
        return current_return >= self.min_return

    # ==================== SOLUÇÃO INICIAL ====================

    def generate_initial_solution(self) -> Solution:
        """
        Gera solução inicial com k_max ativos igualmente ponderados.
        Garante soma 1 e respeita [ε, δ] quando possível.
        """
        selected = np.random.choice(self.n_assets, self.k_max, replace=False)
        z = np.zeros(self.n_assets, dtype=int)
        z[selected] = 1

        x = np.zeros(self.n_assets, dtype=float)
        x[selected] = 1.0 / self.k_max

        # Clipping para respeitar [epsilon, delta] (ajuste suave)
        x[selected] = np.clip(x[selected], self.epsilon, self.delta)
        # Renormaliza mantendo apenas selecionados
        s = x.sum()
        if s > 0:
            x = x / s

        sol = Solution(z=z, x=x, cost=0.0)
        sol.cost = self.calculate_cost(sol)
        return sol

    # ==================== OPERADORES DE VIZINHANÇA ====================

    def generate_neighbors_idID(self, solution: Solution, step: Optional[float] = None) -> List[Tuple]:
        """Operador idID (increase, decrease, insert, delete)."""
        neighbors: List[Tuple] = []
        step_candidates = [0.01, 0.05, 0.1, 0.2]  # vários tamanhos de movimento

        for s in step_candidates:
            for i in range(self.n_assets):
                if solution.z[i] == 1:
                    n_inc = self._increase_asset(solution, i, s)
                    if n_inc is not None:
                        neighbors.append(('increase', i, -1, n_inc))
                    n_dec = self._decrease_asset(solution, i, s)
                    if n_dec is not None:
                        neighbors.append(('decrease', i, -1, n_dec))

            if int(solution.z.sum()) < self.k_max:
                for i in range(self.n_assets):
                    if solution.z[i] == 0:
                        n_ins = self._insert_asset(solution, i)
                        if n_ins is not None:
                            neighbors.append(('insert', i, -1, n_ins))

        return neighbors

    def _increase_asset(self, solution: Solution, asset_idx: int, step: float) -> Optional[Solution]:
        new_solution = deepcopy(solution)
        new_x_i = new_solution.x[asset_idx] * (1.0 + step)
        new_x_i = min(new_x_i, self.delta)

        delta_x = new_x_i - new_solution.x[asset_idx]
        if delta_x <= 0:
            return None

        new_solution.x[asset_idx] = new_x_i
        self._rebalance_portfolio(new_solution, exclude_idx=asset_idx, delta=-delta_x)
        new_solution.cost = self.calculate_cost(new_solution)
        return new_solution

    def _decrease_asset(self, solution: Solution, asset_idx: int, step: float) -> Optional[Solution]:
        new_solution = deepcopy(solution)
        new_x_i = new_solution.x[asset_idx] * (1.0 - step)

        if new_x_i < self.epsilon:
            return self._delete_asset(solution, asset_idx)

        delta_x = new_solution.x[asset_idx] - new_x_i
        new_solution.x[asset_idx] = new_x_i
        self._rebalance_portfolio(new_solution, exclude_idx=asset_idx, delta=delta_x)
        new_solution.cost = self.calculate_cost(new_solution)
        return new_solution

    def _insert_asset(self, solution: Solution, asset_idx: int) -> Optional[Solution]:
        if int(solution.z.sum()) >= self.k_max:
            return None

        new_solution = deepcopy(solution)
        new_solution.z[asset_idx] = 1
        add = max(self.epsilon, 1e-9)
        new_solution.x[asset_idx] = add
        self._rebalance_portfolio(new_solution, exclude_idx=asset_idx, delta=-add)
        # renormaliza
        s = new_solution.x.sum()
        if s > 0:
            new_solution.x = new_solution.x / s
        new_solution.cost = self.calculate_cost(new_solution)
        return new_solution

    def _delete_asset(self, solution: Solution, asset_idx: int) -> Solution:
        new_solution = deepcopy(solution)
        freed = new_solution.x[asset_idx]
        new_solution.z[asset_idx] = 0
        new_solution.x[asset_idx] = 0.0
        self._rebalance_portfolio(new_solution, exclude_idx=asset_idx, delta=freed)
        # renormaliza
        s = new_solution.x.sum()
        if s > 0:
            new_solution.x = new_solution.x / s
        new_solution.cost = self.calculate_cost(new_solution)
        return new_solution

    def _rebalance_portfolio(self, solution: Solution, exclude_idx: int, delta: float) -> None:
        """
        Redistribui 'delta' entre os ativos presentes (z=1) exceto 'exclude_idx',
        preservando limites [ε, δ]. Se não houver espaço proporcional, usa divisão uniforme.
        """
        active = np.where((solution.z == 1) & (np.arange(self.n_assets) != exclude_idx))[0]
        if len(active) == 0:
            return

        adjustable = np.array([max(solution.x[i] - self.epsilon, 0.0) for i in active])
        total_adjustable = float(adjustable.sum())

        if delta < 0:
            # precisamos tirar dos ativos (redução)
            take = -delta
            if total_adjustable > 0:
                for idx, i in enumerate(active):
                    w = adjustable[idx] / total_adjustable
                    solution.x[i] -= take * w
                    solution.x[i] = float(np.clip(solution.x[i], 0.0, self.delta))
            else:
                # sem espaço proporcional -> dividir uniforme
                for i in active:
                    solution.x[i] -= take / len(active)
                    solution.x[i] = float(np.clip(solution.x[i], 0.0, self.delta))
        else:
            # precisamos adicionar aos ativos
            room = np.array([max(self.delta - solution.x[i], 0.0) for i in active])
            total_room = float(room.sum())
            if total_room > 0:
                for idx, i in enumerate(active):
                    w = room[idx] / total_room if total_room > 0 else 1.0 / len(active)
                    solution.x[i] += delta * w
                    solution.x[i] = float(np.clip(solution.x[i], 0.0, self.delta))
            else:
                for i in active:
                    solution.x[i] += delta / len(active)
                    solution.x[i] = float(np.clip(solution.x[i], 0.0, self.delta))

        # garante não-negatividade e limites
        solution.x = np.clip(solution.x, 0.0, self.delta)

    def generate_neighbors_TID(self, solution: Solution, step: Optional[float] = None) -> List[Tuple]:
        """Operador TID (transfer, insert, delete via transferência)."""
        neighbors: List[Tuple] = []
        step_candidates = [0.01, 0.05, 0.1, 0.2]

        for s in step_candidates:
            for i in range(self.n_assets):
                if solution.z[i] == 0:
                    continue
                for j in range(self.n_assets):
                    if i == j:
                        continue
                    n = self._transfer_between_assets(solution, i, j, s)
                    if n is not None:
                        neighbors.append(('transfer', i, j, n))
        return neighbors


    def _transfer_between_assets(self, solution: Solution, from_idx: int, to_idx: int, step: float) -> Optional[Solution]:
        new_solution = deepcopy(solution)

        transfer = new_solution.x[from_idx] * step
        new_from = new_solution.x[from_idx] - transfer

        if new_from < self.epsilon:
            transfer = new_solution.x[from_idx]
            new_solution.x[from_idx] = 0.0
            new_solution.z[from_idx] = 0
        else:
            new_solution.x[from_idx] = new_from

        if new_solution.z[to_idx] == 0:
            if int(new_solution.z.sum()) >= self.k_max:
                return None
            new_solution.z[to_idx] = 1
            if transfer < self.epsilon:
                transfer = self.epsilon

        new_to = new_solution.x[to_idx] + transfer
        new_solution.x[to_idx] = min(new_to, self.delta)

        # renormaliza pesos para somar 1
        s = new_solution.x.sum()
        if s <= 0:
            return None
        new_solution.x = new_solution.x / s
        new_solution.cost = self.calculate_cost(new_solution)
        return new_solution

    def generate_all_neighbors(
        self,
        solution: Solution,
        use_idID: Optional[bool] = None,
        use_TID: Optional[bool] = None,
        step_override: Optional[float] = None,
        single_operator: Optional[str] = None
    ) -> List[Tuple]:
        """
        Gera vizinhança combinando operadores. Pode forçar um único operador
        (modo token-ring) e/ou sobrescrever o passo.
        """
        if use_idID is None:
            use_idID = self.use_idID
        if use_TID is None:
            use_TID = self.use_TID

        all_neighbors: List[Tuple] = []

        # token-ring: operador único por fase
        if single_operator == 'idID':
            step = step_override if step_override is not None else None
            all_neighbors.extend(self.generate_neighbors_idID(solution, step))
            return self._maybe_sample_neighbors(all_neighbors)

        if single_operator == 'TID':
            step = step_override if step_override is not None else None
            all_neighbors.extend(self.generate_neighbors_TID(solution, step))
            return self._maybe_sample_neighbors(all_neighbors)

        # padrão: combinar conforme flags
        if use_idID:
            step = step_override if step_override is not None else None
            all_neighbors.extend(self.generate_neighbors_idID(solution, step))

        if use_TID:
            step = step_override if step_override is not None else None
            all_neighbors.extend(self.generate_neighbors_TID(solution, step))

        return self._maybe_sample_neighbors(all_neighbors)

    # ==================== SELEÇÃO E CRITÉRIOS ====================

    def aspiration_criterion(self, move_cost: float, current_best_cost: float) -> bool:
        """Permite movimento tabu se melhorar o melhor global."""
        return move_cost < current_best_cost

    def select_best_neighbor(self, neighbors: List[Tuple]) -> Tuple[Optional[Tuple], Optional[Solution]]:
        """Escolhe o melhor vizinho respeitando tabu e aspiração."""
        best_neighbor = None
        best_move = None
        best_cost = float('inf')

        for move_type, asset_i, asset_j, neighbor_solution in neighbors:
            move = (move_type, asset_i, asset_j)
            is_tabu = self.tabu_list.is_tabu(move)

            # se ainda não temos melhor global (primeiras iterações)
            current_best_cost = self.best_solution.cost if self.best_solution is not None else float('inf')
            aspiration = self.aspiration_criterion(neighbor_solution.cost, current_best_cost)

            if not is_tabu or aspiration:
                if neighbor_solution.cost < best_cost:
                    best_cost = neighbor_solution.cost
                    best_neighbor = neighbor_solution
                    best_move = move

        return best_move, best_neighbor

    # ==================== HISTÓRICO ====================

    def update_history(self, iteration: int, is_feasible: bool) -> None:
        self.history['iteration'].append(int(iteration))
        self.history['current_cost'].append(float(self.current_solution.cost))
        self.history['best_cost'].append(float(self.best_solution.cost))
        self.history['n_assets'].append(int(self.current_solution.z.sum()))
        self.history['penalty_weight'].append(float(self.penalty_manager.weight))
        self.history['is_feasible'].append(bool(is_feasible))

    # ==================== AUXILIARES (Schaerf/Token-Ring) ====================

    def _current_token_operator_and_step(self) -> Tuple[Optional[str], Optional[float]]:
        """Retorna (operador, passo_base) da fase atual se token-ring estiver habilitado."""
        if not self.token_ring_enabled or not self.token_ring_sequence:
            return None, None
        phase = (self._token_phase // max(1, self.token_ring_interval)) % len(self.token_ring_sequence)
        op, base_step = self.token_ring_sequence[phase]
        return op, base_step

    def _maybe_sample_neighbors(self, neighbors: List[Tuple]) -> List[Tuple]:
        """Amostra estocasticamente a vizinhança caso neighbor_sample_ratio < 1.0."""
        return neighbors

    # ==================== LOOP PRINCIPAL ====================

    def run(
        self,
        max_iterations: Optional[int] = None,
        max_idle_iterations: Optional[int] = None,
        K: Optional[int] = None,
        H: Optional[int] = None,
        verbose: bool = True
    ) -> Solution:
        """Executa o Tabu Search e retorna a melhor Solution encontrada."""

        # Parâmetros (override)
        if max_iterations is not None:
            self.max_iterations = int(max_iterations)
        if max_idle_iterations is not None:
            self.max_idle_iterations = int(max_idle_iterations)
        if K is not None:
            self.K = int(K)
        if H is not None:
            self.H = int(H)

        # Reset estado
        self.penalty_manager.reset()
        self.tabu_list.clear()
        self.history = {k: [] for k in self.history}
        self._token_phase = 0

        # Solução inicial
        self.current_solution = self.generate_initial_solution()
        self.best_solution = deepcopy(self.current_solution)

        if verbose:
            is_feas = self.is_solution_feasible(self.current_solution)
            print("Solução inicial:")
            print(f"  Custo: {self.current_solution.cost:.6f}")
            print(f"  Ativos: {int(self.current_solution.z.sum())}")
            print(f"  Viável: {is_feas}")
            print(f"  Peso penalização: {self.penalty_manager.weight:.2f}")
            print("-" * 60)

        idle_counter = 0

        # Loop principal
        for iteration in range(self.max_iterations):
            single_operator, base_step = self._current_token_operator_and_step()

            step_override = None
            if single_operator is not None and base_step is not None:
                lower = max(0.01, base_step - self.neighbor_generator.step_variation)
                upper = min(0.99, base_step + self.neighbor_generator.step_variation)
                step_override = float(np.random.uniform(lower, upper))

            neighbors = self.generate_all_neighbors(
                self.current_solution,
                use_idID=self.use_idID,
                use_TID=self.use_TID,
                step_override=step_override,
                single_operator=single_operator
            )
            print(f"[DEBUG] Iteração {iteration}: {len(neighbors)} vizinhos gerados")
            if len(neighbors) == 0:
                if verbose:
                    print(f"Iteração {iteration}: vizinhança vazia. Encerrando.")
                break

            best_move, best_neighbor = self.select_best_neighbor(neighbors)
            if best_neighbor is None:
                if verbose:
                    print(f"Iteração {iteration}: nenhum movimento permitido. Encerrando.")
                break

            # aplica movimento
            self.current_solution = best_neighbor
            self.tabu_list.add_move(best_move)

            # shifting penalty
            feasible = self.is_solution_feasible(self.current_solution)
            weight_changed = self.penalty_manager.update(feasible, self.K, self.H)
            if weight_changed:
                self.current_solution.cost = self.calculate_cost(self.current_solution)
                self.best_solution.cost = self.calculate_cost(self.best_solution)

            # melhora global?
            if self.current_solution.cost < self.best_solution.cost:
                self.best_solution = deepcopy(self.current_solution)
                idle_counter = 0
                if verbose:
                    print(f"Iteração {iteration}: nova melhor solução!")
                    print(f"  Custo: {self.best_solution.cost:.6f}")
                    print(f"  Ativos: {int(self.best_solution.z.sum())}")
                    print(f"  Viável: {feasible}")
                    print(f"  Movimento: {best_move[0]}")
                    print(f"  Peso: {self.penalty_manager.weight:.2f}")
            else:
                idle_counter += 1

            # manutenção
            self.tabu_list.update()
            self.update_history(iteration, feasible)

            # estagnação
            if idle_counter >= self.max_idle_iterations:
                if verbose:
                    print(f"\nParada por estagnação: {idle_counter} iterações sem melhoria.")
                break

            # token-ring: avança fase
            if self.token_ring_enabled:
                self._token_phase += 1

            # log periódico
            if verbose and (iteration + 1) % 50 == 0:
                print(f"\nIteração {iteration + 1}/{self.max_iterations}")
                print(f"  Custo atual: {self.current_solution.cost:.6f}")
                print(f"  Melhor custo: {self.best_solution.cost:.6f}")
                print(f"  Viável: {feasible}")
                print(f"  Peso: {self.penalty_manager.weight:.2f}")
                print(f"  Tabu: {len(self.tabu_list.tabu_moves)} movimentos")
                print(f"  Idle: {idle_counter}")

        if verbose:
            final_feasible = self.is_solution_feasible(self.best_solution)
            print("\n" + "=" * 60)
            print("BUSCA FINALIZADA")
            print(f"Melhor custo: {self.best_solution.cost:.6f}")
            print(f"Ativos: {int(self.best_solution.z.sum())}")
            print(f"Viável: {final_feasible}")
            print(f"Iterações: {len(self.history['iteration'])}")
            print("=" * 60)

        return self.best_solution

    # ==================== INFORMAÇÕES ====================

    def get_portfolio_info(self, solution: Optional[Solution] = None) -> Optional[Dict]:
        if solution is None:
            solution = self.best_solution
        if solution is None:
            return None

        port_return = float(np.dot(self.expected_returns, solution.x))
        variance = float(solution.x.T @ self.cov_matrix @ solution.x)
        std = float(np.sqrt(max(variance, 0.0)))
        sharpe = (port_return / std) if std > 0 else 0.0
        selected = np.where(solution.z == 1)[0]

        return {
            "return": port_return,
            "variance": variance,
            "std_dev": std,
            "sharpe_ratio": sharpe,
            "n_assets": int(len(selected)),
            "selected_assets": selected.tolist(),
            "weights": {int(i): float(solution.x[i]) for i in selected},
            "is_feasible": self.is_solution_feasible(solution)
        }


# ==================== FUNÇÃO AUXILIAR (opcional) ====================

def run_multiple_trials(tabu_search: TabuSearch, n_trials: int = 5, verbose: bool = False) -> Tuple[Solution, List[Solution]]:
    """Roda múltiplas execuções e retorna a melhor solução encontrada."""
    all_solutions: List[Solution] = []
    best_overall: Optional[Solution] = None
    best_cost = float('inf')

    print(f"Executando {n_trials} rodadas do Tabu Search...")
    print("=" * 60)

    for trial in range(n_trials):
        print(f"\nRodada {trial + 1}/{n_trials}")
        print("-" * 60)
        sol = tabu_search.run(verbose=verbose)
        all_solutions.append(sol)
        if sol.cost < best_cost:
            best_cost = sol.cost
            best_overall = deepcopy(sol)
            print(f"  → Nova melhor global! Custo: {best_cost:.6f}")

    print("\n" + "=" * 60)
    print("TODAS AS RODADAS CONCLUÍDAS")
    print(f"Melhor custo: {best_cost:.6f}")
    print("=" * 60)

    return best_overall, all_solutions
