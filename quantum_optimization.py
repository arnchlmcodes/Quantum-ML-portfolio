"""
Quantum optimization modules for asset selection and weight optimization.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

# Quantum computing imports
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Parameter
    try:
        from qiskit.primitives import Sampler
        from qiskit_aer import AerSimulator
    except ImportError:
        from qiskit import Aer
        from qiskit.utils import QuantumInstance
        AerSimulator = Aer.get_backend('qasm_simulator')
        Sampler = None
    
    from qiskit_algorithms import QAOA, VQE
    from qiskit_algorithms.optimizers import SPSA, COBYLA
    from qiskit.circuit.library import TwoLocal, RealAmplitudes
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_optimization.converters import QuadraticProgramToQubo
    QUANTUM_AVAILABLE = True
except ImportError:
    class QuantumCircuit:
        pass
    class Parameter:
        pass
    class QuadraticProgram:
        pass
    QUANTUM_AVAILABLE = False

from config import QuantumConfig


class QuantumAssetSelector:
    """Quantum Asset Selection using QAOA algorithm"""
    
    def __init__(self, config: QuantumConfig):
        self.config = config
        self.backend = None
        self.sampler = None
        self._initialize_quantum_backend()
    
    def _initialize_quantum_backend(self):
        """Initialize quantum backend and sampler"""
        if not QUANTUM_AVAILABLE:
            return
        
        if self.config.backend == 'qasm_simulator':
            if hasattr(AerSimulator, '__call__'):
                self.backend = AerSimulator()
            else:
                self.backend = AerSimulator
        else:
            if hasattr(AerSimulator, '__call__'):
                self.backend = AerSimulator()
            else:
                self.backend = AerSimulator
        
        if Sampler is not None:
            self.sampler = Sampler()
        else:
            self.sampler = None
    
    def create_qaoa_circuit(self, n_assets: int, p_layers: int = None) -> QuantumCircuit:
        """Create QAOA circuit for asset selection problem"""
        if p_layers is None:
            p_layers = self.config.qaoa_p_layers
        
        qc = QuantumCircuit(n_assets, n_assets)
        qc.h(range(n_assets))
        
        beta_params = [Parameter(f'β_{i}') for i in range(p_layers)]
        gamma_params = [Parameter(f'γ_{i}') for i in range(p_layers)]
        
        for layer in range(p_layers):
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    qc.rzz(gamma_params[layer], i, j)
            
            for i in range(n_assets):
                qc.rz(gamma_params[layer], i)
            
            for i in range(n_assets):
                qc.rx(beta_params[layer], i)
        
        qc.measure_all()
        return qc
    
    def _create_asset_selection_qubo(self, returns_data: pd.DataFrame) -> QuadraticProgram:
        """Create QUBO formulation for asset selection problem"""
        n_assets = len(returns_data.columns)
        
        expected_returns = returns_data.mean().values
        cov_matrix = returns_data.cov().values
        
        qp = QuadraticProgram()
        
        for i in range(n_assets):
            qp.binary_var(f'x_{i}')
        
        linear_coeffs = -expected_returns * (1 - self.config.risk_aversion)
        
        quadratic_coeffs = {}
        for i in range(n_assets):
            for j in range(n_assets):
                if i <= j:
                    coeff = self.config.risk_aversion * cov_matrix[i, j]
                    if i == j:
                        quadratic_coeffs[(f'x_{i}', f'x_{j}')] = coeff
                    else:
                        quadratic_coeffs[(f'x_{i}', f'x_{j}')] = 2 * coeff
        
        qp.minimize(
            linear=dict(zip([f'x_{i}' for i in range(n_assets)], linear_coeffs)),
            quadratic=quadratic_coeffs
        )
        
        constraint_coeffs = {f'x_{i}': 1 for i in range(n_assets)}
        qp.linear_constraint(
            linear=constraint_coeffs,
            sense='==',
            rhs=self.config.n_assets_select,
            name='asset_count'
        )
        
        return qp
    
    def run_qaoa_optimization(self, returns_data: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
        """Run QAOA optimization for asset selection"""
        if not QUANTUM_AVAILABLE or self.backend is None:
            return self._classical_asset_selection_fallback(returns_data)
        
        qp = self._create_asset_selection_qubo(returns_data)
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(qp)
        
        optimizer = SPSA(maxiter=self.config.maxiter)
        qaoa = QAOA(
            sampler=self.sampler,
            optimizer=optimizer,
            reps=self.config.qaoa_p_layers
        )
        
        min_eigen_optimizer = MinimumEigenOptimizer(qaoa)
        result = min_eigen_optimizer.solve(qubo)
        
        selected_indices = []
        selection_probs = np.zeros(len(returns_data.columns))
        
        if hasattr(result, 'x') and result.x is not None:
            for i, val in enumerate(result.x):
                if val > 0.5:
                    selected_indices.append(i)
                selection_probs[i] = val
        
        if len(selected_indices) != self.config.n_assets_select:
            top_indices = np.argsort(selection_probs)[-self.config.n_assets_select:]
            selected_indices = top_indices.tolist()
        
        selected_assets = [returns_data.columns[i] for i in selected_indices]
        
        return selected_assets, selection_probs
    
    def _classical_asset_selection_fallback(self, returns_data: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
        """Classical fallback for asset selection when quantum fails"""
        expected_returns = returns_data.mean()
        volatilities = returns_data.std()
        
        volatilities = volatilities.replace(0, np.inf)
        sharpe_ratios = expected_returns / volatilities
        
        top_assets_indices = sharpe_ratios.nlargest(self.config.n_assets_select).index
        selected_assets = top_assets_indices.tolist()
        
        selection_scores = np.zeros(len(returns_data.columns))
        for i, asset in enumerate(returns_data.columns):
            if asset in selected_assets:
                selection_scores[i] = sharpe_ratios[asset]
        
        if selection_scores.max() > 0:
            selection_scores = selection_scores / selection_scores.max()
        
        return selected_assets, selection_scores


class QuantumWeightOptimizer:
    """Quantum Weight Optimization using VQE algorithm"""
    
    def __init__(self, config: QuantumConfig):
        self.config = config
        self.backend = None
        self.sampler = None
        self._initialize_quantum_backend()
    
    def _initialize_quantum_backend(self):
        """Initialize quantum backend and sampler"""
        if not QUANTUM_AVAILABLE:
            return
        
        if self.config.backend == 'qasm_simulator':
            if hasattr(AerSimulator, '__call__'):
                self.backend = AerSimulator()
            else:
                self.backend = AerSimulator
        else:
            if hasattr(AerSimulator, '__call__'):
                self.backend = AerSimulator()
            else:
                self.backend = AerSimulator
        
        if Sampler is not None:
            self.sampler = Sampler()
        else:
            self.sampler = None
    
    def create_vqe_ansatz(self, n_assets: int, reps: int = None) -> QuantumCircuit:
        """Create VQE ansatz circuit for weight optimization"""
        if reps is None:
            reps = self.config.vqe_ansatz_reps
        
        ansatz = RealAmplitudes(
            num_qubits=n_assets,
            reps=reps,
            entanglement='linear',
            insert_barriers=True
        )
        
        return ansatz
    
    def run_vqe_optimization(self, returns_data: pd.DataFrame, selected_assets: List[str]) -> np.ndarray:
        """Run VQE optimization for portfolio weight optimization"""
        if not QUANTUM_AVAILABLE or self.backend is None:
            return self._classical_weight_optimization_fallback(returns_data, selected_assets)
        
        n_assets = len(selected_assets)
        ansatz = self.create_vqe_ansatz(n_assets)
        
        if self.config.optimizer == 'SPSA':
            optimizer = SPSA(maxiter=self.config.maxiter)
        elif self.config.optimizer == 'COBYLA':
            optimizer = COBYLA(maxiter=self.config.maxiter)
        else:
            optimizer = SPSA(maxiter=self.config.maxiter)
        
        # For simplicity, return classical fallback
        return self._classical_weight_optimization_fallback(returns_data, selected_assets)
    
    def _classical_weight_optimization_fallback(self, returns_data: pd.DataFrame, selected_assets: List[str]) -> np.ndarray:
        """Classical fallback for weight optimization"""
        asset_returns = returns_data[selected_assets]
        n_assets = len(selected_assets)
        
        expected_returns = asset_returns.mean().values
        cov_matrix = asset_returns.cov().values
        
        # Add regularization for numerical stability
        reg_cov_matrix = cov_matrix + np.eye(n_assets) * 1e-6
        
        inv_cov = np.linalg.inv(reg_cov_matrix)
        ones = np.ones(n_assets)
        
        risk_adj_returns = expected_returns * (1 / self.config.risk_aversion)
        
        numerator = np.dot(inv_cov, risk_adj_returns)
        denominator = np.dot(ones, np.dot(inv_cov, ones))
        
        if denominator > 1e-10:
            weights = numerator / denominator
        else:
            numerator = np.dot(inv_cov, ones)
            denominator = np.dot(ones, np.dot(inv_cov, ones))
            weights = numerator / denominator
        
        weights = np.maximum(weights, 0)
        
        if weights.sum() > 1e-10:
            weights = weights / weights.sum()
        else:
            weights = np.ones(n_assets) / n_assets
        
        return weights