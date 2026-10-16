"""
Centralized random seed management for reproducible PEGKi simulations.

This module provides a single source of truth for all randomness in the framework,
ensuring bit-level reproducibility across runs when the same master seed is used.
"""

import random
import zlib
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class SeedScope(Enum):
    """Enumeration of different random number scopes in the simulation."""
    MASTER = "master"
    AGENT_INITIALIZATION = "agent_init"
    POLICY_LEARNING = "policy_learning"
    COLOR_MIXING = "color_mixing"
    MATCHMAKING = "matchmaking"
    RATING_SYSTEM = "rating_system"
    TARGET_GENERATION = "target_generation"
    NOISE_INJECTION = "noise_injection"
    VISUALIZATION = "visualization"


@dataclass
class SeedState:
    """Container for random number generator states."""
    python_state: Any
    numpy_state: Dict[str, Any]
    torch_state: Any


class SimulationSeed:
    """
    Centralized random seed manager for reproducible simulations.
    
    This class manages all random number generation in PEGKí, providing:
    - Deterministic seeding from a master seed
    - Scope-specific random number generators
    - State saving/loading for checkpointing
    - Bit-level reproducibility guarantees
    
    Example:
        >>> seed_manager = SimulationSeed(master_seed=42)
        >>> agent_rng = seed_manager.get_rng(SeedScope.AGENT_INITIALIZATION)
        >>> policy_rng = seed_manager.get_rng(SeedScope.POLICY_LEARNING)
        >>> 
        >>> # Later, reset to same state
        >>> seed_manager.reset_to_master()
    """
    
    def __init__(self, master_seed: int = 42):
        """
        Initialize the simulation seed manager.
        
        Args:
            master_seed: Master seed for all randomness in the simulation
        """
        self.master_seed = master_seed
        self._scope_seeds: Dict[SeedScope, int] = {}
        self._rng_instances: Dict[SeedScope, np.random.Generator] = {}
        self._initial_states: Dict[SeedScope, SeedState] = {}
        
        # Initialize all scopes with deterministic seeds derived from master
        self._initialize_seeds()
        self._setup_global_seeds()
        
    def _initialize_seeds(self) -> None:
        """Initialize deterministic seeds for each scope based on master seed."""
        # Use numpy random generator to create deterministic scope seeds
        master_rng = np.random.default_rng(self.master_seed)
        
        for scope in SeedScope:
            # Generate deterministic seed for each scope
            scope_seed = int(master_rng.integers(0, 2**31 - 1))
            self._scope_seeds[scope] = scope_seed
            
            # Create dedicated RNG instance for this scope
            self._rng_instances[scope] = np.random.default_rng(scope_seed)
            
        # Store initial states for reset capability
        self._save_initial_states()
        
    def _setup_global_seeds(self) -> None:
        """Setup global random number generators."""
        # Set global Python random seed
        random.seed(self._scope_seeds[SeedScope.MASTER])
        
        # Set global NumPy seed
        np.random.seed(self._scope_seeds[SeedScope.MASTER])
        
        # Set PyTorch seeds if available (lazy import to avoid slow startup)
        try:
            import torch as _torch
            _torch.manual_seed(self._scope_seeds[SeedScope.MASTER])
            if _torch.cuda.is_available():
                _torch.cuda.manual_seed(self._scope_seeds[SeedScope.MASTER])
                _torch.cuda.manual_seed_all(self._scope_seeds[SeedScope.MASTER])
        except ImportError:
            pass  # PyTorch not available
            
    def _save_initial_states(self) -> None:
        """Save initial RNG states for reset capability."""
        for scope in SeedScope:
            self._initial_states[scope] = SeedState(
                python_state=random.getstate(),
                numpy_state=self._rng_instances[scope].bit_generator.state,
                torch_state=__import__('torch').get_rng_state() if 'torch' in __import__('sys').modules else None
            )
    
    def get_rng(self, scope: SeedScope) -> np.random.Generator:
        """
        Get a dedicated random number generator for a specific scope.
        
        Args:
            scope: The scope for which to get the RNG
            
        Returns:
            NumPy random generator for the specified scope
        """
        return self._rng_instances[scope]
    
    def get_seed(self, scope: SeedScope) -> int:
        """
        Get the seed value for a specific scope.
        
        Args:
            scope: The scope for which to get the seed
            
        Returns:
            Seed value for the specified scope
        """
        return self._scope_seeds[scope]
    
    def reset_to_master(self) -> None:
        """Reset all random number generators to their initial states."""
        self._initialize_seeds()
        self._setup_global_seeds()
    
    def reset_scope(self, scope: SeedScope) -> None:
        """
        Reset a specific scope to its initial state.
        
        Args:
            scope: The scope to reset
        """
        scope_seed = self._scope_seeds[scope]
        self._rng_instances[scope] = np.random.default_rng(scope_seed)
    
    def get_child_seed(self, scope: SeedScope, identifier: str) -> int:
        """
        Generate a deterministic child seed for sub-components.
        
        Args:
            scope: Parent scope
            identifier: Unique identifier for the child component
            
        Returns:
            Deterministic seed based on scope and identifier
        """
        # crc32, not hash(): str hashing is salted per process, which made
        # these "deterministic" seeds differ on every run and every worker
        combined_string = f"{scope.value}_{identifier}_{self.master_seed}"
        return zlib.crc32(combined_string.encode("utf-8")) % (2**31 - 1)
    
    def save_state(self) -> Dict[str, Any]:
        """
        Save current state of all RNGs for checkpointing.
        
        Returns:
            Dictionary containing all RNG states
        """
        state = {
            'master_seed': self.master_seed,
            'scope_seeds': self._scope_seeds.copy(),
            'rng_states': {}
        }
        
        for scope, rng in self._rng_instances.items():
            state['rng_states'][scope.value] = rng.bit_generator.state
            
        return state
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load RNG states from saved state.
        
        Args:
            state: Previously saved state dictionary
        """
        self.master_seed = state['master_seed']
        self._scope_seeds = state['scope_seeds'].copy()
        
        # Restore RNG states
        for scope_name, rng_state in state['rng_states'].items():
            scope = SeedScope(scope_name)
            self._rng_instances[scope] = np.random.default_rng()
            self._rng_instances[scope].bit_generator.state = rng_state
    
    def get_reproducibility_info(self) -> Dict[str, Any]:
        """
        Get information about current seed configuration for reproducibility reports.
        
        Returns:
            Dictionary with reproducibility information
        """
        return {
            'master_seed': self.master_seed,
            'framework_version': 'PEGKi-1.0',
            'scope_seeds': {scope.value: seed for scope, seed in self._scope_seeds.items()},
            'python_version': f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
            'numpy_version': np.__version__,
            'torch_version': __import__('sys').modules['torch'].__version__ if 'torch' in __import__('sys').modules else None
        }


# Global instance for framework-wide use
_global_seed_manager: Optional[SimulationSeed] = None


def get_seed_manager() -> SimulationSeed:
    """Get the global seed manager instance."""
    global _global_seed_manager
    if _global_seed_manager is None:
        _global_seed_manager = SimulationSeed()
    return _global_seed_manager


def initialize_global_seed(master_seed: int = 42) -> SimulationSeed:
    """
    Initialize the global seed manager with a specific master seed.
    
    Args:
        master_seed: Master seed for the entire simulation
        
    Returns:
        The initialized global seed manager
    """
    global _global_seed_manager
    _global_seed_manager = SimulationSeed(master_seed)
    return _global_seed_manager


def get_rng(scope: SeedScope) -> np.random.Generator:
    """
    Convenience function to get RNG for a scope from global manager.
    
    Args:
        scope: The scope for which to get the RNG
        
    Returns:
        NumPy random generator for the specified scope
    """
    return get_seed_manager().get_rng(scope)