from .generator import SimulationRunner
# Try to expose the core if available
try:
    from .core import SymplecticIntegrator
except ImportError:
    pass

__version__ = "0.1.0"
