from .utils import lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules={'engine', 'metric', 'modules', 'solver', 'utils'},
    submod_attrs={}
)

__version__ = '0.0.1'
