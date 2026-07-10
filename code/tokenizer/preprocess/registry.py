from typing import Callable, Dict

PreprocessFn = Callable[..., None]
_REGISTRY: Dict[str, PreprocessFn] = {}


def register_preprocessor(name: str) -> Callable[[PreprocessFn], PreprocessFn]:
    def decorator(fn: PreprocessFn) -> PreprocessFn:
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_preprocessor(name: str) -> PreprocessFn:
    if name not in _REGISTRY:
        raise KeyError(f"Preprocessor '{name}' is not registered.")
    return _REGISTRY[name]


def available_preprocessors() -> list[str]:
    return sorted(_REGISTRY.keys())
