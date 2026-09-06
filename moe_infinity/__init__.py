try:
    # Generated at build time by setuptools-scm (see pyproject.toml).
    from ._version import __version__
except Exception:  # pragma: no cover - source tree built without setuptools-scm
    __version__ = "0.0.0+unknown"

__all__ = ["MoE", "OffloadEngine", "__version__"]


def __getattr__(name: str):
    import importlib

    if name == "MoE":
        module = importlib.import_module(
            "moe_infinity.entrypoints.big_modeling"
        )
        return getattr(module, "MoE")
    if name == "OffloadEngine":
        module = importlib.import_module("moe_infinity.runtime")
        return getattr(module, "OffloadEngine")
    raise AttributeError(f"module 'moe_infinity' has no attribute {name!r}")
