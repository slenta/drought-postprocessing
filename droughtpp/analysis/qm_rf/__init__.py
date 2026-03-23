"""qm_rf workflow package"""


def run_qm_then_rf(*args, **kwargs):
    from .main_qm_rf import run_qm_then_rf as _run_qm_then_rf

    return _run_qm_then_rf(*args, **kwargs)


__all__ = [
    "run_qm_then_rf",
    "main_qm_rf",
    "random_forest",
    "quantile_mapping",
    "utils",
]
