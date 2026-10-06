"""Refuse staged work that would consume the requested 15 GiB reserve."""

from repository import ASSETS, BUILD, SITE
import shutil
from common import PROOF

RESERVE_BYTES = 15 * 1024**3


def tree_bytes(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def require_space(estimated_bytes, *, path=PROOF):
    if not isinstance(estimated_bytes, int) or estimated_bytes < 0:
        raise ValueError("Run footprint must be a nonnegative byte estimate")
    free = shutil.disk_usage(path).free
    required = RESERVE_BYTES + estimated_bytes
    if free < required:
        raise OSError(
            f"Insufficient storage: {free} bytes free; need {required} "
            f"(15 GiB reserve plus {estimated_bytes} estimated run bytes)"
        )
    return dict(
        free_bytes=free,
        reserve_bytes=RESERVE_BYTES,
        estimated_run_bytes=estimated_bytes,
        required_bytes=required,
    )


def build_budget():
    # Conservative bound includes a complete extra site, staging, export
    # intermediates, and evidence. Shared datasets can reduce actual allocation.
    estimate = 2 * tree_bytes(SITE) + tree_bytes(ASSETS) + tree_bytes(BUILD) + 1024**3
    return require_space(estimate)
