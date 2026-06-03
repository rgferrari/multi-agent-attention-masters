"""Shared helpers for training scripts."""

from __future__ import annotations

import os
import re


_RUN_DIR_RE = re.compile(r"^run(\d+)$")


def _last_run_number(base_dir: str) -> int:
    os.makedirs(base_dir, exist_ok=True)
    last_run = 0
    for entry in os.listdir(base_dir):
        path = os.path.join(base_dir, entry)
        if not os.path.isdir(path):
            continue
        match = _RUN_DIR_RE.match(entry)
        if match:
            last_run = max(last_run, int(match.group(1)))
    return last_run


def get_next_run_dir(base_dir: str) -> str:
    """Return the next available run directory inside ``base_dir``.

    The directory name follows the pattern ``runN`` where ``N`` starts at 1.
    """
    last_run = _last_run_number(base_dir)

    next_run = last_run + 1
    return os.path.join(base_dir, f"run{next_run}")


def get_next_shared_run_dirs(*base_dirs: str) -> list[str]:
    """Return next run directories sharing the same run index across roots."""
    if not base_dirs:
        raise ValueError("At least one base_dir is required")

    next_run = max(_last_run_number(base_dir) for base_dir in base_dirs) + 1
    run_name = f"run{next_run}"
    return [os.path.join(base_dir, run_name) for base_dir in base_dirs]
