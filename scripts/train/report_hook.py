"""Optional end-of-training report generation.

Called by every training loop after ``writer.close()`` (the event file is only
guaranteed flushed at that point, and the report reads it back off disk).

Opt-in via ``common.auto_report``; off by default so existing invocations and
``train_all.sh`` are unchanged. Every failure is swallowed and reported as a
warning -- a plotting error must never fail a finished training run.

Scope is deliberately the environment that just finished, not a cross-algorithm
comparison: the other algorithms may not have been trained yet, which would
produce misleading partial figures. The four-way comparison belongs in
``train_all.sh``, after every run completes.
"""

from __future__ import annotations

import os
from pathlib import Path


def _parse_run_path(args, log_dir: str) -> tuple[str, str] | None:
    """Recover ``(env_type, env)`` from the run directory layout.

    All loops write to ``<log_dir>/<env_type>/<env>/<algo>/runN``; deriving the
    pair from the path keeps this generic rather than special-casing each
    algorithm's config.
    """
    base = getattr(args, "log_dir", "runs")
    try:
        rel = os.path.relpath(log_dir, base)
    except ValueError:
        return None
    parts = [p for p in rel.split(os.sep) if p and p != os.curdir]
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def maybe_generate_report(args, log_dir: str) -> None:
    if not getattr(args, "auto_report", False):
        return

    parsed = _parse_run_path(args, log_dir)
    if parsed is None:
        print(f"[report] skipped: could not derive env_type/env from {log_dir}")
        return
    env_type, env = parsed

    try:
        # Imported lazily so a missing matplotlib never breaks training.
        from reports.report import generate_run_report

        log_root = getattr(args, "log_dir", "runs")
        # A non-default log_dir means this is a scratch/experimental run, so
        # keep its figures beside it instead of overwriting reports/.
        out_dir = None
        if os.path.normpath(log_root) != "runs":
            out_dir = Path(log_root).parent / "reports" / env_type / env

        written = generate_run_report(env_type, env, out_dir=out_dir, runs_root=log_root)
    except Exception as exc:  # noqa: BLE001 - never fail training over a plot
        print(f"[report] skipped ({type(exc).__name__}): {exc}")
        return

    for path in written:
        print(f"[report] {path}")
