"""Sync checkpoints/ and runs/ with Hugging Face Hub.

No Google Cloud or OAuth app needed — just a free HuggingFace account and
an access token. Files already on the Hub with identical content are skipped
on upload; files already present locally are skipped on download.

First-time setup:
  1. Create a free account at https://huggingface.co
  2. Generate a token at https://huggingface.co/settings/tokens (role: write)
  3. pip install huggingface_hub
  4. Log in once:
       huggingface-cli login
     (or set the HF_TOKEN environment variable)

Upload:
  python scripts/hf_sync.py --repo rgferrari/multi-agent-attention-masters
  python scripts/hf_sync.py --repo rgferrari/multi-agent-attention-masters --checkpoints
  python scripts/hf_sync.py --repo rgferrari/multi-agent-attention-masters --runs
  python scripts/hf_sync.py --repo rgferrari/multi-agent-attention-masters --dry-run

Download (on another machine):
  python scripts/hf_sync.py --repo rgferrari/multi-agent-attention-masters --download
  python scripts/hf_sync.py --repo rgferrari/multi-agent-attention-masters --download --checkpoints
  python scripts/hf_sync.py --repo rgferrari/multi-agent-attention-masters --download --runs

The HF repo is created automatically as private if it doesn't exist yet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _get_api():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("Missing package. Install with:\n  pip install huggingface_hub")
    return HfApi()


def _ensure_repo(api, repo_id: str) -> None:
    from huggingface_hub.utils import RepositoryNotFoundError
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
    except RepositoryNotFoundError:
        print(f"Creating private repo: {repo_id}")
        api.create_repo(repo_id=repo_id, repo_type="model", private=True)


def _upload(api, local: Path, repo_id: str, dry_run: bool) -> None:
    dest = local.name
    print(f"\n{'[dry-run] ' if dry_run else ''}→ {local.relative_to(REPO_ROOT)}/ → hf:{repo_id}/{dest}/")
    if dry_run:
        files = list(local.rglob("*"))
        n_files = sum(1 for f in files if f.is_file())
        total_mb = sum(f.stat().st_size for f in files if f.is_file()) / 1024 ** 2
        print(f"   {n_files} files, {total_mb:.1f} MB")
        return

    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(local),
        path_in_repo=dest,
        commit_message=f"sync {dest}/",
    )


def _download(api, folder: str, repo_id: str) -> None:
    from huggingface_hub import snapshot_download
    print(f"\n← hf:{repo_id}/{folder}/ → {folder}/")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=f"{folder}/**",
        local_dir=str(REPO_ROOT),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync checkpoints and runs with Hugging Face Hub")
    parser.add_argument("--repo", required=True, help="HF repo id, e.g. rgferrari/multi-agent-attention-masters")
    parser.add_argument("--checkpoints", action="store_true", help="checkpoints/ only")
    parser.add_argument("--runs", action="store_true", help="runs/ only")
    parser.add_argument("--download", action="store_true", help="Download from Hub instead of uploading")
    parser.add_argument("--dry-run", action="store_true", help="Preview upload without transferring (upload only)")
    args = parser.parse_args()

    do_checkpoints = args.checkpoints or not (args.checkpoints or args.runs)
    do_runs = args.runs or not (args.checkpoints or args.runs)

    api = _get_api()

    if args.download:
        if do_checkpoints:
            _download(api, "checkpoints", args.repo)
        if do_runs:
            _download(api, "runs", args.repo)
    else:
        targets: list[Path] = []
        if do_checkpoints:
            p = REPO_ROOT / "checkpoints"
            targets.append(p) if p.exists() else print("checkpoints/ not found, skipping")
        if do_runs:
            p = REPO_ROOT / "runs"
            targets.append(p) if p.exists() else print("runs/ not found, skipping")

        if not targets:
            sys.exit("Nothing to upload.")

        if not args.dry_run:
            _ensure_repo(api, args.repo)
        for target in targets:
            _upload(api, target, args.repo, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
