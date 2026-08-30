#!/usr/bin/env python3
"""Mirror training checkpoints to a Hugging Face repo, skipping what is already there.

Written because both training boxes are rented and may be reclaimed at short notice, so a
checkpoint that exists only on local disk is one eviction away from being lost.

Skipping is by **per-file size**, not by folder presence. A partially uploaded folder is the
common failure here, and a folder-level check would silently accept it. Size is a weak
check, but the Xet backend does not report per-file hashes in the tree listing for every
repo, and a weak check that is actually performed beats a strong one that is skipped.

Usage:
    python3 hf_mirror.py --src <ckpt-root> --repo <org/name> --prefix <path-in-repo> \
        --token-file <file> [--dry-run] [--only globalstep124,globalstep115]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def local_manifest(root: Path) -> dict[str, int]:
    """Map every file under ``root`` to its size, keyed by path relative to ``root``.

    Args:
        root: Directory to walk.

    Returns:
        ``{relative_posix_path: size_bytes}``.
    """
    out: dict[str, int] = {}
    for p in root.rglob("*"):
        if p.is_file():
            out[p.relative_to(root).as_posix()] = p.stat().st_size
    return out


def remote_manifest(api, repo: str, prefix: str) -> dict[str, int]:
    """Sizes of files already in ``repo`` under ``prefix``.

    Returns an empty mapping when the prefix does not exist yet, which is not an error --
    it is the normal state for the first upload.
    """
    try:
        # list() inside the try, not outside: list_repo_tree returns a GENERATOR, so a 404
        # for a prefix that does not exist yet is raised on first iteration. Leaving the
        # iteration outside the try lets that 404 escape the handler written to absorb it.
        entries = list(
            api.list_repo_tree(repo, path_in_repo=prefix, recursive=True, repo_type="model")
        )
    except Exception as exc:  # noqa: BLE001 - a missing prefix surfaces under several names
        # huggingface_hub has renamed these across versions (EntryNotFoundError,
        # RemoteEntryNotFoundError, RepositoryNotFoundError), so match on the signal that
        # has stayed stable rather than on a class that moves.
        text = str(exc).lower()
        if "404" in text or "not found" in text or "does not exist" in text:
            return {}
        raise
    out: dict[str, int] = {}
    for e in entries:
        size = getattr(e, "size", None)
        lfs = getattr(e, "lfs", None)
        if size is None and lfs is not None:
            size = getattr(lfs, "size", None)
        if size is not None:
            out[e.path[len(prefix) + 1 :] if e.path.startswith(prefix + "/") else e.path] = size
    return out


def main() -> int:
    """Mirror each checkpoint directory, newest first, skipping complete ones."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="directory holding checkpoint subdirectories")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--prefix", required=True, help="path inside the repo, e.g. step0k")
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--only", default="", help="comma-separated substrings; upload only matches")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    token = Path(args.token_file).read_text().strip()
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)

    src = Path(args.src)
    wanted = [w for w in args.only.split(",") if w]
    # Newest first: if the box is reclaimed mid-upload, the most recent checkpoint survives.
    dirs = sorted((d for d in src.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True)
    if wanted:
        dirs = [d for d in dirs if any(w in d.name for w in wanted)]
    if not dirs:
        print(f"no checkpoint directories under {src}", file=sys.stderr)
        return 1

    for d in dirs:
        prefix = f"{args.prefix}/{d.name}"
        loc = local_manifest(d)
        rem = remote_manifest(api, args.repo, prefix)
        missing = {k: v for k, v in loc.items() if rem.get(k) != v}
        gb = sum(missing.values()) / 1e9
        if not missing:
            print(f"SKIP {d.name}: {len(loc)} files already match by size", flush=True)
            continue
        print(f"UPLOAD {d.name}: {len(missing)}/{len(loc)} files, {gb:.2f} GB", flush=True)
        if args.dry_run:
            continue
        api.upload_folder(
            folder_path=str(d),
            path_in_repo=prefix,
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"mirror {args.prefix}/{d.name}",
        )
        # Verify rather than trust the call's return: the Xet backend reports "0.00B
        # transferred" for real multi-GB uploads, so the echo cannot be used as evidence.
        rem2 = remote_manifest(api, args.repo, prefix)
        still = {k: v for k, v in loc.items() if rem2.get(k) != v}
        if still:
            print(f"  INCOMPLETE {d.name}: {len(still)} files still mismatched", flush=True)
        else:
            print(f"  VERIFIED {d.name}: all {len(loc)} files match by size", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
