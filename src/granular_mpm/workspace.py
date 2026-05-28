from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def scan_workspace(root: Path, max_depth: int = 2) -> dict[str, Any]:
    root = root.resolve()
    repos = []
    for candidate in _walk_limited(root, max_depth=max_depth):
        if (candidate / ".git").exists():
            repos.append(
                {
                    "name": candidate.name,
                    "path": candidate.as_posix(),
                    "has_pyproject": (candidate / "pyproject.toml").exists(),
                    "has_requirements": (candidate / "requirements.txt").exists(),
                    "has_dockerfile": any(p.name.lower() == "dockerfile" for p in candidate.iterdir() if p.is_file()),
                    "has_environment_yml": (candidate / "environment.yml").exists() or (candidate / "environment.yaml").exists(),
                }
            )

    env = {
        "python": sys.executable,
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "docker": bool(os.environ.get("container") or Path("/.dockerenv").exists()),
    }
    local_envs = [
        path.as_posix()
        for path in [root / ".venv", root / "venv"]
        if path.exists()
    ]
    return {
        "root": root.as_posix(),
        "repo_count": len(repos),
        "repos": repos,
        "environment": env,
        "local_envs": local_envs,
    }


def _walk_limited(root: Path, max_depth: int) -> list[Path]:
    out: list[Path] = []
    root_depth = len(root.parts)
    for current, dirs, _files in os.walk(root):
        path = Path(current)
        depth = len(path.parts) - root_depth
        if depth > max_depth:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".pytest_cache", "outputs", "dist"}]
        out.append(path)
    return out
