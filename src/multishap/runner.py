from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence


LEGACY_SCRIPT_CANDIDATES = (
    "multishap_vqa.py",
    "scripts/multishap_vqa.py",
    "src/multishap_vqa.py",
)


def _find_legacy_script() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    for relpath in LEGACY_SCRIPT_CANDIDATES:
        candidate = repo_root / relpath
        if candidate.exists():
            return candidate

    searched = ", ".join(LEGACY_SCRIPT_CANDIDATES)
    raise FileNotFoundError(
        "Could not locate an executable legacy analysis script. "
        f"Searched: {searched}. "
        "Please add the analysis entry script to the repository or refactor "
        "its main logic into the multishap package."
    )


def run_legacy_cli(argv: Sequence[str] | None = None) -> int:
    """Execute the legacy repository CLI through the formal package entry point.

    Parameters
    ----------
    argv:
        Optional command-line arguments excluding the executable name.

    Returns
    -------
    int
        The legacy script exit code.
    """

    script_path = _find_legacy_script()
    args = [sys.executable, str(script_path)]
    if argv is not None:
        args.extend(list(argv))

    completed = subprocess.run(args, check=False)
    return int(completed.returncode)
