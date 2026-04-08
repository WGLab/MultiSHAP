from __future__ import annotations

import argparse
import sys

from .runner import run_legacy_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multishap",
        description=(
            "MultiSHAP command-line interface. This package currently provides "
            "a formal installation surface and dispatches to the repository's "
            "legacy analysis script."
        ),
        add_help=True,
    )
    parser.add_argument(
        "legacy_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the legacy analysis script.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    forwarded = list(args.legacy_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    raise SystemExit(run_legacy_cli(forwarded))


if __name__ == "__main__":
    main()
