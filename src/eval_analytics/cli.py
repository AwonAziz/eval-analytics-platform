"""Command line interface.

    eap build     regenerate artifacts, rebuild the warehouse, run the gate
    eap migrate   apply pending schema migrations
    eap check     run the quality gate against the existing warehouse
    eap report    print one analytical answer
    eap serve     start the SQL API
    eap summary   print row counts and provenance
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import SETTINGS
from .serialize import jsonable
from .warehouse.migrate import current_version


def _resolve_db(args: argparse.Namespace) -> Path:
    """Resolve the warehouse path.

    ``--db`` is accepted on either side of the subcommand, because argparse
    only parses global options that appear *before* the subcommand and it is
    far more natural to type ``eap build --db x`` than ``eap --db x build``.
    The two positions write to different destinations so neither clobbers the
    other; the post-subcommand value wins when both are given.
    """
    chosen = getattr(args, "db_after", None) or getattr(args, "db", None)
    return Path(chosen or SETTINGS.duckdb_path)


def _print_json(payload: Any) -> None:
    """Emit JSON with real numbers, not stringified numpy scalars."""
    print(json.dumps(jsonable(payload), indent=2))


def cmd_build(args: argparse.Namespace) -> int:
    from .ingest.pipeline import run_pipeline
    from .quality import run_gate

    db_path = _resolve_db(args)
    report = run_pipeline(
        db_path=db_path,
        regenerate_artifacts=not args.no_regenerate,
        fail_fast=args.fail_fast,
        verbose=not args.quiet,
    )
    if report.status != "success":
        print("\nbuild failed", file=sys.stderr)
        return 1

    gate = run_gate(db_path, max_rejections=args.max_rejections)
    print()
    print(gate.summary())
    return 0 if gate.passed else 1


def cmd_migrate(args: argparse.Namespace) -> int:
    from .ingest.pipeline import migrate_warehouse

    db_path = _resolve_db(args)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    migrate_warehouse(db_path, verbose=True)
    print(f"schema version: {current_version(db_path)}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from .quality import run_gate

    db_path = _resolve_db(args)
    gate = run_gate(db_path, max_rejections=args.max_rejections)
    if args.json:
        _print_json(gate.to_dict())
    else:
        print(gate.summary())
    return 0 if gate.passed else 1


def cmd_summary(args: argparse.Namespace) -> int:
    from .analytics.queries import data_quality_summary

    payload = data_quality_summary(_resolve_db(args))
    if args.json:
        _print_json(payload)
        return 0
    print("row counts")
    for row in payload["counts"]:
        print(f"  {row['object']:<26} {row['rows']:>8}")
    print("\nprovenance")
    for row in payload["provenance"]:
        print(
            f"  {row['data_origin']:<10} {row['predictions']:>6} predictions "
            f"({row['unlabelled']} unlabelled)  {row['first_date']} .. {row['last_date']}"
        )
    latest = payload["latest_ingestion_run"]
    if latest:
        print(
            f"\nlast ingestion run {latest['pipeline_run_id']}: {latest['status']} "
            f"(landed {latest['rows_landed']}, rejected {latest['rows_rejected']})"
        )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .analytics import queries as Q

    db_path = _resolve_db(args)
    name = args.name
    runner = {
        "lora-vs-full": lambda: Q.lora_vs_full_finetune(db_path, n_boot=args.n_boot),
        "calibration": lambda: Q.calibration_by_arm(db_path),
        "quantisation": lambda: Q.quantisation_comparison(db_path),
        "drift": lambda: Q.drift_trend(db_path),
        "length-buckets": lambda: Q.error_by_length_bucket(db_path),
        "slice-losses": lambda: Q.slice_losses(db_path, n_boot=args.n_boot),
    }
    if name not in runner:
        print(f"unknown report {name!r}; choose from: {', '.join(runner)}", file=sys.stderr)
        return 2
    _print_json(runner[name]())
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api.app import create_app

    app = create_app(_resolve_db(args))
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eap", description="Evaluation and telemetry analytics platform"
    )
    parser.add_argument(
        "--db", dest="db", help="path to the DuckDB warehouse (also accepted after the subcommand)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Repeated on every subcommand so `eap build --db x` works as well as
    # `eap --db x build`. `db_after` never collides with the global `db`.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", dest="db_after", help=argparse.SUPPRESS)

    build = sub.add_parser("build", parents=[common], help="rebuild the warehouse from raw artifacts")
    build.add_argument("--no-regenerate", action="store_true", help="use artifacts already on disk")
    build.add_argument("--fail-fast", action="store_true", help="stop on the first invalid row")
    build.add_argument("--max-rejections", type=int, default=0, help="tolerated rejected rows")
    build.add_argument("--quiet", action="store_true")
    build.set_defaults(func=cmd_build)

    migrate = sub.add_parser("migrate", parents=[common], help="apply pending schema migrations")
    migrate.set_defaults(func=cmd_migrate)

    check = sub.add_parser("check", parents=[common], help="run the warehouse quality gate")
    check.add_argument("--max-rejections", type=int, default=0)
    check.add_argument("--json", action="store_true")
    check.set_defaults(func=cmd_check)

    summary = sub.add_parser("summary", parents=[common], help="row counts and provenance")
    summary.add_argument("--json", action="store_true")
    summary.set_defaults(func=cmd_summary)

    report = sub.add_parser("report", parents=[common], help="print one analytical answer as JSON")
    report.add_argument(
        "name",
        choices=[
            "lora-vs-full",
            "calibration",
            "quantisation",
            "drift",
            "length-buckets",
            "slice-losses",
        ],
    )
    report.add_argument("--n-boot", type=int, default=4000)
    report.set_defaults(func=cmd_report)

    serve = sub.add_parser("serve", parents=[common], help="start the SQL API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
