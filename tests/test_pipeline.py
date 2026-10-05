"""Quality gate, pipeline determinism, and the CLI."""

from __future__ import annotations

import json

import pytest

from eval_analytics.cli import main
from eval_analytics.config import Settings
from eval_analytics.ingest.pipeline import run_pipeline
from eval_analytics.quality import (
    check_freshness,
    check_last_ingestion,
    check_referential_integrity,
    check_rejections,
    check_required_coverage,
    run_gate,
)
from eval_analytics.warehouse import query
from eval_analytics.warehouse.engine import reset_engines


class TestGateOnACleanBuild:
    def test_every_check_passes(self, db_path: str):
        report = run_gate(db_path)
        assert report.passed, report.summary()
        assert report.failures == []

    def test_all_five_checks_ran(self, db_path: str):
        names = {c.name for c in run_gate(db_path).checks}
        assert names == {
            "last_ingestion_run",
            "validation_rejections",
            "referential_integrity",
            "mart_freshness",
            "analytical_coverage",
        }

    def test_no_rejections_recorded(self, db_path: str):
        result = check_rejections(db_path, max_rejections=0)
        assert result.passed
        assert result.evidence["n_rejections"] == 0

    def test_rejection_limit_is_honoured(self, db_path: str):
        """A non-zero allowance must still be reported honestly."""
        result = check_rejections(db_path, max_rejections=5)
        assert result.passed
        assert "limit 5" in result.detail

    def test_integrity_and_coverage_pass(self, db_path: str):
        assert check_referential_integrity(db_path).passed
        assert check_required_coverage(db_path).passed
        assert check_last_ingestion(db_path).passed
        assert check_freshness(db_path).passed

    def test_freshness_compares_each_mart_to_its_own_source(self, db_path: str):
        evidence = check_freshness(db_path).evidence["marts"]
        for entry in evidence.values():
            assert "lag_days_vs_source" in entry
            assert entry["source_table"].startswith("stg_")

    def test_summary_renders(self, db_path: str):
        assert "PASS" in run_gate(db_path).summary()


class TestGateDetectsFailures:
    def test_a_rejected_row_fails_the_gate(self, tmp_path, repo_root):
        """A malformed artifact must turn the gate red."""
        from eval_analytics.ingest.artifacts import generate_all

        generated = tmp_path / "raw" / "generated"
        generate_all(generated)
        target = generated / "exp-001-full-finetune" / "eval_report.json"
        report = json.loads(target.read_text(encoding="utf-8"))
        report["macro_f1"] = 4.2
        target.write_text(json.dumps(report), encoding="utf-8")

        settings = Settings(
            duckdb_path=tmp_path / "bad.duckdb",
            raw_dir=tmp_path / "raw",
            generated_dir=generated,
            telemetry_db_path=None,
        )
        built = run_pipeline(
            db_path=settings.duckdb_path,
            settings=settings,
            regenerate_artifacts=False,
            verbose=False,
        )
        assert built.rows_rejected == 1

        result = check_rejections(settings.duckdb_path, max_rejections=0)
        assert not result.passed
        assert "exceeds limit" in result.detail
        assert result.evidence["n_rejections"] == 1
        reset_engines()

    def test_coverage_fails_without_two_arms(self, tmp_path, repo_root):
        """A warehouse with one arm cannot support an arm comparison."""
        from eval_analytics.ingest.artifacts import generate_all

        generated = tmp_path / "raw" / "generated"
        generate_all(generated)
        (generated / "exp-001-full-finetune").rename(generated / "disabled-ft")
        for name in ("eval_report.json", "model_card.json", "test_preds.npy"):
            (generated / "disabled-ft" / name).unlink()
        (generated / "disabled-ft" / "test_labels.npy").unlink()
        (generated / "disabled-ft" / "test_example_ids.npy").unlink()
        (generated / "disabled-ft" / "test_token_lengths.npy").unlink()
        (generated / "disabled-ft" / "classes.json").unlink()

        settings = Settings(
            duckdb_path=tmp_path / "one_arm.duckdb",
            raw_dir=tmp_path / "raw",
            generated_dir=generated,
            telemetry_db_path=None,
        )
        run_pipeline(
            db_path=settings.duckdb_path,
            settings=settings,
            regenerate_artifacts=False,
            verbose=False,
        )
        coverage = check_required_coverage(settings.duckdb_path)
        assert not coverage.passed
        assert "arm comparison impossible" in coverage.detail
        reset_engines()


class TestDeterminism:
    def test_rebuilding_yields_identical_fact_keys(self, settings: Settings, tmp_path):
        """Surrogate keys must be a pure function of the inputs."""
        first = tmp_path / "a.duckdb"
        second = tmp_path / "b.duckdb"

        for target in (first, second):
            run_pipeline(db_path=target, settings=settings, verbose=False)

        keys = "SELECT evaluation_run_key, experiment_id, model_id, slice_id FROM fact_evaluation_run ORDER BY 1"
        assert (
            query(first, keys).to_dict("records")
            == query(second, keys).to_dict("records")
        )

        preds = "SELECT prediction_key, example_id, predicted_label FROM fact_prediction ORDER BY 1"
        assert (
            query(first, preds).to_dict("records")
            == query(second, preds).to_dict("records")
        )
        reset_engines()

    def test_headline_metrics_are_reproducible(self, settings: Settings, tmp_path):
        first, second = tmp_path / "a.duckdb", tmp_path / "b.duckdb"
        for target in (first, second):
            run_pipeline(db_path=target, settings=settings, verbose=False)

        sql = """
            SELECT m.arm, m.quantisation, r.accuracy, r.ece, r.macro_f1
            FROM v_model_quality_headline r JOIN dim_model m USING (model_id)
            ORDER BY m.arm, m.quantisation
        """
        assert query(first, sql).to_dict("records") == query(second, sql).to_dict("records")
        reset_engines()


class TestExtract:
    def test_estimate_tokens_scales_with_word_count(self):
        from eval_analytics.ingest.extract import estimate_tokens

        assert estimate_tokens("") is None
        assert estimate_tokens("   ") is None
        assert estimate_tokens("hello world") == 3
        assert estimate_tokens("a b c d") == 5

    def test_generated_artifacts_land_all_expected_files(self, settings: Settings):
        assert settings.generated_dir.exists()
        run_dirs = sorted(p for p in settings.generated_dir.iterdir() if p.is_dir())
        assert len(run_dirs) == 3
        for run_dir in run_dirs:
            for name in (
                "test_preds.npy", "test_labels.npy", "test_example_ids.npy",
                "test_token_lengths.npy", "classes.json", "eval_report.json",
                "model_card.json",
            ):
                assert (run_dir / name).exists(), f"{run_dir.name} missing {name}"

    def test_serving_report_covers_every_quantisation(self, settings: Settings):
        report = json.loads(
            (settings.generated_dir / "exp-001-lora" / "serving_report.json").read_text(
                encoding="utf-8"
            )
        )
        assert {c["quantisation"] for c in report["configs"]} == {"fp32", "fp16", "int8"}
        assert {r["quantisation"] for r in report["requests"]} == {"fp32", "fp16", "int8"}


class TestCli:
    def test_check_exits_zero_on_a_clean_warehouse(self, db_path: str, capsys):
        assert main(["--db", db_path, "check"]) == 0
        assert "warehouse gate: PASS" in capsys.readouterr().out

    def test_check_json_emits_the_report(self, db_path: str, capsys):
        assert main(["--db", db_path, "check", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["passed"] is True
        assert len(payload["checks"]) == 5

    def test_summary_reports_provenance(self, db_path: str, capsys):
        assert main(["--db", db_path, "summary", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["counts"]
        assert payload["latest_ingestion_run"]["status"] == "success"

    @pytest.mark.parametrize(
        "name", ["lora-vs-full", "calibration", "quantisation", "drift",
                 "length-buckets", "slice-losses"]
    )
    def test_every_report_command_emits_json(self, db_path: str, capsys, name: str):
        assert main(["--db", db_path, "report", name, "--n-boot", "200"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload

    def test_report_json_keeps_numbers_as_numbers(self, db_path: str, capsys):
        """A count must serialise as a number, not a stringified numpy int."""
        assert main(["--db", db_path, "report", "quantisation"]) == 0
        payload = json.loads(capsys.readouterr().out)
        summary = payload["agreement"]["summary"]
        assert isinstance(summary["n_paired"], int)
        assert isinstance(summary["n_agree"], int)
        assert isinstance(summary["agreement_rate"], float)
        assert summary["n_paired"] - summary["n_agree"] == summary["disagreements"]
        for row in payload["latency_by_quantisation"]:
            assert isinstance(row["n_requests"], int)
            assert isinstance(row["p50_latency_ms"], float)

    def test_migrate_is_a_no_op_when_current(self, db_path: str, capsys):
        assert main(["--db", db_path, "migrate"]) == 0
        assert "already up to date" in capsys.readouterr().out

    def test_db_is_accepted_on_either_side_of_the_subcommand(self, db_path: str, capsys):
        """`eap build --db x` is the natural order, not `eap --db x build`."""
        for argv in (
            ["--db", db_path, "check"],
            ["check", "--db", db_path],
        ):
            assert main(argv) == 0, argv
            assert "warehouse gate: PASS" in capsys.readouterr().out, argv

    def test_post_subcommand_db_wins_when_both_are_given(self, db_path: str, capsys):
        assert main(["--db", "nonexistent.duckdb", "check", "--db", db_path]) == 0
        assert "warehouse gate: PASS" in capsys.readouterr().out


class TestTelemetryIntegration:
    """Optional: exercises the real upstream database when it is present."""

    def test_real_rows_are_ingested_and_typed(self, telemetry_settings: Settings | None):
        if telemetry_settings is None:
            pytest.skip("upstream telemetry database not present")

        report = run_pipeline(
            db_path=telemetry_settings.duckdb_path,
            settings=telemetry_settings,
            verbose=False,
        )
        assert report.status == "success"
        assert report.rows_rejected == 0

        provenance = query(
            telemetry_settings.duckdb_path,
            """
            SELECT data_origin, COUNT(*) AS n,
                   SUM(CASE WHEN true_label IS NULL THEN 1 ELSE 0 END) AS unlabelled
            FROM fact_prediction GROUP BY data_origin
            """,
        ).to_dict("records")
        origins = {row["data_origin"] for row in provenance}
        assert origins == {"real", "synthetic"}

        real = next(row for row in provenance if row["data_origin"] == "real")
        assert real["n"] > 1000
        # Unlabeled production traffic must be preserved, not discarded.
        assert real["unlabelled"] > 0

        assert check_referential_integrity(telemetry_settings.duckdb_path).passed
        reset_engines()
