"""Sparring survival benchmark の検証。"""

from __future__ import annotations

import pytest

from robotdance_benchmarks.sparring_survival import (
    evaluate_sparring_survival,
    render_sparring_survival_markdown,
    run_sparring_survival_benchmark,
    write_sparring_survival_csv,
)


@pytest.fixture(scope="module")
def mujoco():
    return pytest.importorskip("mujoco")


def test_evaluate_sparring_boxing_runs(mujoco) -> None:
    row = evaluate_sparring_survival(
        "unitree_g1", "unitree_h1", "boxing", duration=2.5,
    )
    assert row.p1 == "unitree_g1"
    assert row.p2 == "unitree_h1"
    assert 0.0 <= row.p1_survival <= 1.0
    assert 0.0 <= row.p2_survival <= 1.0
    assert row.min_survival == min(row.p1_survival, row.p2_survival)
    assert row.p1_hits >= 0
    assert row.winner in ("unitree_g1", "unitree_h1")


def test_benchmark_report_structure(mujoco, tmp_path) -> None:
    report = run_sparring_survival_benchmark(
        robots=["unitree_g1", "unitree_h2"],
        opponent="unitree_h1",
        styles=["boxing", "kick"],
        duration=2.5,
        compare_refine=True,
    )
    assert report["opponent"] == "unitree_h1"
    # g1 + h2 vs h1 × 2 styles × raw/refine = 8
    assert len(report["rows"]) == 8
    md = render_sparring_survival_markdown(report)
    assert "Sparring Survival Benchmark" in md
    assert "unitree_g1" in md
    csv_path = write_sparring_survival_csv(report, tmp_path / "sparring.csv")
    assert csv_path.is_file()
    text = csv_path.read_text(encoding="utf-8")
    assert "min_survival" in text
    assert "p1_survival" in text


def test_benchmark_skips_opponent_from_p1_list(mujoco) -> None:
    report = run_sparring_survival_benchmark(
        robots=["unitree_h1"],
        opponent="unitree_h1",
        styles=["boxing"],
        duration=2.0,
        compare_refine=False,
    )
    assert len(report["rows"]) == 0


def test_benchmark_retarget_backend_compare(mujoco) -> None:
    from robotdance_retarget.gmr_backend import gmr_available

    if not gmr_available():
        pytest.skip("GMR 未導入")
    report = run_sparring_survival_benchmark(
        robots=["unitree_g1"],
        opponent="unitree_h1",
        styles=["kick"],
        duration=2.5,
        compare_refine=False,
        retarget_backends=["kinematic", "gmr"],
    )
    assert len(report["rows"]) == 2
    backends = {r["retarget_backend"] for r in report["rows"]}
    assert backends == {"kinematic", "gmr"}
    md = render_sparring_survival_markdown(report)
    assert "Retarget backend comparison" in md
