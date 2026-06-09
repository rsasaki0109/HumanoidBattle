"""Sparring survival benchmark — robot × opponent × style の 2 体 PD 接触 survival を定量比較。

depth-refine 前後の p1/p2 survival / min_survival を並べ、sparring 投入前のゲート判定に使う。
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from robotdance_retarget.dispatch import check_retarget_backend_for_robots
from robotdance_retarget.gmr_backend import ROBOT_TO_GMR, gmr_available
from robotdance_sim.arena import run_fight
from robotdance_sim.fight_moves import FIGHT_STYLE_NAMES
from robotdance_unitree import get_morphology

_DEFAULT_ROBOTS = (
    "unitree_g1", "unitree_h1", "unitree_h2",
    "booster_t1", "apptronik_apollo", "fourier_n1",
)
_DEFAULT_OPPONENT = "unitree_h1"


@dataclass
class SparringSurvivalRow:
    p1: str
    p2: str
    style: str
    depth_refine: bool
    p1_survival: float
    p2_survival: float
    min_survival: float
    p1_hits: int
    p2_hits: int
    winner: str
    retarget_backend: str = "kinematic"


def _gmr_supported(robot: str) -> bool:
    return robot in ROBOT_TO_GMR


def evaluate_sparring_survival(
    p1: str,
    p2: str,
    style: str,
    *,
    depth_refine: bool = False,
    duration: float = 3.0,
    separation: float = 0.17,
    retarget_backend: str = "kinematic",
) -> SparringSurvivalRow:
    """1 (p1, p2, style, depth_refine) の PD sparring を評価（render=False）。"""
    check_retarget_backend_for_robots([p1, p2], retarget_backend)
    res = run_fight(
        get_morphology(p1), get_morphology(p2),
        name_a=p1, name_b=p2, duration=duration, separation=separation,
        style=style, render=False, depth_refine=depth_refine,
        retarget_backend=retarget_backend, sparring=True,
    )
    p1s = res.p1_survival if res.p1_survival is not None else 0.0
    p2s = res.p2_survival if res.p2_survival is not None else 0.0
    return SparringSurvivalRow(
        p1=p1,
        p2=p2,
        style=style,
        depth_refine=depth_refine,
        p1_survival=p1s,
        p2_survival=p2s,
        min_survival=round(min(p1s, p2s), 3),
        p1_hits=res.p1_hits,
        p2_hits=res.p2_hits,
        winner=res.winner,
        retarget_backend=retarget_backend,
    )


def run_sparring_survival_benchmark(
    robots: Optional[list[str]] = None,
    opponent: str = _DEFAULT_OPPONENT,
    styles: Optional[list[str]] = None,
    *,
    duration: float = 3.0,
    separation: float = 0.17,
    compare_refine: bool = True,
    retarget_backends: Optional[list[str]] = None,
) -> dict:
    """robots（p1）× opponent（p2）× styles × retarget_backends の sparring survival を回す。"""
    robots = list(robots or _DEFAULT_ROBOTS)
    styles = list(styles or sorted(FIGHT_STYLE_NAMES))
    backends = list(retarget_backends or ["kinematic"])
    skipped_gmr: list[str] = []
    rows: list[SparringSurvivalRow] = []

    for backend in backends:
        if backend == "gmr" and not gmr_available():
            raise RuntimeError(
                "retarget backend 'gmr' が未導入です。"
                " git clone https://github.com/YanjieZe/GMR.git && pip install -e GMR/"
            )
        for p1 in robots:
            if p1 == opponent:
                continue
            if backend == "gmr" and (
                not _gmr_supported(p1) or not _gmr_supported(opponent)
            ):
                skipped_gmr.append(p1)
                continue
            for style in styles:
                rows.append(evaluate_sparring_survival(
                    p1, opponent, style, depth_refine=False,
                    duration=duration, separation=separation,
                    retarget_backend=backend,
                ))
                if compare_refine:
                    rows.append(evaluate_sparring_survival(
                        p1, opponent, style, depth_refine=True,
                        duration=duration, separation=separation,
                        retarget_backend=backend,
                    ))

    pd_rows = list(rows)
    return {
        "robots": robots,
        "opponent": opponent,
        "styles": styles,
        "duration": duration,
        "separation": separation,
        "compare_refine": compare_refine,
        "retarget_backends": backends,
        "skipped_gmr_robots": sorted(set(skipped_gmr)),
        "rows": [asdict(r) for r in pd_rows],
        "rescued": _rescued_pairs(pd_rows),
        "regressed": _regressed_pairs(pd_rows),
        "rescued_by_gmr": _rescued_by_gmr(pd_rows),
        "regressed_by_gmr": _regressed_by_gmr(pd_rows),
    }


def _pair_key(row: SparringSurvivalRow) -> tuple[str, str, str, str]:
    return row.p1, row.p2, row.style, row.retarget_backend


def _by_pair(rows: list[SparringSurvivalRow]) -> dict[tuple[str, str, str, str], dict[bool, SparringSurvivalRow]]:
    out: dict[tuple[str, str, str, str], dict[bool, SparringSurvivalRow]] = {}
    for row in rows:
        out.setdefault(_pair_key(row), {})[row.depth_refine] = row
    return out


def _backend_pair_key(row: SparringSurvivalRow) -> tuple[str, str, str, bool]:
    return row.p1, row.p2, row.style, row.depth_refine


def _by_backend_pair(
    rows: list[SparringSurvivalRow],
) -> dict[tuple[str, str, str, bool], dict[str, SparringSurvivalRow]]:
    out: dict[tuple[str, str, str, bool], dict[str, SparringSurvivalRow]] = {}
    for row in rows:
        out.setdefault(_backend_pair_key(row), {})[row.retarget_backend] = row
    return out


def _rescued_pairs(rows: list[SparringSurvivalRow]) -> list[dict]:
    """raw で min_survival < 1 → refine で改善した (p1, style)。"""
    rescued = []
    for key, pair in _by_pair(rows).items():
        raw = pair.get(False)
        ref = pair.get(True)
        if raw is None or ref is None:
            continue
        if raw.min_survival < 1.0 and ref.min_survival > raw.min_survival:
            rescued.append({
                "p1": key[0],
                "p2": key[1],
                "style": key[2],
                "raw_min_survival": raw.min_survival,
                "ref_min_survival": ref.min_survival,
                "delta_min_survival": round(ref.min_survival - raw.min_survival, 3),
            })
    return sorted(rescued, key=lambda x: (-x["delta_min_survival"], x["p1"], x["style"]))


def _regressed_pairs(rows: list[SparringSurvivalRow]) -> list[dict]:
    regressed = []
    for key, pair in _by_pair(rows).items():
        raw = pair.get(False)
        ref = pair.get(True)
        if raw is None or ref is None:
            continue
        if ref.min_survival < raw.min_survival:
            regressed.append({
                "p1": key[0],
                "p2": key[1],
                "style": key[2],
                "raw_min_survival": raw.min_survival,
                "ref_min_survival": ref.min_survival,
                "delta_min_survival": round(ref.min_survival - raw.min_survival, 3),
            })
    return sorted(regressed, key=lambda x: (x["delta_min_survival"], x["p1"], x["style"]))


def _rescued_by_gmr(rows: list[SparringSurvivalRow]) -> list[dict]:
    rescued = []
    for key, pair in _by_backend_pair(rows).items():
        kin = pair.get("kinematic")
        gmr = pair.get("gmr")
        if kin is None or gmr is None:
            continue
        if kin.min_survival < gmr.min_survival:
            rescued.append({
                "p1": key[0],
                "p2": key[1],
                "style": key[2],
                "depth_refine": key[3],
                "kin_min_survival": kin.min_survival,
                "gmr_min_survival": gmr.min_survival,
                "delta_min_survival": round(gmr.min_survival - kin.min_survival, 3),
            })
    return sorted(rescued, key=lambda x: (-x["delta_min_survival"], x["p1"], x["style"]))


def _regressed_by_gmr(rows: list[SparringSurvivalRow]) -> list[dict]:
    regressed = []
    for key, pair in _by_backend_pair(rows).items():
        kin = pair.get("kinematic")
        gmr = pair.get("gmr")
        if kin is None or gmr is None:
            continue
        if gmr.min_survival < kin.min_survival:
            regressed.append({
                "p1": key[0],
                "p2": key[1],
                "style": key[2],
                "depth_refine": key[3],
                "kin_min_survival": kin.min_survival,
                "gmr_min_survival": gmr.min_survival,
                "delta_min_survival": round(gmr.min_survival - kin.min_survival, 3),
            })
    return sorted(regressed, key=lambda x: (x["delta_min_survival"], x["p1"], x["style"]))


_CSV_COLUMNS = [
    "p1", "p2", "style", "depth_refine", "retarget_backend",
    "p1_survival", "p2_survival", "min_survival",
    "p1_hits", "p2_hits", "winner",
]


def write_sparring_survival_csv(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow({c: row.get(c) for c in _CSV_COLUMNS})
    return path


def render_sparring_survival_markdown(report: dict) -> str:
    lines = [
        "# Sparring Survival Benchmark",
        "",
        "2 体 PD sparring（limb 接触あり）での fight motion 生存率。",
        "min_survival = min(p1_survival, p2_survival)。ヒット採点は幾何のまま。",
        "",
        f"- p1 robots: {', '.join(report['robots'])}",
        f"- opponent (p2): {report['opponent']}",
        f"- styles: {', '.join(report['styles'])}",
        f"- duration: {report['duration']}s（karate/kathak はフィクスチャ長）",
        f"- separation: {report['separation']}m",
        f"- retarget backends: {', '.join(report.get('retarget_backends', ['kinematic']))}",
    ]
    if report.get("skipped_gmr_robots"):
        lines.append(
            f"- GMR skipped p1: {', '.join(report['skipped_gmr_robots'])}（未対応機種）"
        )
    lines.append("")

    backends = report.get("retarget_backends", ["kinematic"])
    if len(backends) > 1 or (len(backends) == 1 and backends[0] != "kinematic"):
        lines += [
            "## Retarget backend comparison",
            "",
            "| p1 | style | refine | kin min | gmr min | Δ min | kin p1/p2 | gmr p1/p2 |",
            "|----|-------|--------|---------|---------|-------|-----------|-----------|",
        ]
        by_backend = {}
        for row in report["rows"]:
            key = (row["p1"], row["style"], row["depth_refine"])
            by_backend.setdefault(key, {})[row.get("retarget_backend", "kinematic")] = row
        for (p1, style, refine), pair in sorted(by_backend.items()):
            kin = pair.get("kinematic")
            gmr = pair.get("gmr")
            if not kin or not gmr:
                continue
            delta = gmr["min_survival"] - kin["min_survival"]
            refine_s = "yes" if refine else "no"
            lines.append(
                f"| {p1} | {style} | {refine_s} | {kin['min_survival']:.3f} | "
                f"{gmr['min_survival']:.3f} | {delta:+.3f} | "
                f"{kin['p1_survival']:.2f}/{kin['p2_survival']:.2f} | "
                f"{gmr['p1_survival']:.2f}/{gmr['p2_survival']:.2f} |"
            )
        lines.append("")
        if report.get("rescued_by_gmr"):
            lines += ["## Rescued by GMR (vs kinematic)", ""]
            for r in report["rescued_by_gmr"]:
                refine = "refine" if r["depth_refine"] else "raw"
                lines.append(
                    f"- **{r['p1']} vs {r['p2']} / {r['style']} ({refine})**: "
                    f"kin {r['kin_min_survival']:.3f} → gmr {r['gmr_min_survival']:.3f} "
                    f"(Δ {r['delta_min_survival']:+.3f})"
                )
            lines.append("")
        if report.get("regressed_by_gmr"):
            lines += ["## Regressed by GMR (honest)", ""]
            for r in report["regressed_by_gmr"]:
                refine = "refine" if r["depth_refine"] else "raw"
                lines.append(
                    f"- **{r['p1']} vs {r['p2']} / {r['style']} ({refine})**: "
                    f"kin {r['kin_min_survival']:.3f} → gmr {r['gmr_min_survival']:.3f} "
                    f"(Δ {r['delta_min_survival']:+.3f})"
                )
            lines.append("")

    if report["compare_refine"]:
        for backend in backends:
            lines += [
                f"## Raw vs depth-refine ({backend})",
                "",
                "| p1 | style | raw min | ref min | Δ min | raw p1/p2 surv | ref p1/p2 surv | hits |",
                "|----|-------|---------|---------|-------|----------------|----------------|------|",
            ]
            by_pair = {}
            for row in report["rows"]:
                if row.get("retarget_backend", "kinematic") != backend:
                    continue
                by_pair.setdefault((row["p1"], row["style"]), {})[row["depth_refine"]] = row
            for (p1, style), pair in sorted(by_pair.items()):
                raw = pair.get(False, {})
                ref = pair.get(True, {})
                if not raw or not ref:
                    continue
                delta = ref["min_survival"] - raw["min_survival"]
                lines.append(
                    f"| {p1} | {style} | {raw['min_survival']:.3f} | "
                    f"{ref['min_survival']:.3f} | {delta:+.3f} | "
                    f"{raw['p1_survival']:.2f}/{raw['p2_survival']:.2f} | "
                    f"{ref['p1_survival']:.2f}/{ref['p2_survival']:.2f} | "
                    f"{raw['p1_hits']}-{raw['p2_hits']} |"
                )
            lines.append("")
        if report["rescued"]:
            lines += ["## Rescued by depth-refine", ""]
            for r in report["rescued"]:
                lines.append(
                    f"- **{r['p1']} vs {r['p2']} / {r['style']}**: "
                    f"min {r['raw_min_survival']:.3f} → {r['ref_min_survival']:.3f} "
                    f"(Δ {r['delta_min_survival']:+.3f})"
                )
            lines.append("")
        if report["regressed"]:
            lines += ["## Regressed (honest)", ""]
            for r in report["regressed"]:
                lines.append(
                    f"- **{r['p1']} vs {r['p2']} / {r['style']}**: "
                    f"min {r['raw_min_survival']:.3f} → {r['ref_min_survival']:.3f} "
                    f"(Δ {r['delta_min_survival']:+.3f})"
                )
            lines.append("")
    else:
        lines += [
            "## Results",
            "",
            "| p1 | style | p1 surv | p2 surv | min | hits | winner |",
            "|----|-------|---------|---------|-----|------|--------|",
        ]
        for row in report["rows"]:
            lines.append(
                f"| {row['p1']} | {row['style']} | {row['p1_survival']:.3f} | "
                f"{row['p2_survival']:.3f} | {row['min_survival']:.3f} | "
                f"{row['p1_hits']}-{row['p2_hits']} | {row['winner']} |"
            )
        lines.append("")

    lines.append(
        "> ⚠️ v0 baseline: 2 体 PD sparring（limb 接触あり）。"
        "ヒット採点は幾何のまま。contact-dynamics scoring は未対応。"
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "SparringSurvivalRow",
    "evaluate_sparring_survival",
    "render_sparring_survival_markdown",
    "run_sparring_survival_benchmark",
    "write_sparring_survival_csv",
]
