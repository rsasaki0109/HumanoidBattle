"""Contact-dynamics fight 採点の検証。"""

from __future__ import annotations

from pathlib import Path

import pytest

from robotdance_core.cli import main
from robotdance_sim.arena import run_fight
from robotdance_unitree import get_morphology


def test_contact_scoring_requires_sparring():
    pytest.importorskip("mujoco")
    with pytest.raises(ValueError, match="sparring"):
        run_fight(
            get_morphology("unitree_g1"), get_morphology("unitree_h1"),
            name_a="unitree_g1", name_b="unitree_h1", duration=2.0, render=False,
            contact_scoring=True,
        )


def test_run_fight_contact_scoring_runs():
    pytest.importorskip("mujoco")
    res = run_fight(
        get_morphology("unitree_g1"), get_morphology("unitree_h1"),
        name_a="unitree_g1", name_b="unitree_h1", duration=2.5, render=False,
        style="boxing", sparring=True, contact_scoring=True,
    )
    assert res.scoring_mode == "contact"
    assert res.sparring is True
    assert res.p1_geom_hits is not None
    assert res.p2_geom_hits is not None
    assert res.p1_hits >= 0
    assert res.winner in ("unitree_g1", "unitree_h1", "DRAW")


def test_fight_hud_contact_prefix():
    import numpy as np

    from robotdance_core.cli import _fight_hud
    from robotdance_sim.arena import FightResult

    frame = np.full((80, 120, 3), 30, np.uint8)
    res = FightResult(
        "unitree_g1", "unitree_h1", 3, 2, "unitree_g1",
        frames=[frame], p1_cum=[3], p2_cum=[2],
        sparring=True, p1_survival=1.0, p2_survival=0.9,
        scoring_mode="contact", p1_geom_hits=5, p2_geom_hits=4,
    )
    out = _fight_hud(res)[0]
    assert out.shape[0] == 80 + 50


def test_cli_contact_scoring_requires_sparring(tmp_path: Path) -> None:
    assert main([
        "demo-fight", "--p1", "unitree_g1", "--p2", "unitree_h1",
        "--contact-scoring", "-o", str(tmp_path / "x.gif"),
    ]) == 1
