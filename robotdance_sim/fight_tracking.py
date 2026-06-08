"""Fight motion を RL tracking / assisted 向けに retarget するヘルパー。"""

from __future__ import annotations

from robotdance_core.rd_mir import RdMir
from robotdance_core.rd_motion import RdMotion
from robotdance_motion.fight_refinement import refine_for_fight
from robotdance_retarget.dispatch import retarget_with_backend
from robotdance_retarget.embodiment import RobotMorphology
from robotdance_sim.arena import motion_for_style
from robotdance_unitree import get_morphology

# multi-motion tracking 用の合成 fight スイート（karate/kathak は長いので除外）。
FIGHT_TRACKING_SUITE = ("boxing", "hook", "kick", "dodge")


def fight_motion_mir(
    style: str,
    *,
    depth_refine: bool = False,
    duration: float = 3.0,
) -> RdMir:
    """fight style の RD-MIR を返す（任意で depth-refine）。"""
    mir = motion_for_style(style, duration=duration)
    if depth_refine:
        mir = refine_for_fight(mir)
    return mir


def fight_tracking_reference(
    robot: str,
    style: str,
    *,
    depth_refine: bool = False,
    duration: float = 3.0,
    morphology: RobotMorphology | None = None,
    retarget_backend: str = "kinematic",
) -> RdMotion:
    """fight motion を指定ロボットへ retarget し tracking 参照を返す。"""
    from robotdance_retarget.dispatch import check_retarget_backend_for_robots

    morph = morphology or get_morphology(robot)
    check_retarget_backend_for_robots([robot], retarget_backend)
    return retarget_with_backend(
        fight_motion_mir(style, depth_refine=depth_refine, duration=duration),
        morph, retarget_backend,
    )


def fight_tracking_suite(
    morphology: RobotMorphology,
    *,
    depth_refine: bool = False,
    duration: float = 3.0,
    retarget_backend: str = "kinematic",
) -> list[tuple[str, RdMotion]]:
    """合成 fight 4 技の tracking 参照スイート。"""
    return [
        (
            style,
            retarget_with_backend(
                fight_motion_mir(style, depth_refine=depth_refine, duration=duration),
                morphology, retarget_backend,
            ),
        )
        for style in FIGHT_TRACKING_SUITE
    ]


__all__ = [
    "FIGHT_TRACKING_SUITE",
    "fight_motion_mir",
    "fight_tracking_reference",
    "fight_tracking_suite",
]
