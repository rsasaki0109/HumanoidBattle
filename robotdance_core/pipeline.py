"""End-to-end pipeline orchestration（ショーケース, v0）。

RobotDance の主要スタックを 1 本に繋ぐ:

    (data/synth) → RD-MIR → retarget → sim_certificate → [tracking policy + export] → model cards

各成果物（RD-MIR / RD-Motion / RD-Policy）と説明責任カード（Model/Policy Card）を出力ディレクトリに
書き出し、サマリ dict を返す。重い段階は依存が無ければ graceful にスキップする:
  - sim_certificate: mujoco（`[sim]`）が無ければ skip
  - tracking policy + export: torch（`[learn]`）+ mujoco が無ければ skip

⚠️ v0: 近似プロキシ・近似質量で実機保証ではない。生成/学習物は retarget→sim_certificate→safety guard
の安全パイプラインを通すこと。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .rd_mir import RdMir


def _has(mod: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def run_pipeline(
    out_dir: str | Path,
    *,
    mir: Optional[RdMir] = None,
    caption: Optional[str] = None,
    robot: str = "unitree_g1",
    do_sim: bool = True,
    train_policy: bool = False,
    iterations: int = 20,
) -> dict[str, Any]:
    """end-to-end pipeline を実行し、サマリ dict を返す。"""
    from robotdance_core.model_card import build_motion_card, render_markdown
    from robotdance_retarget.kinematic import retarget
    from robotdance_unitree import get_morphology

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stages: list[dict[str, Any]] = []
    artifacts: dict[str, str] = {}

    # 1. RD-MIR（入口）。
    if mir is None:
        from robotdance_core.synthetic import generate_dance

        mir = generate_dance(duration=2.0, beats_per_second=1.0)
        mir.semantics = {"action_label": caption or "synthetic dance"}
    mir_path = out_dir / "motion.rdmir.json"
    mir.save(mir_path)
    artifacts["rd_mir"] = str(mir_path)
    stages.append({"stage": "rd_mir", "ok": True,
                   "detail": f"{mir.num_frames} frames, caption='{(mir.semantics or {}).get('action_label')}'"})

    # 2. retarget。
    morph = get_morphology(robot)
    motion = retarget(mir, morph)
    stages.append({"stage": "retarget", "ok": True,
                   "detail": f"→ {robot} ({len(motion.keypoints_3d or [])} frames)"})

    # 3. sim_certificate（任意 / mujoco）。
    verdict = None
    if do_sim and _has("mujoco"):
        from robotdance_sim.backend import certify

        certify(motion, morph, backend="mujoco")
        verdict = (motion.sim_certificate or {}).get("verdict")
        stages.append({"stage": "sim_certificate", "ok": True, "detail": f"backend=mujoco verdict={verdict}"})
    else:
        stages.append({"stage": "sim_certificate", "ok": False,
                       "detail": "skip（mujoco 未インストール）" if do_sim else "skip（--no-sim）"})

    motion_path = out_dir / "motion.rdmotion.json"
    motion.save(motion_path)
    artifacts["rd_motion"] = str(motion_path)

    # 4. Motion Card（説明責任）。
    card = build_motion_card(motion, mir=mir)
    card_path = out_dir / "MOTION_CARD.md"
    card_path.write_text(render_markdown(card), encoding="utf-8")
    artifacts["motion_card"] = str(card_path)
    stages.append({"stage": "motion_card", "ok": True,
                   "detail": f"license={card['license']['state']} failure_modes={len(card['failure_modes'])}"})

    # 5. tracking policy + export（任意 / torch + mujoco）。
    policy_summary: dict[str, Any] = {}
    if train_policy and _has("torch") and _has("mujoco"):
        policy_summary = _train_and_export_policy(mir, morph, robot, iterations, out_dir, artifacts, stages)
    elif train_policy:
        stages.append({"stage": "tracking_policy", "ok": False,
                       "detail": "skip（torch / mujoco 未インストール）"})

    return {
        "robot": robot,
        "out_dir": str(out_dir),
        "verdict": verdict,
        "stages": stages,
        "artifacts": artifacts,
        "policy": policy_summary,
    }


def _train_and_export_policy(mir, morph, robot, iterations, out_dir, artifacts, stages):  # noqa: ANN001
    """tracking policy を学習 → RD-Policy + ONNX + Policy Card を出力する。"""
    from robotdance_core.model_card import build_policy_card, render_markdown
    from robotdance_models.policy_export import export_tracking_policy
    from robotdance_models.tracking_policy import train_tracking_policy
    from robotdance_retarget.kinematic import retarget

    ref = retarget(mir, morph)
    ckpt = out_dir / "tracking_policy.pt"
    policy, info = train_tracking_policy(ref, morph, iterations=iterations, out_path=ckpt)
    roll = policy.rollout()[1]
    artifacts["policy_weights"] = str(ckpt)

    onnx = out_dir / "tracking_policy.onnx"
    rdpol = export_tracking_policy(
        ckpt, robot=robot, onnx_path=onnx, out_path=out_dir / "policy.rdpolicy.json",
        training={"framework": "ppo", "iterations": iterations,
                  "survival_ratio": roll["survival_ratio"], "device": info["device"]},
        reference_motion_ids=[mir.motion_id],
    )
    artifacts["rd_policy"] = str(out_dir / "policy.rdpolicy.json")
    artifacts["policy_onnx"] = str(onnx)
    stages.append({"stage": "tracking_policy", "ok": True,
                   "detail": f"survival={roll['survival_ratio']:.0%} rmse={roll['mean_pose_rmse']:.3f}"})

    pcard = out_dir / "POLICY_CARD.md"
    pcard.write_text(render_markdown(build_policy_card(rdpol)), encoding="utf-8")
    artifacts["policy_card"] = str(pcard)
    stages.append({"stage": "policy_export", "ok": True,
                   "detail": f"RD-Policy(+ONNX) obs={rdpol.observation.dim} act={rdpol.action.dim}"})
    return {"survival_ratio": roll["survival_ratio"], "mean_pose_rmse": roll["mean_pose_rmse"],
            "onnx": str(onnx)}
