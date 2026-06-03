"""Tracking policy → RD-Policy artifact の export（§3/§4.5, v0）。

学習済み tracking policy（`tracking_policy.py` の checkpoint）を、配布可能な **RD-Policy**
artifact（.rdpolicy, `robotdance_core.rd_policy`）にまとめる。policy の I/O 規約・アーキテクチャ・
学習来歴・**安全制約**・weights 参照を 1 つの spec 適合 JSON にする。weights 本体は埋め込まず
参照する（license/容量 safe）。任意で **ONNX** にも書き出し、実機ランタイムへの橋渡しにする。

torch は ONNX export / checkpoint 読込時のみ必要（assembly 自体は依存なしで CI 検証可能）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

from robotdance_core.model_card import _collect_failures
from robotdance_core.rd_policy import Action, Observation, RdPolicy, Weights

# TrackingEnv._make_obs の観測レイアウト（説明用）。
_OBS_COMPONENTS = [
    "base_height(1)", "base_quat(4)", "upright(1)", "base_vel(6)",
    "joint_vel(n_act)", "pose_error_to_next_ref(n_act)", "phase(1)",
]


def sha256_file(path: str | Path) -> str:
    """ファイルの SHA-256 を返す（weights 整合性チェック用）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def tracking_policy_artifact(
    *,
    obs_dim: int,
    action_dim: int,
    hidden: int,
    robot: str,
    policy_id: str,
    weights_format: str = "pytorch",
    weights_ref: Optional[str] = None,
    weights_sha256: Optional[str] = None,
    residual_scale: float = 6.0,
    kp: float = 60.0,
    kd: float = 3.0,
    control_mode: str = "torque",
    fps: float = 30.0,
    training: Optional[dict[str, Any]] = None,
    safety_limits: Optional[dict[str, Any]] = None,
    reference_motion_ids: Optional[list[str]] = None,
    license_state: str = "research_only",
    runtime_adapter: str = "unitree_sdk2",
) -> RdPolicy:
    """学習情報から RD-Policy を組み立てる（torch 非依存の純粋な assembly）。"""
    failures = _collect_failures(["rl_tracking_policy", "policy"])
    return RdPolicy(
        policy_id=policy_id,
        policy_type="tracking",
        robot_name=robot,
        license_state=license_state,  # type: ignore[arg-type]
        runtime_adapter=runtime_adapter,
        observation=Observation(dim=obs_dim, components=list(_OBS_COMPONENTS)),
        action=Action(dim=action_dim, space="residual_torque", scale=residual_scale,
                      base_actuated=False),
        control={"control_mode": control_mode, "dt": round(1.0 / fps, 5),
                 "kp": kp, "kd": kd, "residual_scale": residual_scale,
                 "note": "関節空間 PD（参照 qpos アンカー）+ 方策の残差トルク。base 非駆動。"},
        architecture={"type": "mlp_actor_critic", "hidden": hidden, "layers": 2,
                      "policy_head": "diagonal_gaussian"},
        weights=Weights(format=weights_format, ref=weights_ref,  # type: ignore[arg-type]
                        sha256=weights_sha256),
        training=training or {},
        safety_limits=safety_limits or {
            "note": "実機コマンド直前に joint-space safety guard（位置/速度/加速度/トルク）を通すこと。",
        },
        failure_modes=failures,
        provenance={"reference_motion_ids": reference_motion_ids or [],
                    "trainer": "robotdance_models.tracking_policy.ppo"},
    )


def export_tracking_policy(
    checkpoint_path: str | Path,
    *,
    robot: str,
    policy_id: Optional[str] = None,
    training: Optional[dict[str, Any]] = None,
    safety_limits: Optional[dict[str, Any]] = None,
    reference_motion_ids: Optional[list[str]] = None,
    onnx_path: Optional[str | Path] = None,
    out_path: Optional[str | Path] = None,
    license_state: str = "research_only",
) -> RdPolicy:
    """tracking policy checkpoint(.pt) → RD-Policy。任意で ONNX も書き出す。"""
    import torch

    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    obs_dim = int(ckpt["obs_dim"])
    action_dim = int(ckpt["action_dim"])
    hidden = int(ckpt.get("hidden", 128))

    weights_format = "pytorch"
    weights_ref: str = checkpoint_path.name
    if onnx_path is not None:
        _export_onnx(ckpt, obs_dim, action_dim, hidden, onnx_path)
        weights_format = "onnx"
        weights_ref = Path(onnx_path).name

    weights_file = onnx_path if onnx_path is not None else checkpoint_path
    policy = tracking_policy_artifact(
        obs_dim=obs_dim, action_dim=action_dim, hidden=hidden, robot=robot,
        policy_id=policy_id or f"rdpolicy-tracking-{robot}",
        weights_format=weights_format, weights_ref=weights_ref,
        weights_sha256=sha256_file(weights_file),
        training=training, safety_limits=safety_limits,
        reference_motion_ids=reference_motion_ids, license_state=license_state,
    )
    if out_path is not None:
        policy.save(out_path)
    return policy


def _export_onnx(ckpt: dict, obs_dim: int, action_dim: int, hidden: int,
                 onnx_path: str | Path) -> None:
    """ActorCritic の平均行動（決定論方策）を ONNX に書き出す。"""
    import torch
    from torch import nn

    from .tracking_policy import ActorCritic

    ac = ActorCritic(obs_dim, action_dim, hidden=hidden)
    ac.load_state_dict(ckpt["state_dict"])
    ac.eval()

    class MeanActor(nn.Module):
        def __init__(self, body: ActorCritic) -> None:
            super().__init__()
            self.body = body

        def forward(self, obs: "torch.Tensor") -> "torch.Tensor":
            return self.body(obs)[0]  # mu（決定論方策）

    dummy = torch.zeros(1, obs_dim)
    torch.onnx.export(
        MeanActor(ac), dummy, str(onnx_path),
        input_names=["observation"], output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
    )
