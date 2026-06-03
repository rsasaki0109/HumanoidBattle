"""RD-Policy spec / モデル / export（§3/§4.5）の検証。

モデル・schema・純 assembly は依存なしで CI 検証。torch checkpoint export は importorskip。
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from robotdance_core.rd_policy import RdPolicy
from robotdance_models.policy_export import sha256_file, tracking_policy_artifact

_SCHEMA = json.loads(
    (Path(__file__).resolve().parent.parent / "specs" / "rd-policy" / "rd-policy.schema.json")
    .read_text(encoding="utf-8")
)


def _valid(policy: RdPolicy) -> None:
    jsonschema.Draft202012Validator(_SCHEMA).validate(policy.to_dict())


def test_artifact_is_schema_valid() -> None:
    p = tracking_policy_artifact(
        obs_dim=121, action_dim=54, hidden=128, robot="unitree_g1",
        policy_id="rdpolicy-test", weights_ref="policy.pt", weights_sha256="deadbeef",
        training={"framework": "ppo", "iterations": 40}, reference_motion_ids=["dance", "idle"],
    )
    assert p.policy_type == "tracking"
    assert p.observation.dim == 121
    assert p.action.dim == 54
    assert p.action.space == "residual_torque"
    assert p.action.base_actuated is False
    assert p.weights.format == "pytorch"
    assert p.failure_modes  # 既知 failure mode が付く
    assert "note" in (p.safety_limits or {})
    _valid(p)


def test_round_trip(tmp_path: Path) -> None:
    p = tracking_policy_artifact(
        obs_dim=10, action_dim=4, hidden=64, robot="unitree_h1",
        policy_id="rt", weights_ref="w.pt", weights_sha256="ab",
    )
    path = p.save(tmp_path / "p.rdpolicy.json")
    p2 = RdPolicy.load(path)
    assert p2.policy_id == "rt"
    assert p2.robot_name == "unitree_h1"
    assert p2.action.dim == 4


def test_schema_rejects_missing_required() -> None:
    bad = {"rd_policy_version": "0", "policy_id": "x", "policy_type": "tracking"}  # 不足
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(_SCHEMA).validate(bad)


def test_schema_rejects_bad_policy_type() -> None:
    p = tracking_policy_artifact(obs_dim=10, action_dim=4, hidden=64, robot="r",
                                 policy_id="x", weights_ref="w.pt")
    d = p.to_dict()
    d["policy_type"] = "invalid"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(_SCHEMA).validate(d)


def test_sha256_file(tmp_path: Path) -> None:
    f = tmp_path / "w.bin"
    f.write_bytes(b"robotdance")
    digest = sha256_file(f)
    assert len(digest) == 64
    assert digest == sha256_file(f)  # 決定的


def test_export_from_checkpoint(tmp_path: Path) -> None:
    """tracking policy checkpoint → RD-Policy（torch 必要, CI では skip）。"""
    torch = pytest.importorskip("torch")
    from robotdance_models.policy_export import export_tracking_policy
    from robotdance_models.tracking_policy import ActorCritic

    ac = ActorCritic(121, 54, hidden=128)
    ckpt = tmp_path / "tp.pt"
    torch.save({"state_dict": ac.state_dict(), "obs_dim": 121, "action_dim": 54, "hidden": 128},
               ckpt)
    out = tmp_path / "tp.rdpolicy.json"
    policy = export_tracking_policy(ckpt, robot="unitree_g1", out_path=out,
                                    training={"framework": "ppo"})
    assert out.exists()
    assert policy.weights.format == "pytorch"
    assert policy.weights.sha256 == sha256_file(ckpt)
    assert policy.observation.dim == 121
    _valid(policy)
