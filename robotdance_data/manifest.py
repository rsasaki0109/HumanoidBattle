"""RD-Manifest の読み込み・検証と license firewall（v0）。

RobotDance のデータ戦略の核: raw video / 制約付き mocap を再配布せず、manifest で
「どの source を・どの権利で・どう再構築するか」を管理する。派生 motion（pose/RD-MIR）の
公開可否は manifest の権利フィールドで判定する（license firewall）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robotdance_core.rd_mir import LicenseState

_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "specs" / "rd-manifest" / "rd-manifest.schema.json"
)


def load_manifest(path: str | Path) -> dict[str, Any]:
    """manifest JSON を読み、rd-manifest v0 schema で検証して返す。"""
    import jsonschema

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(data)
    return data


@dataclass(frozen=True)
class FirewallDecision:
    """license firewall の判定結果。"""

    can_export_derived: bool   # 派生 motion（RD-MIR 等）を公開・配布してよいか
    license_state: LicenseState  # 付与すべき RD-MIR の license_state
    reason: str


def evaluate(manifest: dict[str, Any]) -> FirewallDecision:
    """manifest 1 件の派生 motion 公開可否を判定する。

    ルール（保守的）:
      - derived_motion_allowed が False → 公開不可
      - license_declared が unknown → 公開不可（license_state="unknown"）
      - それ以外は commercial/redistribution/training フラグから license_state を決める
    """
    declared = manifest.get("license_declared", "unknown")
    derived_ok = manifest.get("derived_motion_allowed", False)

    if declared == "unknown":
        return FirewallDecision(False, "unknown", "license_declared=unknown → 派生 motion 非公開")
    if not derived_ok:
        return FirewallDecision(
            False, "unknown", "derived_motion_allowed=false → 派生 motion 非公開"
        )

    if manifest.get("commercial_allowed"):
        state: LicenseState = "commercial_allowed"
    elif manifest.get("redistribution_allowed"):
        state = "redistributable"
    elif manifest.get("training_allowed"):
        state = "trainable"
    else:
        state = "research_only"
    return FirewallDecision(True, state, f"derived_motion_allowed=true, license={declared}")
