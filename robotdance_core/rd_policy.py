"""RD-Policy の Python データモデル（v0）。

specs/rd-policy/rd-policy.schema.json に対応。学習済み motion policy の配布 artifact
（.rdpolicy）。policy の I/O 規約・アーキテクチャ・学習来歴・安全制約・**weights 参照**を保持する。
weights 本体は埋め込まず参照する（license/容量 safe）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

LicenseState = Literal[
    "redistributable", "trainable", "commercial_allowed", "research_only", "unknown"
]
PolicyType = Literal["tracking", "skill"]
WeightsFormat = Literal["pytorch", "onnx", "none"]


class Observation(BaseModel):
    model_config = ConfigDict(extra="allow")
    dim: int = Field(ge=1)
    components: list[str] = Field(default_factory=list)


class Action(BaseModel):
    model_config = ConfigDict(extra="allow")
    dim: int = Field(ge=1)
    space: str
    scale: Optional[float] = None
    base_actuated: bool = False


class Weights(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: WeightsFormat
    ref: Optional[str] = None
    sha256: Optional[str] = None


class RdPolicy(BaseModel):
    """RD-Policy v0。"""

    model_config = ConfigDict(extra="forbid")

    rd_policy_version: Literal["0"] = "0"
    policy_id: str
    policy_type: PolicyType
    robot_name: str
    observation: Observation
    action: Action
    weights: Weights

    license_state: LicenseState = "unknown"
    runtime_adapter: Optional[str] = None
    control: Optional[dict[str, Any]] = None
    architecture: Optional[dict[str, Any]] = None
    training: Optional[dict[str, Any]] = None
    safety_limits: Optional[dict[str, Any]] = None
    failure_modes: list[dict[str, Any]] = Field(default_factory=list)
    provenance: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)

    def save(self, path: str | Path, *, indent: int = 2) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=indent), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RdPolicy":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
