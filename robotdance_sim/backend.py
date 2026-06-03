"""Sim backend 抽象 + registry（§4.3, v0）。

sim_certificate（物理ベースの feasibility 検証）は v0 で MuJoCo 一択だったが、Isaac Lab /
Genesis 等の GPU 並列 sim へ差し替えたいことがある。本モジュールは **backend 抽象**（`SimBackend`）
と **registry** を提供し、`certify(motion, morphology, backend=...)` で backend を選べるようにする。

各 backend は **同じ certificate dict 契約**（`passed` / `verdict` / `backend` / `metrics` /
`reasons`）を満たせばよい。MuJoCo を参照実装として登録し、Isaac Lab は contract のみの scaffold
（未インストールなら明示エラー）として登録する。

⚠️ v0: Isaac Lab 本体（NVIDIA Omniverse 依存・大容量）は **同梱・実行しない**（license/容量 safe）。
本モジュールは backend を pluggable にする土台で、Isaac Lab の実装そのものは利用者環境で行う。
"""

from __future__ import annotations

import importlib.util
from typing import Any, Callable, Protocol, runtime_checkable

from robotdance_core.rd_motion import RdMotion
from robotdance_retarget.embodiment import RobotMorphology

# certificate dict が満たすべき必須キー（backend 契約）。
_REQUIRED_KEYS = ("backend", "passed", "verdict", "metrics", "reasons")


def _has_module(name: str) -> bool:
    """module が import 可能か（dotted name で親が無い場合も安全に False を返す）。"""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


@runtime_checkable
class SimBackend(Protocol):
    """物理 sim backend の契約。"""

    name: str

    def available(self) -> bool:
        """この環境で backend が利用可能か（依存がインストール済みか）。"""
        ...

    def simulate_certificate(
        self, motion: RdMotion, morphology: RobotMorphology, **kwargs: Any
    ) -> dict[str, Any]:
        """RD-Motion を物理検証し sim_certificate dict を返す（_REQUIRED_KEYS を満たすこと）。"""
        ...


class MujocoBackend:
    """MuJoCo 参照実装（`robotdance_sim.mujoco_backend` をラップ）。"""

    name = "mujoco"

    def available(self) -> bool:
        return _has_module("mujoco")

    def simulate_certificate(
        self, motion: RdMotion, morphology: RobotMorphology, **kwargs: Any
    ) -> dict[str, Any]:
        from .mujoco_backend import simulate_certificate as _sc

        return _sc(motion, morphology, **kwargs)


class IsaacLabBackend:
    """Isaac Lab backend の scaffold（contract のみ, v0）。

    Isaac Lab（`isaaclab` / `omni.isaac.lab`）が無い環境では `available()` が False になり、
    `simulate_certificate` は導入手順を示して明示エラーを投げる。実装は利用者環境で contract
    （_REQUIRED_KEYS の certificate dict を返す）に従って行う。
    """

    name = "isaaclab"

    def available(self) -> bool:
        return any(_has_module(m)
                   for m in ("isaaclab", "omni.isaac.lab"))

    def simulate_certificate(
        self, motion: RdMotion, morphology: RobotMorphology, **kwargs: Any
    ) -> dict[str, Any]:
        if not self.available():
            raise RuntimeError(
                "Isaac Lab backend は未インストール。NVIDIA Omniverse + Isaac Lab を導入し "
                "（https://isaac-sim.github.io/IsaacLab/）、本 backend を契約に従って実装すること。"
                " v0 は backend 抽象のみ提供する（license/容量のため本体は同梱しない）。"
            )
        raise NotImplementedError(
            "Isaac Lab backend は scaffold（contract のみ）。simulate_certificate の実装は今後。"
        )


_REGISTRY: dict[str, Callable[[], SimBackend]] = {}


def register_backend(name: str, factory: Callable[[], SimBackend]) -> None:
    """backend factory を登録する（新 backend はここに追加）。"""
    _REGISTRY[name] = factory


def backend_names() -> list[str]:
    """登録済み backend 名の一覧。"""
    return sorted(_REGISTRY)


def get_backend(name: str = "mujoco") -> SimBackend:
    """名前から backend インスタンスを得る。"""
    if name not in _REGISTRY:
        raise ValueError(f"未知の sim backend: {name!r}（利用可能: {backend_names()}）")
    return _REGISTRY[name]()


def backend_status() -> list[dict[str, Any]]:
    """各 backend の {name, available} 一覧。"""
    return [{"name": n, "available": get_backend(n).available()} for n in backend_names()]


def simulate_certificate(
    motion: RdMotion, morphology: RobotMorphology, *, backend: str = "mujoco", **kwargs: Any
) -> dict[str, Any]:
    """選択 backend で sim_certificate を計算する（契約キーを検証）。"""
    cert = get_backend(backend).simulate_certificate(motion, morphology, **kwargs)
    missing = [k for k in _REQUIRED_KEYS if k not in cert]
    if missing:
        raise ValueError(f"backend {backend!r} の certificate に必須キーが不足: {missing}")
    return cert


def certify(
    motion: RdMotion, morphology: RobotMorphology, *, backend: str = "mujoco", **kwargs: Any
) -> RdMotion:
    """選択 backend で sim_certificate を計算して motion に格納し、同じ motion を返す。"""
    motion.sim_certificate = simulate_certificate(motion, morphology, backend=backend, **kwargs)
    return motion


# 既定 backend を登録。
register_backend("mujoco", MujocoBackend)
register_backend("isaaclab", IsaacLabBackend)
