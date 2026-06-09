"""MuJoCo 接触力ベースの fight ヒット採点（sparring 専用）。

幾何距離ではなく ``mj_contactForce`` の法線力が閾値を超え、striker body が
相手の head/chest/spine に接触したフレームをヒットとする。honest scope:
接触は sim 上で発生するが、打撃の「きれいさ」や反動伝播は未モデル。
"""

from __future__ import annotations

import numpy as np

from robotdance_core.skeleton import index_of

# 法線力 [N] の下限。 incidental brushing を除外する経験的閾値。
CONTACT_HIT_MIN_FORCE = 20.0


def _body_id(model, name: str) -> int:
    return model.body(name).id


def _joint_body_ids(model, prefix: str, joint_names: tuple[str, ...]) -> frozenset[int]:
    return frozenset(_body_id(model, f"{prefix}body_{index_of(j)}") for j in joint_names)


class ContactHitDetector:
    """arena 上の a_/b_ プレフィックス付き 2 体向け接触ヒット検出。"""

    def __init__(
        self,
        model,
        *,
        prefix_a: str,
        prefix_b: str,
        striker_joints: tuple[str, ...],
        target_joints: tuple[str, ...],
        body_target_joints: tuple[str, ...],
        min_normal_force: float = CONTACT_HIT_MIN_FORCE,
    ) -> None:
        self._model = model
        self._min_force = float(min_normal_force)
        self._striker_a = _joint_body_ids(model, prefix_a, striker_joints)
        self._striker_b = _joint_body_ids(model, prefix_b, striker_joints)
        self._target_a = _joint_body_ids(model, prefix_a, target_joints)
        self._target_b = _joint_body_ids(model, prefix_b, target_joints)
        self._body_b = _joint_body_ids(model, prefix_b, body_target_joints)
        self._body_a = _joint_body_ids(model, prefix_a, body_target_joints)

    @staticmethod
    def _is_ground(body_id: int) -> bool:
        return body_id == 0  # worldbody（arena の ground plane）

    def substep_hits(self, data) -> tuple[bool, bool, bool, bool]:
        """1 mj_step 後の接触を走査。返り値: p1_hit, p2_hit, p1_body, p2_body。"""
        import mujoco

        p1_hit = p2_hit = False
        p1_body = p2_body = False
        forces = np.zeros(6, dtype=np.float64)

        for i in range(data.ncon):
            con = data.contact[i]
            b1 = self._model.geom_bodyid[con.geom1]
            b2 = self._model.geom_bodyid[con.geom2]
            if ContactHitDetector._is_ground(b1) or ContactHitDetector._is_ground(b2):
                continue

            mujoco.mj_contactForce(self._model, data, i, forces)
            if forces[0] < self._min_force:
                continue

            # A strikes B
            if (b1 in self._striker_a and b2 in self._target_b) or (
                b2 in self._striker_a and b1 in self._target_b
            ):
                p1_hit = True
                struck = b2 if b1 in self._striker_a else b1
                if struck in self._body_b:
                    p1_body = True

            # B strikes A
            if (b1 in self._striker_b and b2 in self._target_a) or (
                b2 in self._striker_b and b1 in self._target_a
            ):
                p2_hit = True
                struck = b2 if b1 in self._striker_b else b1
                if struck in self._body_a:
                    p2_body = True

        return p1_hit, p2_hit, p1_body, p2_body


def target_joint_names() -> tuple[str, ...]:
    return ("head", "chest", "spine")


def body_target_joint_names() -> tuple[str, ...]:
    return ("chest", "spine")


__all__ = [
    "CONTACT_HIT_MIN_FORCE",
    "ContactHitDetector",
    "body_target_joint_names",
    "target_joint_names",
]
