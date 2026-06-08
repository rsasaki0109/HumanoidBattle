"""HumanoidBattle arena — 2 体を MuJoCo シーンで対面させ、ボクシング動作を物理エンジン上で再生し、
拳が相手の頭/胴に届いたかを幾何で判定する「実際に殴り合う」GIF を作る。

⚠️ 設計（正直な範囲）: 完全 forward dynamics で動かすと、バランス制御は v0 未解決のため両者倒れる
（[[real-video-demo-pipeline]] の depth/balance frontier）。そこで本 arena は **kinematic playback**:
毎フレーム両者の qpos を retarget 結果から設定して `mj_forward`（FK＋衝突検出は走るが時間積分しない＝
倒れない）。ヒットは MuJoCo 衝突系ではなく、**拳(手首)と相手の頭/胸の幾何距離**で判定（堅牢・調整可）。
「振り付けされたボクシングを実物理エンジンの 3D で描き、ヒットは幾何で採点」——接触ダイナミクス
（打撃の反動で相手がよろける）は v0 では出さない。これは honest な妥協で docstring に明示する。

スコア: 各ファイターが相手にクリーンヒットさせた回数（同一パンチは cooldown で 1 ヒット）。多い方が勝者。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from robotdance_core.rd_mir import RdMir
from robotdance_core.skeleton import index_of

_FPS = 30.0


def generate_boxing(*, duration: float = 4.0, fps: float = _FPS, lead: str = "left",
                    motion_id: str = "rdmir-synth-boxing-0001") -> RdMir:
    """ボクシングのコンビネーション（ガード→ジャブ→クロス→フック）を合成 RD-MIR で返す。

    立位（脚は固定）で上体・腕だけを動かす。x=前方（相手方向）へ拳を伸ばす。lead で利き腕を選ぶ。
    """
    from robotdance_core.synthetic import _REST

    n = round(fps * duration)
    t = np.arange(n) / fps
    kps = np.repeat(_REST[None].astype(np.float64), n, axis=0)

    li, ri = index_of("left_wrist"), index_of("right_wrist")
    lei, rei = index_of("left_elbow"), index_of("right_elbow")
    # ガード姿勢（拳を顔の前に上げる, x やや前・z 高め・中央寄り）。
    guard = {
        li: np.array([0.16, 0.07, 1.36]), lei: np.array([0.10, 0.15, 1.20]),
        ri: np.array([0.16, -0.07, 1.36]), rei: np.array([0.10, -0.15, 1.20]),
    }
    # フルエクステンション（パンチ）: 拳を前方 +x へ突き出す。
    ext = {
        li: np.array([0.62, 0.03, 1.34]), lei: np.array([0.34, 0.10, 1.30]),
        ri: np.array([0.62, -0.03, 1.34]), rei: np.array([0.34, -0.10, 1.30]),
    }

    def _pulse(center: float, width: float) -> np.ndarray:
        """center 付近で 0→1→0 に立ち上がるパンチ包絡（cos^2 窓）。"""
        x = (t - center) / width
        return np.where(np.abs(x) < 1.0, np.cos(x * math.pi / 2) ** 2, 0.0)

    # コンビネーション: 周期ごとに [lead ジャブ, 逆クロス, lead フック]。
    period = 1.6
    n_cyc = max(1, int(duration / period))
    left_lead = lead == "left"
    left_p = np.zeros(n)
    right_p = np.zeros(n)
    for c in range(n_cyc):
        base = c * period + 0.3
        (left_p if left_lead else right_p)[:] += _pulse(base, 0.22)          # ジャブ
        (right_p if left_lead else left_p)[:] += _pulse(base + 0.55, 0.22)   # クロス
        (left_p if left_lead else right_p)[:] += _pulse(base + 1.05, 0.26)   # フック/2nd
    left_p = np.clip(left_p, 0, 1)
    right_p = np.clip(right_p, 0, 1)

    for f in range(n):
        for w, e, p in ((li, lei, left_p[f]), (ri, rei, right_p[f])):
            kps[f, w] = guard[w] * (1 - p) + ext[w] * p
            kps[f, e] = guard[e] * (1 - p) + ext[e] * p
        # 軽い体重移動（前後の bob）でリズムを出す（脚は据え置き）。
        bob = 0.02 * math.sin(2 * math.pi * t[f] / period)
        kps[f, :11, 0] += bob

    from robotdance_core.rd_mir import Skeleton
    from robotdance_core.skeleton import JOINT_NAMES, PARENTS

    return RdMir(
        motion_id=motion_id,
        source_ref={"dataset_name": "robotdance-synthetic", "generator": "synthetic.generate_boxing"},
        license_state="redistributable",
        fps=fps,
        duration=duration,
        skeleton=Skeleton(joint_names=JOINT_NAMES, parents=PARENTS),
        root_trajectory={"position": kps[:, 0, :].tolist()},
        keypoints_3d=kps.tolist(),
        privacy_flags={"synthetic": True, "face_visible": False},
        semantics={"action_label": "boxing", "style_tag": "synthetic_demo"},
        extractor_versions={"generator": "robotdance.synthetic.v0"},
    )


_Z180 = np.array([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])  # z 軸 180° 回転（対面）。


@dataclass
class FightResult:
    p1: str
    p2: str
    p1_hits: int
    p2_hits: int
    winner: str
    frames: list = field(repr=False, default_factory=list)
    fps: float = _FPS
    p1_cum: list = field(repr=False, default_factory=list)  # 各フレーム時点の累積ヒット
    p2_cum: list = field(repr=False, default_factory=list)


def _single_model(morph):
    import mujoco

    from .mjcf import build_mjcf
    return mujoco.MjModel.from_xml_string(build_mjcf(morph, ground=False))


def _arena_kps(robot_kps: np.ndarray, R: np.ndarray, stance_xy: np.ndarray) -> np.ndarray:
    """robot kps[T,J,3] を pelvis 基準へ中心化→R 回転→足が地面 z=0 に来るよう配置した arena 座標。"""
    pel = robot_kps[:, index_of("pelvis"), :][:, None, :]
    centered = (robot_kps - pel) @ R.T
    z0 = centered[:, :, 2].min()
    out = centered.copy()
    out[:, :, 0] += stance_xy[0]
    out[:, :, 1] += stance_xy[1]
    out[:, :, 2] += -z0
    return out


def run_fight(morph_a, morph_b, *, name_a: str, name_b: str, separation: float = 0.17,
              hit_radius: float = 0.20, duration: float = 4.0, fps: float = _FPS,
              width: int = 480, height: int = 360, render: bool = True) -> FightResult:
    """2 体をボクシングさせ、拳→相手頭/胸の幾何ヒットを採点し、GIF フレームを返す。"""
    import mujoco

    from robotdance_retarget.kinematic import retarget

    # 1. 各ファイターのボクシング motion → robot kps → 単体 qpos。
    box_a = generate_boxing(duration=duration, fps=fps, lead="left")
    box_b = generate_boxing(duration=duration, fps=fps, lead="right")
    rk_a = retarget(box_a, morph_a).keypoints_3d_array()
    rk_b = retarget(box_b, morph_b).keypoints_3d_array()
    n = min(rk_a.shape[0], rk_b.shape[0])
    rk_a, rk_b = rk_a[:n], rk_b[:n]

    ak_a = _arena_kps(rk_a, np.eye(3), np.array([-separation, 0.0]))
    ak_b = _arena_kps(rk_b, _Z180, np.array([+separation, 0.0]))

    ma, mb = _single_model(morph_a), _single_model(morph_b)
    qa = _poses_to_qpos_arena(ma, morph_a, rk_a)
    qb = _poses_to_qpos_arena(mb, morph_b, rk_b)

    # 2. arena 組み立て（MjSpec.attach, ライト+コーナーカラー）。
    model, info = _build_arena(morph_a, morph_b, separation, ak_a, ak_b)
    data = mujoco.MjData(model)

    # 3. ヒット判定（拳 vs 相手頭/胸の最小距離, cooldown で 1 パンチ 1 ヒット）。
    p1_hits, p2_hits, frames, p1_cum, p2_cum = _play_and_score(
        model, data, ma, mb, qa, qb, ak_a, ak_b, info, separation,
        hit_radius, fps, width, height, render)

    winner = name_a if p1_hits > p2_hits else name_b if p2_hits > p1_hits else "DRAW"
    return FightResult(name_a, name_b, p1_hits, p2_hits, winner, frames, fps, p1_cum, p2_cum)


def _poses_to_qpos_arena(single_model, morph, robot_kps: np.ndarray) -> np.ndarray:
    from .mujoco_backend import _poses_to_qpos
    return _poses_to_qpos(single_model, morph, robot_kps)


def _build_arena(morph_a, morph_b, separation, ak_a, ak_b):
    """2 体を対面配置した MuJoCo モデルを MjSpec.attach で生成（ライト+赤/青コーナー）。"""
    import mujoco

    from .mjcf import build_mjcf

    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1],
                            rgba=[0.55, 0.57, 0.6, 1.0], name="ground")
    spec.worldbody.add_light(pos=[0, 0, 4.0], dir=[0, 0, -1])
    spec.worldbody.add_light(pos=[2.0, 2.0, 3.0], dir=[-0.5, -0.5, -1])
    q180 = [math.cos(math.pi / 2), 0, 0, math.sin(math.pi / 2)]
    for pfx, morph, pos, quat in (
        ("a_", morph_a, [-separation, 0, 0], [1, 0, 0, 0]),
        ("b_", morph_b, [separation, 0, 0], q180),
    ):
        child = mujoco.MjSpec.from_string(build_mjcf(morph, ground=False))
        fr = spec.worldbody.add_frame(pos=pos, quat=quat)
        spec.attach(child, prefix=pfx, frame=fr)
    model = spec.compile()
    # コーナーカラー: a_=赤, b_=青。
    for g in range(model.ngeom):
        bname = model.body(model.geom_bodyid[g]).name
        if bname.startswith("a_"):
            model.geom_rgba[g] = [0.85, 0.22, 0.22, 1.0]
        elif bname.startswith("b_"):
            model.geom_rgba[g] = [0.22, 0.4, 0.9, 1.0]
    info = {"q_a_adr": model.joint("a_root").qposadr[0],
            "q_b_adr": model.joint("b_root").qposadr[0]}
    return model, info


def _play_and_score(model, data, ma, mb, qa, qb, ak_a, ak_b, info, separation,
                    hit_radius, fps, width, height, render):
    import mujoco

    n = min(qa.shape[0], qb.shape[0])
    # arena 座標での拳と的の軌跡（ヒット判定はこの幾何で行う＝描画 qpos と一致）。
    lw, rw = index_of("left_wrist"), index_of("right_wrist")
    # 的: 頭・胸・みぞおち（spine）。背の低いファイターが背の高い相手にボディ打ちできるよう
    # 低い的も含める（reach 差はあっても完封にしない）。
    targets = (index_of("head"), index_of("chest"), index_of("spine"))
    # 各 fighter の ball-joint quat を arena qpos に書くためのアドレス対応。
    a_adr = [model.joint(f"a_jnt_{j}").qposadr[0] for j in range(1, 19)]
    b_adr = [model.joint(f"b_jnt_{j}").qposadr[0] for j in range(1, 19)]
    sa_adr = [ma.joint(f"jnt_{j}").qposadr[0] for j in range(1, 19)]
    sb_adr = [mb.joint(f"jnt_{j}").qposadr[0] for j in range(1, 19)]

    # 立ち高さ（足が地面 z=0 へ来るよう root z を上げる）。
    ha = float(-_centered_min_z(ma, qa))
    hb = float(-_centered_min_z(mb, qb))

    renderer = None
    cam = None
    if render:
        renderer = mujoco.Renderer(model, height, width)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.distance = 2.6
        cam.azimuth = 90
        cam.elevation = -8
        cam.lookat = [0, 0, 1.0]

    zw = np.array([1.0, 1.0, 0.5])  # z（高さ）を弱め重み: 前後/左右で届けば多少高さがズレても可。

    def _fist(att, dfn, fr):
        """フレーム fr での攻撃側の拳(左右) × 守備側の的（頭/胸/みぞおち）の最小距離 [m, z 重み付け]。"""
        return min(float(np.linalg.norm((att[fr, w] - dfn[fr, tg]) * zw))
                   for w in (lw, rw) for tg in targets)

    p1_hits = p2_hits = 0
    a_cd = b_cd = 0  # cooldown frames
    frames = []
    p1_cum: list[int] = []
    p2_cum: list[int] = []
    q180 = np.array([math.cos(math.pi / 2), 0, 0, math.sin(math.pi / 2)])
    for f in range(n):
        # arena qpos を構成（root を stance に固定, ball joint をコピー）。
        data.qpos[info["q_a_adr"]:info["q_a_adr"] + 3] = [-separation, 0, ha]
        data.qpos[info["q_a_adr"] + 3:info["q_a_adr"] + 7] = [1, 0, 0, 0]
        data.qpos[info["q_b_adr"]:info["q_b_adr"] + 3] = [separation, 0, hb]
        data.qpos[info["q_b_adr"] + 3:info["q_b_adr"] + 7] = q180
        for k in range(18):
            data.qpos[a_adr[k]:a_adr[k] + 4] = qa[f, sa_adr[k]:sa_adr[k] + 4]
            data.qpos[b_adr[k]:b_adr[k] + 4] = qb[f, sb_adr[k]:sb_adr[k] + 4]
        mujoco.mj_forward(model, data)

        # ヒット判定（arena 座標の幾何, 各拳 × 相手の頭/胸の最小距離・対称）。
        a_fist = _fist(ak_a, ak_b, f)
        b_fist = _fist(ak_b, ak_a, f)
        a_cd = max(0, a_cd - 1)
        b_cd = max(0, b_cd - 1)
        if a_fist < hit_radius and a_cd == 0:
            p1_hits += 1
            a_cd = int(0.4 * fps)
        if b_fist < hit_radius and b_cd == 0:
            p2_hits += 1
            b_cd = int(0.4 * fps)
        p1_cum.append(p1_hits)
        p2_cum.append(p2_hits)

        if render:
            renderer.update_scene(data, cam)
            frames.append(renderer.render().copy())
    if renderer is not None:
        del renderer
    return p1_hits, p2_hits, frames, p1_cum, p2_cum


def _centered_min_z(single_model, q: np.ndarray) -> float:
    """qpos[0] を単体モデルに与え root を原点に置いたときの最下 body z（立ち高さ補正用）。"""
    import mujoco

    d = mujoco.MjData(single_model)
    d.qpos[:] = q[0]
    d.qpos[0:3] = [0, 0, 0]
    d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(single_model, d)
    return float(d.xpos[:, 2].min())


__all__ = ["generate_boxing", "run_fight", "FightResult"]
