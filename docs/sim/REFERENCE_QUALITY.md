# Reference qpos 品質（twist 安定化の効果）

> 生成: `python3 -m robotdance_sim.reference_quality > docs/sim/REFERENCE_QUALITY.md`（決定的・sim 依存）

RL tracking / PD 追従が追う reference qpos 列の **reference 関節速度**（連続フレーム間の
tangent 空間差分 = コントローラが要求する速度）を、単フレーム独立復元と時系列復元
（`_poses_to_qpos`, v0.47 既定）で比較する。極端な屈曲では単フレーム復元が bone 軸まわりに
偽 twist スパイクを生み reference 速度が跳ねる。時系列復元はそれを除去し、物理的に真の
bone 方向速度（twist-free）に整合する。**bone 方向は両者で厳密一致するので位置・COM・
verdict は不変**で、差はすべて不可観測な twist アーティファクト。

| robot | motion | per-frame [rad/s] | temporal [rad/s] | bone-truth [rad/s] | spike factor |
| --- | --- | ---: | ---: | ---: | ---: |
| unitree_g1 | dance_normal | 5.0 | 5.03 | 7.26 | 1.0× |
| unitree_g1 | dance_fast | 8.0 | 8.07 | 11.63 | 1.0× |
| unitree_g1 | idle | 0.8 | 0.81 | 0.93 | 1.0× |
| unitree_g1 | backflip | 21.2 | 4.01 | 4.01 | 5.3× |
| unitree_g1 | overbend | 79.9 | 3.92 | 3.92 | 20.4× |
| unitree_h1 | dance_normal | 5.0 | 5.03 | 7.26 | 1.0× |
| unitree_h1 | dance_fast | 8.0 | 8.07 | 11.63 | 1.0× |
| unitree_h1 | idle | 0.8 | 0.81 | 0.93 | 1.0× |
| unitree_h1 | backflip | 29.4 | 4.01 | 4.01 | 7.3× |
| unitree_h1 | overbend | 3.9 | 3.92 | 3.92 | 1.0× |

> per-frame と temporal が一致する motion（spike factor ≈ 1）は反平行付近に滞在する bone が
> 無く特異点を踏まないため。overbend のような過屈曲でのみ偽スパイクが顕在化する。

## 追従可能性（実機アクチュエータ速度上限 vs reference 要求速度）

reference が要求する関節角速度を、その関節の**実 URDF 速度上限**（v0.38 取り込みの
`per_joint_limits.velocity`）と比較する。上限超過は物理的に追従不能＝コントローラが原理的に
再現できない reference を意味する。単フレーム復元の偽 twist スパイクは速度上限を大きく超え
reference を untrackable にするが、時系列復元は **全 motion で速度包絡内（untrackable 0%）**。
コントローラ学習・PD 追従に渡せる trackable な reference を保証する。

| robot | motion | per-frame untrackable | per-frame max demand | temporal untrackable | temporal max demand |
| --- | --- | ---: | ---: | ---: | ---: |
| unitree_g1 | dance_normal | 0.0% | 0.1× | 0.0% | 0.1× |
| unitree_g1 | dance_fast | 0.0% | 0.2× | 0.0% | 0.2× |
| unitree_g1 | idle | 0.0% | 0.0× | 0.0% | 0.0× |
| unitree_g1 | backflip | 10.6% | 1.8× | 0.0% | 0.2× |
| unitree_g1 | overbend | 6.8% | 2.2× | 0.0% | 0.1× |
| unitree_h1 | dance_normal | 0.0% | 0.2× | 0.0% | 0.3× |
| unitree_h1 | dance_fast | 0.0% | 0.4× | 0.0% | 0.4× |
| unitree_h1 | idle | 0.0% | 0.1× | 0.0% | 0.1× |
| unitree_h1 | backflip | 10.6% | 2.4× | 0.0% | 0.3× |
| unitree_h1 | overbend | 0.0% | 0.3× | 0.0% | 0.3× |

> max demand = max(要求速度 / 実速度上限)。>1.0 で追従不能。backflip は per-frame だと実機の
> アクチュエータ速度を最大 2.4× 超える要求を 10% のフレームで出すが、temporal は ≤0.4× に収まる。

