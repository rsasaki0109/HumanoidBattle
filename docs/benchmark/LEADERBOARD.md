# RobotDance Benchmark

motions: **7** × robots: **4** = 28 runs · sim: **on**

> ⚠️ v0: 近似形態プロキシ。sim は実 URDF 慣性テンソルで検証（v0.52）。実機保証ではない（各 README 参照）。

## Leaderboard（robot 別集計）

| robot | runs | PASS率 | 平均 bone方向cos | 平均 foot_sliding | 平均 height_scale | 平均 屈曲違反率 |
| --- | --- | --- | --- | --- | --- | --- |
| unitree_g1 | 7 | 0.429 | 1.000 | 0.017 | 0.891 | 0.036 |
| unitree_h1 | 7 | 0.571 | 1.000 | 0.021 | 1.149 | 0.000 |
| booster_t1 | 7 | 0.571 | 1.000 | 0.015 | 0.675 | 0.026 |
| apptronik_apollo | 7 | 0.714 | 1.000 | 0.022 | 1.117 | 0.000 |

## 全 run（motion × robot）

| motion | class | robot | verdict | airborne | balance | torque× | 角速度 | foot_slide | bone_cos | 屈曲違反 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dance_normal | dance | unitree_g1 | PASS | 0.000 | 0.000 | 0.339 | 7.260 | 0.004 | 1.000 | 0.000 |
| dance_normal | dance | unitree_h1 | PASS | 0.000 | 0.000 | 0.779 | 7.260 | 0.004 | 1.000 | 0.000 |
| dance_normal | dance | booster_t1 | PASS | 0.000 | 0.000 | 0.324 | 7.260 | 0.004 | 1.000 | 0.000 |
| dance_normal | dance | apptronik_apollo | PASS | 0.000 | 0.000 | 0.448 | 7.260 | 0.004 | 1.000 | 0.000 |
| dance_fast | dance | unitree_g1 | REJECT | 0.000 | 0.458 | 0.525 | 11.630 | 0.006 | 1.000 | 0.000 |
| dance_fast | dance | unitree_h1 | REJECT | 0.000 | 0.000 | 1.254 | 11.630 | 0.006 | 1.000 | 0.000 |
| dance_fast | dance | booster_t1 | PASS | 0.000 | 0.000 | 0.509 | 11.630 | 0.006 | 1.000 | 0.000 |
| dance_fast | dance | apptronik_apollo | PASS | 0.000 | 0.000 | 0.761 | 11.630 | 0.006 | 1.000 | 0.000 |
| idle | dance | unitree_g1 | PASS | 0.000 | 0.000 | 0.147 | 0.930 | 0.002 | 1.000 | 0.000 |
| idle | dance | unitree_h1 | PASS | 0.000 | 0.000 | 0.340 | 0.930 | 0.002 | 1.000 | 0.000 |
| idle | dance | booster_t1 | PASS | 0.000 | 0.000 | 0.216 | 0.930 | 0.002 | 1.000 | 0.000 |
| idle | dance | apptronik_apollo | PASS | 0.000 | 0.000 | 0.343 | 0.930 | 0.002 | 1.000 | 0.000 |
| backflip | backflip | unitree_g1 | REJECT | 0.875 | 0.938 | 0.521 | 4.010 | 0.089 | 1.000 | 0.000 |
| backflip | backflip | unitree_h1 | REJECT | 0.875 | 0.938 | 1.191 | 4.010 | 0.116 | 1.000 | 0.000 |
| backflip | backflip | booster_t1 | REJECT | 0.875 | 0.958 | 1.646 | 4.010 | 0.072 | 1.000 | 0.000 |
| backflip | backflip | apptronik_apollo | REJECT | 0.875 | 0.938 | 1.124 | 4.010 | 0.121 | 1.000 | 0.000 |
| overbend | overbend | unitree_g1 | REJECT | 0.000 | 0.000 | 0.225 | 3.920 | 0.000 | 1.000 | 0.250 |
| overbend | overbend | unitree_h1 | PASS | 0.000 | 0.000 | 0.504 | 3.920 | 0.000 | 1.000 | 0.000 |
| overbend | overbend | booster_t1 | REJECT | 0.000 | 0.000 | 0.216 | 3.920 | 0.000 | 1.000 | 0.183 |
| overbend | overbend | apptronik_apollo | PASS | 0.000 | 0.000 | 0.324 | 3.920 | 0.000 | 1.000 | 0.000 |
| squat | squat | unitree_g1 | PASS | 0.000 | 0.000 | 0.298 | 2.200 | 0.001 | 1.000 | 0.000 |
| squat | squat | unitree_h1 | PASS | 0.000 | 0.000 | 0.615 | 2.200 | 0.000 | 1.000 | 0.000 |
| squat | squat | booster_t1 | PASS | 0.000 | 0.000 | 0.283 | 2.200 | 0.001 | 1.000 | 0.000 |
| squat | squat | apptronik_apollo | PASS | 0.000 | 0.000 | 0.449 | 2.200 | 0.000 | 1.000 | 0.000 |
| march | march | unitree_g1 | REJECT | 0.000 | 0.583 | 1.059 | 5.610 | 0.018 | 1.000 | 0.000 |
| march | march | unitree_h1 | REJECT | 0.000 | 0.450 | 1.103 | 5.610 | 0.018 | 1.000 | 0.000 |
| march | march | booster_t1 | REJECT | 0.000 | 0.450 | 1.587 | 5.610 | 0.016 | 1.000 | 0.000 |
| march | march | apptronik_apollo | REJECT | 0.000 | 0.583 | 2.801 | 5.610 | 0.019 | 1.000 | 0.000 |
