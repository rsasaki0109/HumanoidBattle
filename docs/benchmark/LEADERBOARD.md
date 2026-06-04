# RobotDance Benchmark

motions: **5** × robots: **4** = 20 runs · sim: **on**

> ⚠️ v0: 近似形態プロキシ。sim は実 URDF 慣性テンソルで検証（v0.52）。実機保証ではない（各 README 参照）。

## Leaderboard（robot 別集計）

| robot | runs | PASS率 | 平均 bone方向cos | 平均 foot_sliding | 平均 height_scale | 平均 屈曲違反率 |
| --- | --- | --- | --- | --- | --- | --- |
| unitree_g1 | 5 | 0.400 | 1.000 | 0.020 | 0.906 | 0.050 |
| unitree_h1 | 5 | 0.800 | 1.000 | 0.026 | 1.168 | 0.000 |
| booster_t1 | 5 | 0.600 | 1.000 | 0.017 | 0.686 | 0.037 |
| apptronik_apollo | 5 | 0.800 | 1.000 | 0.027 | 1.136 | 0.000 |

## 全 run（motion × robot）

| motion | class | robot | verdict | airborne | balance | torque× | 角速度 | foot_slide | bone_cos | 屈曲違反 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dance_normal | dance | unitree_g1 | PASS | 0.000 | 0.000 | 0.262 | 7.260 | 0.004 | 1.000 | 0.000 |
| dance_normal | dance | unitree_h1 | PASS | 0.000 | 0.000 | 0.513 | 7.260 | 0.004 | 1.000 | 0.000 |
| dance_normal | dance | booster_t1 | PASS | 0.000 | 0.000 | 0.256 | 7.260 | 0.004 | 1.000 | 0.000 |
| dance_normal | dance | apptronik_apollo | PASS | 0.000 | 0.000 | 0.327 | 7.260 | 0.004 | 1.000 | 0.000 |
| dance_fast | dance | unitree_g1 | REJECT | 0.000 | 0.458 | 0.262 | 11.630 | 0.006 | 1.000 | 0.000 |
| dance_fast | dance | unitree_h1 | PASS | 0.000 | 0.000 | 0.513 | 11.630 | 0.006 | 1.000 | 0.000 |
| dance_fast | dance | booster_t1 | PASS | 0.000 | 0.000 | 0.256 | 11.630 | 0.006 | 1.000 | 0.000 |
| dance_fast | dance | apptronik_apollo | PASS | 0.000 | 0.000 | 0.327 | 11.630 | 0.006 | 1.000 | 0.000 |
| idle | dance | unitree_g1 | PASS | 0.000 | 0.000 | 0.154 | 0.930 | 0.002 | 1.000 | 0.000 |
| idle | dance | unitree_h1 | PASS | 0.000 | 0.000 | 0.351 | 0.930 | 0.002 | 1.000 | 0.000 |
| idle | dance | booster_t1 | PASS | 0.000 | 0.000 | 0.200 | 0.930 | 0.002 | 1.000 | 0.000 |
| idle | dance | apptronik_apollo | PASS | 0.000 | 0.000 | 0.327 | 0.930 | 0.002 | 1.000 | 0.000 |
| backflip | backflip | unitree_g1 | REJECT | 0.875 | 0.938 | 0.389 | 4.010 | 0.089 | 1.000 | 0.000 |
| backflip | backflip | unitree_h1 | REJECT | 0.875 | 0.938 | 0.337 | 4.010 | 0.116 | 1.000 | 0.000 |
| backflip | backflip | booster_t1 | REJECT | 0.875 | 0.958 | 1.325 | 4.010 | 0.072 | 1.000 | 0.000 |
| backflip | backflip | apptronik_apollo | REJECT | 0.875 | 0.938 | 0.724 | 4.010 | 0.121 | 1.000 | 0.000 |
| overbend | overbend | unitree_g1 | REJECT | 0.000 | 0.000 | 0.220 | 3.920 | 0.000 | 1.000 | 0.250 |
| overbend | overbend | unitree_h1 | PASS | 0.000 | 0.000 | 0.486 | 3.920 | 0.000 | 1.000 | 0.000 |
| overbend | overbend | booster_t1 | REJECT | 0.000 | 0.000 | 0.203 | 3.920 | 0.000 | 1.000 | 0.183 |
| overbend | overbend | apptronik_apollo | PASS | 0.000 | 0.000 | 0.324 | 3.920 | 0.000 | 1.000 | 0.000 |
