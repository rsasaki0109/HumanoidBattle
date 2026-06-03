# RobotDance Benchmark

motions: **4** × robots: **2** = 8 runs · sim: **on**

> ⚠️ v0: 近似形態プロキシ + 近似慣性。実機保証ではない（各 README 参照）。

## Leaderboard（robot 別集計）

| robot | runs | PASS率 | 平均 bone方向cos | 平均 foot_sliding | 平均 height_scale |
| --- | --- | --- | --- | --- | --- |
| unitree_g1 | 4 | 0.750 | 1.000 | 0.024 | 0.798 |
| unitree_h1 | 4 | 0.500 | 1.000 | 0.034 | 1.205 |

## 全 run（motion × robot）

| motion | class | robot | verdict | airborne | balance | torque× | 角速度 | foot_slide | bone_cos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dance_normal | dance | unitree_g1 | PASS | 0.000 | 0.000 | 0.152 | 4.990 | 0.004 | 1.000 |
| dance_normal | dance | unitree_h1 | PASS | 0.000 | 0.025 | 0.285 | 4.990 | 0.004 | 1.000 |
| dance_fast | dance | unitree_g1 | PASS | 0.000 | 0.000 | 0.152 | 8.010 | 0.006 | 1.000 |
| dance_fast | dance | unitree_h1 | REJECT | 0.000 | 0.800 | 0.285 | 8.010 | 0.006 | 1.000 |
| idle | dance | unitree_g1 | PASS | 0.000 | 0.000 | 0.105 | 0.780 | 0.002 | 1.000 |
| idle | dance | unitree_h1 | PASS | 0.000 | 0.000 | 0.176 | 0.790 | 0.002 | 1.000 |
| backflip | backflip | unitree_g1 | REJECT | 0.875 | 0.917 | 0.432 | 38.360 | 0.083 | 1.000 |
| backflip | backflip | unitree_h1 | REJECT | 0.875 | 0.958 | 0.662 | 46.840 | 0.122 | 1.000 |
