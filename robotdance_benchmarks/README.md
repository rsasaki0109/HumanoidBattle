# robotdance_benchmarks

extraction / retarget / sim tracking benchmark — 抽出・retarget・追従の評価。

## 実装状況

- `suite.py` — `run_benchmark(motions, robots)`: motion × robot を full pipeline
  （retarget → MuJoCo 物理検証）に通し、既存の全指標を 1 行 = 1 (motion, robot) に集約。
  `default_motion_suite()` は権利クリーンな合成スイート。`run_from_dir()` で `*.rdmir.json` も可。
- `report.py` — CSV 出力 + Markdown **leaderboard**（robot 別 PASS率・平均指標）。

```bash
robotdance benchmark --robots unitree_g1 unitree_h1 -o out/
# → out/benchmark.csv, out/LEADERBOARD.md
```

集約する指標: retarget（height_scale, bone_direction_cosine, foot_sliding）、
sim_certificate（verdict, airborne, balance, torque_ratio, ang_speed）、source 品質（confidence, jitter）。

サンプル結果（合成スイート × G1/H1）: [`../docs/benchmark/LEADERBOARD.md`](../docs/benchmark/LEADERBOARD.md)。

> ⚠️ v0 は近似形態プロキシ + 近似質量。mujoco 未インストール時は sim 指標が None になる
> （retarget 指標のみ）。extraction benchmark（実動画）や leaderboard 提出フローは今後。
