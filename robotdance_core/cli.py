"""robotdance core CLI.

サブコマンド:
  validate  RD-Manifest / RD-MIR / RD-Embodiment の JSON を specs/ の JSON Schema で検証
  synth     合成ダンスモーション RD-MIR を生成（pose モデル不要のデモ種データ）
  view      RD-MIR の 3D スケルトンを GIF に描画

pose 抽出・retarget・sim は後続フェーズで追加する。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# specs/ はリポジトリルート直下（このファイルから 2 つ上）に固定配置されている。
_SPECS_DIR = Path(__file__).resolve().parent.parent / "specs"

_SCHEMAS = {
    "manifest": _SPECS_DIR / "rd-manifest" / "rd-manifest.schema.json",
    "mir": _SPECS_DIR / "rd-mir" / "rd-mir.schema.json",
    "embodiment": _SPECS_DIR / "rd-embodiment" / "rd-embodiment.schema.json",
}


def _validate(spec: str, path: Path) -> int:
    try:
        import jsonschema  # 遅延 import: validate 以外では不要
    except ImportError:
        print("error: jsonschema が必要です（pip install jsonschema）", file=sys.stderr)
        return 2

    schema = json.loads(_SCHEMAS[spec].read_text(encoding="utf-8"))
    instance = json.loads(path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: e.path)
    if errors:
        print(f"✗ {path} は rd-{spec} v0 schema に違反しています:")
        for err in errors:
            loc = "/".join(str(p) for p in err.path) or "(root)"
            print(f"  - {loc}: {err.message}")
        return 1
    print(f"✓ {path} は rd-{spec} v0 schema に適合しています")
    return 0


def _synth(out: Path, duration: float, fps: float) -> int:
    from .synthetic import generate_dance

    mir = generate_dance(duration=duration, fps=fps)
    mir.save(out)
    print(f"✓ 合成 RD-MIR を書き出しました: {out} "
          f"({mir.num_frames} frames, {mir.fps:g} fps, {mir.duration:g}s)")
    return 0


def _view(path: Path, out: Path, stride: int) -> int:
    from .rd_mir import RdMir
    from robotdance_viewer.skeleton_view import render_gif

    mir = RdMir.load(path)
    render_gif(mir, out, stride=stride)
    print(f"✓ スケルトン GIF を書き出しました: {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robotdance", description="RobotDance core CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="JSON を RobotDance spec で検証する")
    p_validate.add_argument("spec", choices=sorted(_SCHEMAS), help="検証する spec")
    p_validate.add_argument("path", type=Path, help="検証対象 JSON ファイル")

    p_synth = sub.add_parser("synth", help="合成ダンスモーション RD-MIR を生成する")
    p_synth.add_argument("-o", "--out", type=Path, default=Path("synthetic_dance.rdmir.json"))
    p_synth.add_argument("--duration", type=float, default=4.0)
    p_synth.add_argument("--fps", type=float, default=30.0)

    p_view = sub.add_parser("view", help="RD-MIR を 3D スケルトン GIF に描画する")
    p_view.add_argument("path", type=Path, help="RD-MIR JSON")
    p_view.add_argument("-o", "--out", type=Path, default=Path("skeleton.gif"))
    p_view.add_argument("--stride", type=int, default=2, help="何フレームおきに描画するか")

    args = parser.parse_args(argv)
    if args.command == "validate":
        return _validate(args.spec, args.path)
    if args.command == "synth":
        return _synth(args.out, args.duration, args.fps)
    if args.command == "view":
        return _view(args.path, args.out, args.stride)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
