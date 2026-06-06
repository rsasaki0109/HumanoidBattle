"""manifest 駆動の dataset ビルダー（v0）。

RD-Manifest のリストを受け取り、ローカルにある source を adapter で RD-MIR 化する。
各 clip は license firewall を通し、公開可なものだけを書き出す。最後に Data Bill of
Materials（build report + DATA_CARD.md）を出力する — どの source が・どの権利で・公開されたかを明示。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from robotdance_core.rd_mir import RdMir

from . import manifest as mf
from .aist import load_aist_pkl
from .amass import load_amass_npz

# dataset 名 → adapter（local file path, license_state → RdMir）。新データセットはここに追加。
ADAPTERS: dict[str, Callable[..., RdMir]] = {
    "amass": load_amass_npz,
    "aist": load_aist_pkl,
    "aist++": load_aist_pkl,
}


def _local_rel(uri: str) -> str:
    """source_uri からローカル相対パス部を取り出す。

    'dataset://amass/foo/bar.npz' → 'foo/bar.npz'、それ以外は uri をそのまま返す。
    """
    if uri.startswith("dataset://"):
        rest = uri[len("dataset://"):]
        return rest.split("/", 1)[1] if "/" in rest else ""
    return uri


def _resolve_source(entry: dict[str, Any], data_root: Path) -> Path | None:
    """manifest entry のローカル source path を解決する（無ければ None）。"""
    rel = _local_rel(entry.get("source_uri", ""))
    for cand in (Path(rel), data_root / rel):
        if rel and cand.exists():
            return cand
    return None


def _dataset_of(entry: dict[str, Any]) -> str:
    """manifest entry から dataset 名を推定する（source_uri 'dataset://<name>/...' 等）。"""
    uri = entry.get("source_uri", "")
    if uri.startswith("dataset://"):
        return uri[len("dataset://"):].split("/", 1)[0]
    return entry.get("source_type", "")


def build_dataset(
    manifests: list[dict[str, Any]],
    *,
    data_root: str | Path = ".",
    out_dir: str | Path = "build",
    dedupe: bool = False,
    dedupe_threshold: float = 0.98,
    qc: bool = True,
) -> dict[str, Any]:
    """manifest 群から RD-MIR を構築し、Data Bill of Materials を返す。

    dedupe=True なら motion embedding で near-duplicate を検出し、各グループから 1 本だけ残す。
    qc=True なら export 済み RD-MIR を motion-doctor で健全性診断し report["health"] に集計する。
    """
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bom: list[dict[str, Any]] = []
    exported = 0
    exported_mirs: list[tuple[dict[str, Any], RdMir, Path]] = []
    for entry in manifests:
        clip_id = entry.get("clip_id", "?")
        decision = mf.evaluate(entry)
        dataset = _dataset_of(entry)
        row: dict[str, Any] = {
            "clip_id": clip_id,
            "source_type": entry.get("source_type"),
            "dataset": dataset,
            "license_declared": entry.get("license_declared", "unknown"),
            "derived_motion_allowed": entry.get("derived_motion_allowed", False),
            "exported": False,
            "license_state": decision.license_state,
            "reason": decision.reason,
            "output": None,
        }
        if not decision.can_export_derived:
            bom.append(row)  # firewall により withheld
            continue
        adapter = ADAPTERS.get(dataset)
        src = _resolve_source(entry, data_root)
        if adapter is None:
            row["reason"] = f"未対応 dataset: {dataset}"
            bom.append(row)
            continue
        if src is None:
            row["reason"] = "local source が見つからない（manifest のみ・要再構築）"
            bom.append(row)
            continue
        mir = adapter(src, license_state=decision.license_state, motion_id=f"rdmir-{clip_id}")
        out_path = out_dir / f"{clip_id}.rdmir.json"
        mir.save(out_path)
        row["exported"] = True
        row["output"] = str(out_path)
        row["duplicate_group"] = None
        bom.append(row)
        exported_mirs.append((row, mir, out_path))
        exported += 1

    if dedupe and len(exported_mirs) > 1:
        exported -= _dedupe_exported(exported_mirs, dedupe_threshold)

    report = {
        "total": len(manifests),
        "exported": exported,
        "withheld": len(manifests) - exported,
        "bill_of_materials": bom,
    }
    if qc:
        report["health"] = _qc_exported(exported_mirs)
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "DATA_CARD.md").write_text(_render_data_card(report), encoding="utf-8")
    return report


def _qc_exported(
    exported: list[tuple[dict[str, Any], RdMir, Path]],
) -> dict[str, Any]:
    """export 済み（dedupe 後に残った）RD-MIR を motion-doctor で診断し集計する。

    各 row の "health"（warn 名リスト）も埋める。keypoints_3d が無い等で診断不能な場合は skip。
    """
    from collections import Counter

    from robotdance_motion.doctor import diagnose_motion, warn_names

    counts: Counter[str] = Counter()
    healthy = warn = skipped = 0
    for row, mir, _ in exported:
        if not row.get("exported"):
            continue  # dedupe で落ちたものは対象外
        try:
            ws = warn_names(diagnose_motion(mir))
        except Exception:  # noqa: BLE001 - keypoints_3d 無し等は診断不能 → skip
            row["health"] = "skipped"
            skipped += 1
            continue
        row["health"] = ws or "ok"
        counts.update(ws)
        if ws:
            warn += 1
        else:
            healthy += 1
    return {
        "checked": healthy + warn,
        "healthy": healthy,
        "warn": warn,
        "skipped": skipped,
        "warn_breakdown": dict(counts.most_common()),
    }


def _dedupe_exported(
    exported: list[tuple[dict[str, Any], RdMir, Path]], threshold: float
) -> int:
    """motion embedding で near-duplicate を検出し、各グループ 1 本だけ残す。除去数を返す。"""
    from robotdance_motion.embeddings import MotionIndex

    index = MotionIndex()
    for _, mir, _ in exported:
        index.add_mir(mir)
    ids = [m.motion_id for _, m, _ in exported]
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in index.duplicates(threshold):
        parent[find(a)] = find(b)

    by_id = {m.motion_id: (row, m, path) for row, m, path in exported}
    groups: dict[str, list[str]] = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)

    removed = 0
    for gid, members in groups.items():
        if len(members) < 2:
            continue
        # 最もフレーム数が多い clip を代表として残す。
        keep = max(members, key=lambda mid: by_id[mid][1].num_frames)
        for mid in members:
            row, _, path = by_id[mid]
            row["duplicate_group"] = keep
            if mid != keep:
                row["exported"] = False
                row["reason"] = f"near-duplicate of {keep}（dedupe）"
                row["output"] = None
                path.unlink(missing_ok=True)
                removed += 1
    return removed


def build_from_file(
    manifest_file: str | Path,
    *,
    data_root: str | Path = ".",
    out_dir: str | Path = "build",
    dedupe: bool = False,
    qc: bool = True,
) -> dict[str, Any]:
    """JSON 配列の manifest ファイルから build する（各要素を schema 検証）。"""
    import jsonschema

    entries = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
    if isinstance(entries, dict):
        entries = [entries]
    schema = json.loads(mf._SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    for e in entries:
        validator.validate(e)
    return build_dataset(entries, data_root=data_root, out_dir=out_dir, dedupe=dedupe, qc=qc)


def _render_data_card(report: dict[str, Any]) -> str:
    """Data Bill of Materials を Markdown で出力する。"""
    lines = [
        "# Data Bill of Materials",
        "",
        f"- total: **{report['total']}**  exported: **{report['exported']}**  "
        f"withheld (license firewall): **{report['withheld']}**",
        "",
        "| clip_id | dataset | license_declared | derived_allowed | exported | license_state | reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in report["bill_of_materials"]:
        lines.append(
            f"| {r['clip_id']} | {r['dataset']} | {r['license_declared']} | "
            f"{r['derived_motion_allowed']} | {'✅' if r['exported'] else '⛔'} | "
            f"{r['license_state']} | {r['reason']} |"
        )
    lines += ["", "> ⛔ = license firewall により派生 motion 非公開。raw source は再配布しない。"]
    h = report.get("health")
    if h:
        lines += [
            "",
            "## Health (motion-doctor)",
            "",
            f"- checked: **{h['checked']}**  healthy: **{h['healthy']}**  "
            f"warn: **{h['warn']}**  skipped: **{h['skipped']}**",
        ]
        if h["warn_breakdown"]:
            lines.append("- warn 内訳: "
                         + ", ".join(f"{k}×{v}" for k, v in h["warn_breakdown"].items()))
    return "\n".join(lines) + "\n"
