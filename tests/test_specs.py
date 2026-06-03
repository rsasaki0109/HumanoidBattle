"""specs/ の JSON Schema 自体が妥当で、examples/ がそれに適合することを検証する最小テスト。"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SPECS = _ROOT / "specs"

_SCHEMAS = {
    "manifest": _SPECS / "rd-manifest" / "rd-manifest.schema.json",
    "mir": _SPECS / "rd-mir" / "rd-mir.schema.json",
    "embodiment": _SPECS / "rd-embodiment" / "rd-embodiment.schema.json",
}

_EXAMPLES = {
    "manifest": _ROOT / "examples" / "minimal_manifest.json",
    "mir": _ROOT / "examples" / "minimal_mir.json",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", sorted(_SCHEMAS))
def test_schema_is_valid_draft202012(name: str) -> None:
    """各 schema が Draft 2020-12 として整合していること。"""
    schema = _load(_SCHEMAS[name])
    jsonschema.Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize("name", sorted(_EXAMPLES))
def test_example_conforms_to_schema(name: str) -> None:
    """examples/ が対応 schema に適合すること。"""
    schema = _load(_SCHEMAS[name])
    instance = _load(_EXAMPLES[name])
    jsonschema.Draft202012Validator(schema).validate(instance)


def test_mir_rejects_unknown_license_state_value() -> None:
    """license_state は enum 外の値を拒否すること（ライセンス安全性の最低限の担保）。"""
    schema = _load(_SCHEMAS["mir"])
    instance = _load(_EXAMPLES["mir"])
    instance["license_state"] = "definitely_fine_trust_me"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(instance)
