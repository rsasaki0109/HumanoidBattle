"""robotdance_core

schemas, validators, metadata, CLI, config — RobotDance の中核。RD-MIR/RD-Manifest 等のスキーマ検証と共通基盤。
"""

from .rd_mir import RdMir, Skeleton, WorldFrame
from .skeleton import CANONICAL_SKELETON, JOINT_NAMES, PARENTS

# synthetic は scipy（demo extra）を要するため遅延 import に留め、ここでは公開しない。
__all__ = [
    "RdMir",
    "Skeleton",
    "WorldFrame",
    "CANONICAL_SKELETON",
    "JOINT_NAMES",
    "PARENTS",
]
