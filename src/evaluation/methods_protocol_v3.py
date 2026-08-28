"""Compatibility re-export for the dependency-free v3 protocol contract."""

from src.methods_protocol_v3_contract import (  # noqa: F401
    MethodsProtocolV3,
    build_methods_protocol_v3,
    protocol_identity_v3,
    protocol_to_mapping_v3,
)

# Legacy names remain rebindable compatibility exports, never protocol truth.
BRIDGE_PRIMARY_BANDS = (("proximal", 1, 5), ("local", 6, 15))
BRIDGE_SECONDARY_BANDS = (("transition", 16, 30), ("distal", 31, 60))

__all__ = [
    "BRIDGE_PRIMARY_BANDS",
    "BRIDGE_SECONDARY_BANDS",
    "MethodsProtocolV3",
    "build_methods_protocol_v3",
    "protocol_identity_v3",
    "protocol_to_mapping_v3",
]
