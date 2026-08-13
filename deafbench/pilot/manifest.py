"""Signed aggregate-only manifests for customer-executed audits."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


EXECUTION_NOTICE = "Customer-executed; results depend on the customer environment."
SELF_SIGNED_NOTICE = (
    "The embedded self-signature detects changes but does not establish signer "
    "identity; authenticity requires an independently trusted key fingerprint."
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MODEL_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_EVALUATOR_REVISION = re.compile(r"[0-9a-f]{40,64}\Z")
_ENTITY_TYPE = re.compile(r"[A-Z][A-Z_]*\Z")
_CONFIGURATION_FIELDS = frozenset(
    {
        "batch_size",
        "device",
        "dtype",
        "keyword_biasing",
        "language",
        "max_new_tokens",
        "num_beams",
        "timestamps",
        "trust_remote_code",
    }
)
_METRIC_FIELDS = frozenset(
    {
        "canonical_semantic_recall_percent",
        "critical_failures_by_entity_type",
        "deletions",
        "insertions",
        "local_rtfx",
        "median_latency_ms",
        "peak_vram_bytes",
        "strict_lexical_recall_percent",
        "substitutions",
        "wer_percent",
    }
)
_MODEL_FIELDS = frozenset(
    {
        "aggregate_metrics",
        "configuration",
        "license_classification",
        "model_id",
        "revision",
    }
)
_PAYLOAD_FIELDS = frozenset(
    {
        "artifact_hashes",
        "sample_count",
        "evaluator_version",
        "execution_notice",
        "models",
        "schema_version",
    }
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _validate_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict) or set(payload) != _PAYLOAD_FIELDS:
        raise ValueError("manifest fields are incomplete or unsupported")
    if payload["schema_version"] != 1 or payload["execution_notice"] != EXECUTION_NOTICE:
        raise ValueError("manifest identity or execution notice is invalid")
    if not isinstance(payload["sample_count"], int) or payload["sample_count"] <= 0:
        raise ValueError("manifest dataset count must be positive")
    if _EVALUATOR_REVISION.fullmatch(str(payload["evaluator_version"])) is None:
        raise ValueError("manifest evaluator version is invalid")

    models = payload["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("manifest requires model aggregates")
    seen: set[str] = set()
    for model in models:
        if not isinstance(model, dict) or set(model) != _MODEL_FIELDS:
            raise ValueError("manifest model fields are unsupported")
        model_id = model["model_id"]
        if not isinstance(model_id, str) or not model_id or model_id in seen:
            raise ValueError("manifest model identifier is invalid or duplicated")
        seen.add(model_id)
        if _MODEL_REVISION.fullmatch(str(model["revision"])) is None:
            raise ValueError("manifest model revision is invalid")
        if model["license_classification"] != "commercial_candidate":
            raise ValueError("manifest model is not a commercial candidate")
        configuration = model["configuration"]
        if (
            not isinstance(configuration, dict)
            or not set(configuration).issubset(_CONFIGURATION_FIELDS)
            or any(
                not isinstance(value, (str, int, float, bool))
                for value in configuration.values()
            )
        ):
            raise ValueError("manifest configuration is not aggregate-safe")
        metrics = model["aggregate_metrics"]
        if not isinstance(metrics, dict) or set(metrics) != _METRIC_FIELDS:
            raise ValueError("manifest aggregate metric fields are incomplete")
        counts = metrics["critical_failures_by_entity_type"]
        if (
            not isinstance(counts, dict)
            or any(
                _ENTITY_TYPE.fullmatch(str(key)) is None
                or not isinstance(value, int)
                or value < 0
                for key, value in counts.items()
            )
            or any(
                not _is_number(value)
                for key, value in metrics.items()
                if key != "critical_failures_by_entity_type"
            )
        ):
            raise ValueError("manifest aggregate metric values are invalid")

    artifacts = payload["artifact_hashes"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("manifest requires artifact hashes")
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) not in (
            {"artifact_type", "sha256"},
            {"artifact_type", "model_id", "sha256"},
        ):
            raise ValueError("manifest artifact hash fields are unsupported")
        if artifact["artifact_type"] not in {
            "local_result_manifest",
            "redacted_report",
        } or _SHA256.fullmatch(str(artifact["sha256"])) is None:
            raise ValueError("manifest artifact hash is invalid")
        if "model_id" in artifact and artifact["model_id"] not in seen:
            raise ValueError("manifest artifact model is unknown")
    return payload


def _is_within(path: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath((path.resolve(), parent.resolve())) == str(
            parent.resolve()
        )
    except ValueError:
        return False


def _load_or_create_key(path: Path) -> Ed25519PrivateKey:
    if path.exists():
        key = serialization.load_pem_private_key(path.read_bytes(), None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError("manifest signing key is not Ed25519")
        return key
    key = Ed25519PrivateKey.generate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    path.chmod(0o600)
    return key


def write_signed_manifest(
    path: Path, *, payload: dict[str, object], key_path: Path
) -> str:
    """Validate, sign, and write a customer-local reproducibility manifest."""

    destination = Path(path)
    private_path = Path(key_path)
    if _is_within(private_path, destination.parent):
        raise ValueError("private signing key cannot be stored in the export")
    validated = _validate_payload(payload)
    key = _load_or_create_key(private_path)
    body = _canonical(validated)
    signature = key.sign(body)
    public_key = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    document = {
        **validated,
        "signature": {
            "algorithm": "Ed25519",
            "trust_model": "self_signed_integrity_only",
            "key_fingerprint_sha256": hashlib.sha256(public_key).hexdigest(),
            "public_key_base64": base64.b64encode(public_key).decode("ascii"),
            "value_base64": base64.b64encode(signature).decode("ascii"),
        },
    }
    encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(encoded, encoding="utf-8", newline="\n")
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def verify_signed_manifest(
    path: Path, *, trusted_key_sha256: str | None = None
) -> bool:
    """Verify integrity and, when pinned separately, the signing key identity."""

    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        signature = document.pop("signature")
        payload = _validate_payload(document)
        if set(signature) != {
            "algorithm",
            "key_fingerprint_sha256",
            "public_key_base64",
            "trust_model",
            "value_base64",
        }:
            return False
        if (
            signature["algorithm"] != "Ed25519"
            or signature["trust_model"] != "self_signed_integrity_only"
        ):
            return False
        public_bytes = base64.b64decode(
            signature["public_key_base64"], validate=True
        )
        fingerprint = hashlib.sha256(public_bytes).hexdigest()
        if signature["key_fingerprint_sha256"] != fingerprint:
            return False
        if trusted_key_sha256 is not None and trusted_key_sha256 != fingerprint:
            return False
        public_key = Ed25519PublicKey.from_public_bytes(public_bytes)
        public_key.verify(
            base64.b64decode(signature["value_base64"], validate=True),
            _canonical(payload),
        )
    except (OSError, ValueError, KeyError, TypeError, InvalidSignature):
        return False
    return True
