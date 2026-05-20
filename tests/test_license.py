from datetime import datetime, timedelta, timezone

from Crypto.PublicKey import ECC

from label_detection.license import (
    LicenseError,
    build_license_payload,
    fingerprint_from_fields,
    sign_license_payload,
    verify_license_signature,
)


def test_fingerprint_from_fields_is_stable_and_ignores_hostname():
    fields = {
        "machine_id": "abc",
        "dmi_product_uuid": "uuid",
        "mac_addresses": "00:11:22:33:44:55",
        "platform_machine": "x86_64",
        "hostname": "host-a",
    }

    first = fingerprint_from_fields(fields)
    second = fingerprint_from_fields({**fields, "hostname": "host-b"})

    assert first == second
    assert len(first.split("-")) == 4


def test_fingerprint_requires_minimum_trusted_fields():
    try:
        fingerprint_from_fields({"machine_id": "abc", "hostname": "host"})
    except LicenseError as exc:
        assert "字段不足" in str(exc)
    else:
        raise AssertionError("expected LicenseError")


def test_license_signature_round_trip_rejects_tampering():
    key = ECC.generate(curve="Ed25519")
    public_key = key.public_key().export_key(format="PEM")
    payload = build_license_payload(
        machine="AAAA-BBBB-CCCC-DDDD",
        customer="test",
        issued_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        license_id="trial-test",
        features=["desktop"],
    )
    payload["signature"] = sign_license_payload(payload, key.export_key(format="PEM"))

    verify_license_signature(payload, public_key)

    payload["customer"] = "tampered"
    try:
        verify_license_signature(payload, public_key)
    except LicenseError as exc:
        assert "签名无效" in str(exc)
    else:
        raise AssertionError("expected LicenseError")
