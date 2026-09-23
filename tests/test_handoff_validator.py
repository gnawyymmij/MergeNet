#!/usr/bin/env python3
"""Offline tests for the result-package validator; no data or GPU required."""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_handoff import L0_FILES  # noqa: E402


def main():
    with tempfile.TemporaryDirectory(prefix="mergenet-handoff-test-") as temp:
        packet = Path(temp) / "share_packet"
        packet.mkdir()
        entries = []
        for rel in sorted(L0_FILES):
            target = packet / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = b"1.0\n" if rel == "SCHEMA_VERSION.txt" else b"{}\n"
            target.write_bytes(payload)
            entries.append({
                "path": rel,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "share_level": "L0",
                "redaction_status": "checked",
            })
        (packet / "file_manifest.json").write_text(
            json.dumps({"files": entries}), encoding="utf-8"
        )
        command = [sys.executable, str(ROOT / "scripts/validate_handoff.py"),
                   str(packet), "--level", "L0"]
        passed = subprocess.run(command, text=True, capture_output=True, check=False)
        assert passed.returncode == 0, passed.stderr + passed.stdout
        assert "PASS L0" in passed.stdout

        (packet / "README.md").write_text("[]\n", encoding="utf-8")
        failed = subprocess.run(command, text=True, capture_output=True, check=False)
        assert failed.returncode != 0
        assert "SHA-256 mismatch" in failed.stderr
    print("handoff validator self-test passed")


if __name__ == "__main__":
    main()
