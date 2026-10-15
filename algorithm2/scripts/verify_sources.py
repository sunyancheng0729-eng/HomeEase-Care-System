"""Offline integrity audit of source texts and current deployment artifacts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "data/raw/uiprmd"
    manifest = json.loads((directory / "manifest.json").read_text())
    for entry in manifest["files"]:
        raw = (directory / entry["file"]).read_bytes()
        assert len(raw) == entry["bytes"], entry["file"]
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"], entry["file"]
        assert hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == entry["git_blob_sha"], entry["file"]
    result = {"verified_source_files": len(manifest["files"]), "source_bytes": sum(e["bytes"] for e in manifest["files"]),
              "source_commit": manifest["commit"], "sha256_and_git_blob_sha": "passed"}
    (ROOT / "reports/source_integrity.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
