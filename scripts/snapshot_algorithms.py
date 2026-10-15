"""Record original source/config/selected weight hashes before integration."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def hashes():
    selected = []
    for name in ("algorithm1", "algorithm2", "algorithm3"):
        for path in (ROOT / name).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".yaml", ".pkl", ".pt", ".onnx", ".joblib", ".bin"}:
                if not any(part in {".venv", "__pycache__", "data", "reports"} for part in path.relative_to(ROOT).parts):
                    selected.append(path)
    return {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(selected)}


if __name__ == "__main__":
    output = ROOT / "artifacts" / "algorithm_integrity.json"
    output.parent.mkdir(exist_ok=True)
    if output.exists():
        before = json.loads(output.read_text())
        after = hashes()
        changes = [name for name, value in before.items() if after.get(name) != value]
        print(json.dumps({"checked": len(before), "changed": changes}, ensure_ascii=False))
        raise SystemExit(bool(changes))
    output.write_text(json.dumps(hashes(), indent=2), encoding="utf-8")
    print(output)
