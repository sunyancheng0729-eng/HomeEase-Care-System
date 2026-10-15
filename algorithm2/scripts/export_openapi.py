import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from homeease.api import app

(ROOT / "docs/openapi.json").write_text(json.dumps(app.openapi(), indent=2), encoding="utf-8")
print("Exported docs/openapi.json")
