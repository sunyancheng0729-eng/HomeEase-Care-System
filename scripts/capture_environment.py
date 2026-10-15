"""Record the installed runtime dependency closure for reproducible validation."""
import json
import platform
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
direct = [Requirement(line).name for line in (ROOT / "requirements.txt").read_text().splitlines()
          if line.strip() and not line.startswith("#")]
pending = direct.copy()
packages = {}
while pending:
    name = canonicalize_name(pending.pop())
    if name in packages:
        continue
    distribution = metadata.distribution(name)
    packages[name] = distribution.version
    for raw in distribution.requires or []:
        requirement = Requirement(raw)
        if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
            pending.append(requirement.name)

result = {"python": platform.python_version(), "platform": platform.platform(), "direct_dependencies": direct,
          "packages": dict(sorted(packages.items())),
          "note": "Verified Windows dependency closure. CUDA PyTorch wheels require their matching wheel index."}
(ROOT / "artifacts" / "environment.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
header = "# Verified Windows environment; Python 3.11.15\n# CUDA wheels: --extra-index-url https://download.pytorch.org/whl/cu128\n"
(ROOT / "requirements-lock.txt").write_text(header + "".join(f"{name}=={version}\n" for name, version in sorted(packages.items())), encoding="ascii")
print(f"Recorded {len(packages)} packages")
