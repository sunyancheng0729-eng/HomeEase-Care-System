"""Acquire only original UI-PRMD text files; pin and record mirror provenance."""
import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
REPO = "jkaethee/UI-PRMD_Exercise_Model"
REVISION = "475c66aba07731adbe1c90a15462fd30be564593"


def fetch(url):
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--movements", nargs="+", default=["m01", "m02", "m05", "m06", "m07"])
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    dest = ROOT / "data/raw/uiprmd"
    dest.mkdir(parents=True, exist_ok=True)
    commit = json.loads(fetch(f"https://api.github.com/repos/{REPO}/commits/{args.revision}"))["sha"]
    tree = json.loads(fetch(f"https://api.github.com/repos/{REPO}/git/trees/{commit}?recursive=1"))
    entries = [e for e in tree["tree"] if e["path"].startswith(("Segmented Movements/Kinect/", "Incorrect Segmented Movements/Kinect/"))
               and e["path"].endswith(".txt") and any(f"/{m}_" in e["path"] for m in args.movements)]

    def download(entry):
        path = entry["path"]
        local = dest / Path(path).name
        url = f"https://raw.githubusercontent.com/{REPO}/{commit}/{quote(path)}"
        if not local.exists() or local.stat().st_size != entry["size"]:
            local.write_bytes(fetch(url))
        raw = local.read_bytes()
        blob_hash = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if blob_hash != entry["sha"]:
            raw = fetch(url)
            blob_hash = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            if blob_hash != entry["sha"]:
                raise ValueError(f"Source blob integrity failed: {path}")
            local.write_bytes(raw)
        return {"file": local.name, "source_path": path, "url": url,
                "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "git_blob_sha": entry["sha"]}

    with ThreadPoolExecutor(max_workers=12) as pool:
        manifest = list(pool.map(download, entries))
    (dest / "manifest.json").write_text(json.dumps({"dataset": "UI-PRMD", "official": "https://webpages.uidaho.edu/ui-prmd/",
        "mirror": REPO, "commit": commit, "movements": args.movements, "files": manifest}, indent=2), encoding="utf-8")
    print(f"Downloaded {len(manifest)} files, {sum(e['bytes'] for e in manifest):,} bytes; commit {commit}", flush=True)


if __name__ == "__main__":
    main()
