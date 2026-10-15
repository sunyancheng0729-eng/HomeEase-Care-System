"""Compare pretrained pose CNNs on a deterministic real COCO validation subset."""
import argparse
import hashlib
import json
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.download_data import fetch


def iou(a, b):
    tl, br = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    area = np.maximum(br - tl, 0).prod()
    return area / max(np.maximum(a[2:] - a[:2], 0).prod() + np.maximum(b[2:] - b[:2], 0).prod() - area, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=100)
    args = parser.parse_args()
    data = ROOT / "data/coco"
    data.mkdir(parents=True, exist_ok=True)
    annotation_path = data / "person_keypoints_val2017.json"
    if not annotation_path.exists():
        archive = data / "annotations_trainval2017.zip"
        if not archive.exists():
            url = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"
            with requests.get(url, stream=True, timeout=180) as response:
                response.raise_for_status()
                with archive.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
        with zipfile.ZipFile(archive) as zipped:
            annotation_path.write_bytes(zipped.read("annotations/person_keypoints_val2017.json"))
    annotations = json.loads(annotation_path.read_text())
    by_image = {}
    for ann in annotations["annotations"]:
        if ann["num_keypoints"] >= 6 and not ann["iscrowd"] and ann["area"] > 4096:
            by_image.setdefault(ann["image_id"], []).append(ann)
    selected = [im for im in sorted(annotations["images"], key=lambda im: im["id"]) if im["id"] in by_image][:args.images]

    def download(im):
        path = data / im["file_name"]
        url = f"https://s3.amazonaws.com/images.cocodataset.org/val2017/{im['file_name']}"
        if not path.exists():
            path.write_bytes(fetch(url))
        return {"id": im["id"], "url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with ThreadPoolExecutor(max_workers=8) as pool:
        manifest = list(pool.map(download, selected))
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2))
    from ultralytics import YOLO
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    images = [cv2.imread(str(data / im["file_name"])) for im in selected]
    device = 0 if torch.cuda.is_available() else "cpu"
    results = []
    for filename, size in [("yolo11n-pose.pt", 320), ("yolo11n-pose.pt", 640), ("yolo11s-pose.pt", 640), ("yolo11m-pose.pt", 640)]:
        weights = output / filename
        if not weights.exists():
            weights.write_bytes(fetch(f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{filename}"))
        model = YOLO(str(weights))
        for _ in range(8):
            model.predict(images[0], imgsz=size, device=device, verbose=False)
        latencies, correct, total, oks, matched, persons = [], 0, 0, [], 0, 0
        for im, pixels in zip(selected, images):
            if device != "cpu":
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = model.predict(pixels, imgsz=size, device=device, verbose=False, conf=.25)[0]
            if device != "cpu":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
            boxes = result.boxes.xyxy.cpu().numpy()
            kp = result.keypoints.xy.cpu().numpy()
            used = set()
            for ann in sorted(by_image[im["id"]], key=lambda ann: -ann["area"]):
                persons += 1
                gt = np.asarray(ann["keypoints"]).reshape(17, 3)
                visible = gt[:, 2] > 0
                total += visible.sum()
                x, y, w, h = ann["bbox"]
                box = np.array([x, y, x + w, y + h])
                options = [(iou(box, b), i) for i, b in enumerate(boxes) if i not in used]
                if not options or max(options)[0] < .5:
                    oks.append(0)
                    continue
                overlap, index = max(options)
                used.add(index)
                matched += 1
                distance = np.linalg.norm(kp[index] - gt[:, :2], axis=1)
                correct += (distance[visible] <= .2 * max(w, h)).sum()
                sigmas = np.asarray([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10
                oks.append(np.exp(-distance[visible] ** 2 / (2 * (2 * sigmas[visible]) ** 2 * ann["area"])).mean())
        results.append({"model": filename, "imgsz": size, "images": len(images), "persons": persons,
                        "matched_persons": matched, "visible_keypoints": int(total), "pck_bbox_02": correct / total,
                        "mean_oks_including_misses": float(np.mean(oks)), "device": str(device),
                        "latency_p50_ms": float(np.median(latencies)), "latency_p95_ms": float(np.quantile(latencies, .95)),
                        "parameters": sum(p.numel() for p in model.model.parameters()),
                        "weight_mib": weights.stat().st_size / (1024 ** 2), "sha256": hashlib.sha256(weights.read_bytes()).hexdigest()})
        print(results[-1], flush=True)
        del model
    (ROOT / "reports/pose_comparison.json").write_text(json.dumps(results, indent=2))
    best = max(r["pck_bbox_02"] for r in results)
    chosen = min([r for r in results if r["pck_bbox_02"] >= best - .03], key=lambda r: r["latency_p50_ms"])
    (output / "selected_pose.json").write_text(json.dumps({"model": chosen["model"], "imgsz": chosen["imgsz"],
        "selection_rule": "maximum PCK minus at most 3 percentage points, minimum observed GPU p50"}, indent=2))
    print("SELECTED POSE", chosen["model"], chosen["imgsz"], flush=True)


if __name__ == "__main__":
    main()
