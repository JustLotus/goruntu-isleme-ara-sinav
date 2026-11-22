"""Classical counting experiments. Training, prediction and scoring are separate."""
import csv
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile
import argparse

import cv2 as cv
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

cv.setNumThreads(1)
BASE = dict(intensity="raw", filter="none", segment="otsu", morph="", kernel=3,
            area=0, split=0, threshold=35, adaptive_block=101, adaptive_c=-3)
CORE = BASE | dict(filter="g5", area=100)
EXPERIMENTS = {
    "E01": BASE, "E02": BASE | dict(area=100),
    "E03": CORE | dict(filter="g3"), "E04": CORE,
    "E05": CORE | dict(filter="median5"), "E06": CORE | dict(filter="sharp"),
    "E07": CORE | dict(intensity="stretch"), "E08": CORE | dict(intensity="clahe"),
    "E09": CORE | dict(segment="fixed"), "E10": CORE | dict(segment="adaptive"),
    "E11": CORE | dict(morph="erosion"), "E12": CORE | dict(morph="dilation"),
    "E13": CORE | dict(morph="opening"), "E14": CORE | dict(morph="closing"),
    "E15": CORE | dict(morph="opening,closing"), "E16": CORE | dict(morph="closing,opening"),
    "E17": CORE | dict(morph="opening,closing", kernel=7),
    "E18": CORE | dict(split=12), "E19": CORE | dict(area=250),
    "E20": CORE | dict(morph="opening", kernel=7),
}
DELTAS = {"E01": "", "E02": "E01", "E03": "E02", "E04": "E03",
          **{f"E{i:02}": "E04" for i in range(5, 15)},
          "E15": "E13", "E16": "E15", "E17": "E15", "E18": "E04",
          "E19": "E04", "E20": "E13"}


def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def decode(archive, member):
    image = cv.imdecode(np.frombuffer(archive.read(member), np.uint8), cv.IMREAD_UNCHANGED)
    if image is None or image.ndim != 2:
        raise ValueError(f"Expected grayscale TIFF: {member}")
    return image


def reference(archive, row):
    path = Path(row["archive_path"])
    member = path.parent.parent / f"{row['sequence']}_GT" / "TRA" / f"man_track{int(row['frame']):03d}.tif"
    labels = np.unique(decode(archive, member.as_posix()))
    return int(np.count_nonzero(labels > 0))


def intensity(image, mode):
    if mode == "raw":
        return image
    lo, hi = np.percentile(image, [1, 99.5])
    if mode == "stretch":
        return np.uint8(np.clip((image.astype(np.float32) - lo) / max(hi-lo, 1) * 255, 0, 255))
    return cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)


def scale8(image):
    if image.dtype == np.uint8:
        return image
    return np.round(image.astype(np.float32) * (255 / np.iinfo(image.dtype).max)).astype(np.uint8)


def count(image, cfg, inspect: bool = False):
    transformed = intensity(image, cfg["intensity"])
    filtered = transformed
    mode = cfg["filter"]
    if mode in {"g3", "g5"}:
        size = int(mode[1:])
        filtered = cv.GaussianBlur(transformed, (size, size), 0)
    elif mode == "median5":
        filtered = cv.medianBlur(transformed, 5)
    elif mode == "sharp":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        filtered = cv.filter2D(transformed, -1, kernel)

    if cfg["segment"] == "otsu":
        _, thresholded = cv.threshold(filtered, 0, 1, cv.THRESH_BINARY | cv.THRESH_OTSU)
    elif cfg["segment"] == "fixed":
        thresholded = scale8(filtered) > cfg["threshold"]
    else:
        thresholded = cv.adaptiveThreshold(scale8(filtered), 1, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv.THRESH_BINARY, cfg["adaptive_block"], cfg["adaptive_c"])
    binary = thresholded.astype(np.uint8)
    operations = {"erosion": cv.MORPH_ERODE, "dilation": cv.MORPH_DILATE,
                  "opening": cv.MORPH_OPEN, "closing": cv.MORPH_CLOSE}
    if cfg["morph"]:
        size = cfg["kernel"]
        element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size))
        for op in cfg["morph"].split(","):
            binary = cv.morphologyEx(binary, operations[op], element)
    _, labels, stats, _ = cv.connectedComponentsWithStats(binary, connectivity=8)
    keep = stats[:, cv.CC_STAT_AREA] >= cfg["area"]
    keep[0] = False
    if cfg["split"]:
        cleaned = keep[labels]
        distance = ndi.distance_transform_edt(cleaned)
        radius = cfg["split"]
        maxima = (distance == ndi.maximum_filter(distance, size=2*radius+1)) & (distance >= radius)
        components_with_peak = np.unique(labels[maxima])
        for component in np.flatnonzero(keep):
            if component not in components_with_peak:
                pixels = np.flatnonzero(labels == component)
                maxima.flat[pixels[np.argmax(distance.flat[pixels])]] = True
        markers, _ = ndi.label(maxima, structure=np.ones((3, 3)))
        labels = watershed(-distance, markers, mask=cleaned)
        keep = np.bincount(labels.ravel()) >= cfg["area"]
        keep[0] = False
    total = int(keep.sum())
    return (total, (transformed, filtered, thresholded, binary, np.where(keep[labels], labels, 0))) if inspect else total


def estimate(image, config):
    times = []
    for _ in range(3):
        start = perf_counter()
        prediction = count(image, config)
        times.append((perf_counter()-start)*1000)
    return prediction, float(np.median(times))


def metrics(actual, predicted, times):
    errors = np.asarray(predicted) - np.asarray(actual)
    return dict(mae=float(np.mean(abs(errors))), rmse=float(np.sqrt(np.mean(errors**2))),
                bias=float(np.mean(errors)), exact_accuracy=float(np.mean(errors == 0)),
                mean_ms=float(np.mean(times)))


def evaluate(images, refs, cfg):
    predictions, times = zip(*(estimate(image, cfg) for image in images))
    return metrics(refs, predictions, times)


def develop(archive, rows):
    groups = {}
    records = []
    for group in ("development", "validation"):
        selected = [row for row in rows if row["group"] == group]
        images = [decode(archive, row["archive_path"]) for row in selected]
        refs = [reference(archive, row) for row in selected]
        records.extend(dict(group=group, frame=row["frame"], reference=ref,
                            shape="x".join(map(str, image.shape)), dtype=image.dtype.name,
                            minimum=int(image.min()), maximum=int(image.max()),
                            p01=np.percentile(image, 1), p50=np.percentile(image, 50),
                            p995=np.percentile(image, 99.5))
                       for row, image, ref in zip(selected, images, refs))
        groups[group] = (images, refs)
    write_rows("development_validation_references.csv", records)
    results = [dict(experiment=key, comparator=DELTAS[key], **cfg,
                    **evaluate(*groups["development"], cfg))
               for key, cfg in EXPERIMENTS.items()]
    write_rows("experiments.csv", results)
    # Top three non-baseline configurations by development MAE, RMSE, then ID.
    candidate_ids = [r["experiment"] for r in sorted(results[1:],
                     key=lambda r: (r["mae"], r["rmse"], r["experiment"]))[:3]]
    comparison = [dict(experiment=key, **evaluate(*groups["validation"], EXPERIMENTS[key]))
                  for key in candidate_ids]
    def complexity(key):
        cfg = EXPERIMENTS[key]
        return sum(bool(cfg[k]) for k in ("morph", "area", "split")) + sum(
            cfg[k] != BASE[k] for k in ("intensity", "filter", "segment"))
    winner = min(comparison, key=lambda r: (r["mae"], r["rmse"], r["mean_ms"],
                                           complexity(r["experiment"]), r["experiment"]))
    for item in comparison:
        item["selected"] = int(item is winner)
    write_rows("validation_results.csv", comparison)
    key = winner["experiment"]
    return {"id": key, "baseline": BASE, "final": EXPERIMENTS[key]}


def predict(archive, rows, configs):
    records = []
    for row in rows:
        image = decode(archive, row["archive_path"])
        base, base_ms = estimate(image, configs["baseline"])
        final, final_ms = estimate(image, configs["final"])
        records.append(dict(block=row["block"], frame=row["frame"], filename=row["filename"],
                            baseline=base, final=final, baseline_ms=base_ms, final_ms=final_ms))
    write_rows("test_predictions_locked.csv", records)


def score(archive, rows, prediction_path="test_predictions_locked.csv"):
    predictions = read_rows(prediction_path)
    keys = ("block", "frame", "filename")
    if [tuple(r[k] for k in keys) for r in rows] != [tuple(r[k] for k in keys) for r in predictions]:
        raise ValueError("Locked predictions do not match the fixed test split")
    results = []
    for row, pred in zip(rows, predictions):
        actual = reference(archive, row)
        results.append(dict(**pred, reference=actual,
                            baseline_abs_error=abs(int(pred["baseline"])-actual),
                            final_abs_error=abs(int(pred["final"])-actual)))
    write_rows("test_results.csv", results)
    summary = [dict(method=method, **metrics(
        [r["reference"] for r in results], [int(r[method]) for r in results],
        [float(r[f"{method}_ms"]) for r in results])) for method in ("baseline", "final")]
    write_rows("test_summary.csv", summary)


def run(stage, dataset):
    rows = read_rows("selected_frames.csv")
    test = [row for row in rows if row["group"] == "test"]
    with ZipFile(dataset) as archive:
        if stage == "develop":
            configs = develop(archive, rows)
            Path("final_pipeline.json").write_text(json.dumps(configs, indent=2)+"\n", encoding="utf-8")
        elif stage == "predict":
            configs = json.loads(Path("final_pipeline.json").read_text(encoding="utf-8"))
            predict(archive, test, configs)
        elif stage == "score":
            score(archive, test)
        else:
            raise ValueError(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["develop", "predict", "score"])
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    run(args.stage, args.dataset)

