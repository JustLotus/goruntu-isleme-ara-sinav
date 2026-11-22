"""Select and record the fixed experiment split without reading annotations."""
import csv
from pathlib import Path
import random
import re
import sys
from zipfile import ZipFile

SEED = 31063
GAP = 5


def select(archive):
    rng = random.Random(SEED)
    candidates = {"01": {}, "02": {}}
    for name in archive.namelist():
        match = re.fullmatch(r"(?:[^/]+/)*(01|02)/t(\d+)\.tif", name)
        if match:
            sequence, frame = match.groups()
            candidates[sequence][int(frame)] = name
    first, second = (sorted(candidates[seq]) for seq in ("01", "02"))
    for _ in range(10_000):
        validation = sorted(rng.sample(first, 5))
        eligible = [i for i in first if all(abs(i - j) > GAP for j in validation)]
        if len(eligible) >= 10:
            development = sorted(rng.sample(eligible, 10))
            break
    else:
        raise RuntimeError("Could not select disjoint development and validation frames")

    starts = [i for i in second if all(j in candidates["02"] for j in range(i, i + 5))]
    for _ in range(10_000):
        blocks = sorted(rng.sample(starts, 3))
        if all(b - (a + 4) > GAP for a, b in zip(blocks, blocks[1:])):
            break
    else:
        raise RuntimeError("Could not select separated test blocks")

    return [
        {"group": group, "sequence": seq, "block": block, "frame": frame,
         "filename": Path(candidates[seq][frame]).name,
         "archive_path": candidates[seq][frame], "seed": SEED}
        for group, seq, block, frames in [
            ("development", "01", "", development), ("validation", "01", "", validation),
            *[("test", "02", b, range(i, i + 5)) for b, i in enumerate(blocks, 1)]
        ]
        for frame in frames
    ]


def save_selection(archive_path, output="selected_frames.csv"):
    with ZipFile(archive_path) as archive:
        rows = select(archive)
    with open(output, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    save_selection(sys.argv[1])

