"""Generate report figures locally; only count-derived charts belong in GitHub."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import cv2 as cv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis_pipeline import BASE, EXPERIMENTS, count, decode, intensity, read_rows

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def panel(ax, image, title, labels=False):
    if labels:
        ax.imshow(np.ma.masked_equal(image, 0), cmap="turbo", interpolation="nearest")
    else:
        low, high = np.percentile(image, [1, 99.7])
        ax.imshow(image, cmap="gray", vmin=low, vmax=max(high, low+1))
    ax.set(title=title, xticks=[], yticks=[])
    ax.set_axis_off()


def save(fig, out, name):
    fig.savefig(out / (name+".png"), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def overlay(ax, image, labels, title):
    panel(ax, image, title)
    ax.imshow(np.ma.masked_equal(labels, 0), cmap="turbo", alpha=.45, interpolation="nearest")


def generate(dataset, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_rows("selected_frames.csv")
    results = read_rows("test_results.csv")
    config = json.loads(Path("final_pipeline.json").read_text(encoding="utf-8"))
    final = config["final"]
    with ZipFile(dataset) as archive:
        development = [r for r in rows if r["group"] == "development"]
        sample = decode(archive, development[4]["archive_path"])
        fig, axes = plt.subplots(2, 3, figsize=(9, 4.9), layout="constrained")
        for i, (mode, title) in enumerate([("raw","Orijinal"),("stretch","Yüzdelik germe"),("clahe","CLAHE 2.0 • 8×8")]):
            transformed = intensity(sample, mode)
            panel(axes[0,i], transformed, title)
            axes[1,i].hist(transformed.ravel(), bins=80, color="#126c83", log=True)
            axes[1,i].set(xlabel="Piksel yoğunluğu", ylabel="Piksel sayısı" if i == 0 else "")
        save(fig, out, "intensity")
        fig, axes = plt.subplots(2, 4, figsize=(10, 5.1), layout="constrained")
        for ax, (key, title) in zip(axes.flat, [
            ("E09","Sabit eşik"),("E04","Otsu"),("E10","Adaptif 101, C=-3"),("E18","Watershed r=12"),
            ("E11","Erozyon 3×3"),("E12","Genişletme 3×3"),("E13","Açma 3×3"),("E14","Kapama 3×3")]):
            total, stages = count(sample, EXPERIMENTS[key], True)
            overlay(ax, sample, stages[-1], f"{title} | {total}")
        save(fig, out, "methods")
        total, stages = count(sample, final, True)
        fig, axes = plt.subplots(2, 3, figsize=(8.5, 5), layout="constrained")
        for ax, data, title in zip(axes.flat, [sample, *stages[:-1]],
                                  ["Girdi", "Yoğunluk", "Filtre", "Eşik maskesi", "Morfoloji"]):
            panel(ax, data, title)
        overlay(axes.flat[-1], sample, stages[-1], f"Son etiketler | {total} hücre")
        save(fig, out, "stages")
        fig, axes = plt.subplots(2, 5, figsize=(11, 4.8), layout="constrained")
        for ax, row in zip(axes.flat, development):
            image = decode(archive, row["archive_path"])
            total, stages = count(image, final, True)
            overlay(ax, image, stages[-1], f"Kare {row['frame']} | {total}")
        save(fig, out, "development_review")
        refs = read_rows("development_validation_references.csv")
        reviewed = [r for r in rows if r["group"] != "test"]
        fig, axes = plt.subplots(3, 5, figsize=(11, 6.8), layout="constrained")
        for ax, row, record in zip(axes.flat, reviewed, refs):
            image = decode(archive, row["archive_path"])
            mask_path = Path(row["archive_path"]).parent.parent / "01_GT" / "TRA" / f"man_track{int(row['frame']):03d}.tif"
            labels = decode(archive, mask_path.as_posix())
            overlay(ax, image, labels, f"Kare {row['frame']} | {record['reference']} kimlik")
        save(fig, out, "reference_review")
        success = min(results, key=lambda r: int(r["final_abs_error"]))
        failure = max(results, key=lambda r: int(r["final_abs_error"]))
        challenging = max((r for r in results if r not in (success, failure)),
                          key=lambda r: int(r["baseline_abs_error"]))
        fig, axes = plt.subplots(2, 3, figsize=(9, 5.3), layout="constrained")
        for col, (result, category) in enumerate(zip([success, failure, challenging],
                                                     ["En düşük hata", "En yüksek hata", "Zor örnek"])):
            row = next(r for r in rows if r["group"] == "test" and r["frame"] == result["frame"])
            image = decode(archive, row["archive_path"])
            for axis, cfg, method in zip(axes[:,col], [BASE, final], ["Baseline", "Nihai"]):
                total, stages = count(image, cfg, True)
                overlay(axis, image, stages[-1], f"{category} • {row['frame']}\n{method} {total} | ref. {result['reference']}")
        save(fig, out, "examples")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), layout="constrained", sharey=True)
    for i, ax in enumerate(axes, 1):
        subset = [r for r in results if int(r["block"]) == i]
        x = [int(r["frame"]) for r in subset]
        for column, title, color, marker in [
            ("reference","Referans","#252e3d","o"),("baseline","Baseline","#c15932","s"),("final","Nihai","#007e85","^")]:
            ax.plot(x, [int(r[column]) for r in subset], label=title, color=color, marker=marker)
        ax.set(title=f"Blok {i}", xlabel="Kare numarası", xticks=x)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Hücre sayısı")
    axes[-1].legend(loc="best", fontsize=8)
    save(fig, out, "temporal_counts")

    fig, ax = plt.subplots(figsize=(10, 1.9))
    ax.set_axis_off()
    nodes = ["TIFF", final["intensity"], final["filter"], final["segment"],
             final["morph"] or "Morfoloji yok",
             "Watershed" if final["split"] else "8-komşuluk", f"Alan ≥ {final['area']}\nSayım"]
    positions = np.linspace(.06, .94, len(nodes))
    for i, (x, label) in enumerate(zip(positions, nodes)):
        ax.text(x, .5, label, ha="center", va="center", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=.4", fc="#e6f2f4", ec="#44828b"))
        if i:
            ax.annotate("", xy=(x-.054,.5), xytext=(positions[i-1]+.054,.5), xycoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="->", color="#44828b"))
    save(fig, out, "flow")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("output")
    args = parser.parse_args()
    generate(args.dataset, args.output)

